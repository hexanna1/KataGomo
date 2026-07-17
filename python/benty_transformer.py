"""Packed Bent-Y attention with symmetry-invariant graph relations."""

from collections import deque
from functools import lru_cache

import torch
from torch.nn import functional as F

from benty import build_topology
from transformer import FeedForward, NestedBlock, norm_act_conv


@lru_cache(maxsize=None)
def graph_geometry(pos_len):
    topologies = [build_topology(size) for size in range(7, pos_len + 1, 6)]
    num_tokens = len(topologies[-1].playable)
    gather = torch.zeros(len(topologies), num_tokens, dtype=torch.long)
    reverse = torch.full((len(topologies), pos_len * pos_len), num_tokens, dtype=torch.long)
    features = torch.zeros(len(topologies), num_tokens, num_tokens, 10)
    counts = torch.tensor([len(topology.playable) for topology in topologies])
    for shape, topology in enumerate(topologies):
        points = topology.playable
        compact = {point: index for index, point in enumerate(points)}
        distances = []
        for start in points:
            distance = {start: 0}
            pending = deque([start])
            while pending:
                point = pending.popleft()
                for neighbor in topology.neighbors[point]:
                    if neighbor not in distance:
                        distance[neighbor] = distance[point] + 1
                        pending.append(neighbor)
            distances.append([distance[point] for point in points])
        corners = [next(point for point in points if topology.side_masks[point] == side)
                   for side in (5, 3, 6)]
        corner_distances = [distances[compact[point]] for point in corners]
        diameter = corner_distances[0][compact[corners[1]]]
        for query, point in enumerate(points):
            y, x = divmod(point, topology.tensor_len)
            canvas_point = y * pos_len + x
            gather[shape, query] = canvas_point
            reverse[shape, canvas_point] = query
            for key in range(len(points)):
                distance = distances[query][key]
                pairs = sorted((values[query], values[key]) for values in corner_distances)
                features[shape, query, key] = torch.tensor([
                    1.0 / (1.0 + distance), float(distance == 0),
                    float(distance == 1), float(distance == 2),
                    *(value / diameter for pair in pairs for value in pair),
                ])
    return gather, reverse, features, counts


class GraphPacking(torch.nn.Module):
    """Pack the pointwise network once and restore its spatial output heads."""
    def __init__(self, pos_len):
        super().__init__()
        gather, reverse, _, counts = graph_geometry(pos_len)
        for key, value in (("gather", gather), ("reverse", reverse), ("counts", counts)):
            self.register_buffer(key, value, persistent=False)
        self.register_buffer("token_positions", torch.arange(gather.shape[1]), persistent=False)

    def pack(self, spatial):
        batch, channels, _, _ = spatial.shape
        mask_sum = spatial[:, 0].sum(dim=(1, 2))
        shape = (mask_sum[:, None] == self.counts[None, :]).long().argmax(dim=1)
        indices = self.gather[shape]
        token_mask = self.token_positions[None, :] < self.counts[shape, None]
        packed = spatial.flatten(2).gather(2, indices[:, None, :].expand(-1, channels, -1))
        return (packed * token_mask[:, None, :]).unsqueeze(2), shape

    def unpack(self, outputs, shape, spatial_size):
        reverse = self.reverse[shape]
        height, width = spatial_size

        def restore(value, fill=0.0):
            value = value.flatten(2)
            batch, channels, _ = value.shape
            value = torch.cat((value, value.new_full((batch, channels, 1), fill)), dim=2)
            return value.gather(2, reverse[:, None, :].expand(-1, channels, -1)).reshape(batch, channels, height, width)

        restored = []
        for head in outputs:
            policy = torch.cat((restore(head[0][:, :, :-1], -5000.0).flatten(2), head[0][:, :, -1:]), dim=2)
            restored.append((policy, *head[1:4], *(restore(value) for value in head[4:8]), head[8]))
        return tuple(restored)


class GraphAttention(torch.nn.Module):
    def __init__(self, name, c_main, config):
        super().__init__()
        self.name = name
        self.num_heads = config["transformer_heads"]
        self.num_kv_heads = self.num_heads
        self.q_head_dim = c_main // self.num_heads
        self.v_head_dim = self.q_head_dim
        self.norm1 = torch.nn.RMSNorm(c_main, eps=1e-6)
        self.q_proj = torch.nn.Linear(c_main, c_main, bias=False)
        self.k_proj = torch.nn.Linear(c_main, c_main, bias=False)
        self.v_proj = torch.nn.Linear(c_main, c_main, bias=False)
        self.out_proj = torch.nn.Linear(c_main, c_main, bias=False)
        self.graph_bias_input = torch.nn.Linear(10, 16, bias=False)
        self.graph_bias_output = torch.nn.Linear(16, self.num_heads, bias=False)

    def add_reg_dict(self, groups):
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj,
                           self.graph_bias_input, self.graph_bias_output):
            groups["normal_attn"].append(projection.weight)
        groups["noreg"].append(self.norm1.weight)

    def forward(self, x, token_mask, features, shape):
        batch, tokens, channels = x.shape
        q, k, v = F.linear(self.norm1(x), torch.cat([
            self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
        ])).chunk(3, dim=-1)
        q, k, v = [value.reshape(batch, tokens, self.num_heads, self.q_head_dim).transpose(1, 2)
                   for value in (q, k, v)]
        bias = self.graph_bias_output(F.silu(self.graph_bias_input(features)))
        bias = bias.permute(0, 3, 1, 2)[shape]
        bias = bias.masked_fill(~token_mask[:, None, None, :], float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out_proj(out.transpose(1, 2).reshape(batch, tokens, channels))


class GraphNestedBlock(NestedBlock):
    def __init__(self, name, config, pos_len, activation, internal_length=2):
        torch.nn.Module.__init__(self)
        self.name = name
        self.norm_kind = config["norm_kind"]
        self.internal_length = internal_length
        c_main, c_mid = config["trunk_num_channels"], config["mid_num_channels"]
        self.normactconvp = norm_act_conv(c_main, c_mid, config, activation, fixup_use_gamma=False)
        self.blockstack = torch.nn.ModuleList()
        for index in range(self.internal_length):
            self.blockstack.append(GraphAttention(name + ".blockstack.attn" + str(index + 1), c_mid, config))
            self.blockstack.append(FeedForward(name + ".blockstack.ffn" + str(index + 1), c_mid, config))
        self.normactconvq = norm_act_conv(c_mid, c_main, config, activation, fixup_use_gamma=True)
        _, _, features, counts = graph_geometry(pos_len)
        for key, value in (("features", features), ("counts", counts)):
            self.register_buffer(key, value, persistent=False)

    def forward(self, x, mask, mask_sum_hw, mask_sum):
        out = self.normactconvp(x, mask, mask_sum_hw, mask_sum)
        batch = out.shape[0]
        shape = (mask_sum_hw.reshape(batch, 1) == self.counts[None, :]).long().argmax(dim=1)
        seq = out.squeeze(2).transpose(1, 2)
        token_mask = mask.flatten(1).bool()
        for index in range(0, len(self.blockstack), 2):
            attention, ffn = self.blockstack[index:index + 2]
            seq = seq + attention(seq, token_mask, self.features, shape)
            seq = seq + ffn(seq)
        out = seq.transpose(1, 2).unsqueeze(2)
        return x + self.normactconvq(out, mask, mask_sum_hw, mask_sum)
