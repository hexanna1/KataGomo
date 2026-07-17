"""Nested transformer blocks with coordinates for each game's board geometry.

Adapted from KataGo and HZY's KataGo_Transformer. See transformer.LICENSE.
"""
import math

import torch
from torch.nn import functional as F

from model_pytorch import NormActConv


class FixedScaleNorm(torch.nn.Module):
    """Masked affine transform with an optional learned offset from unit gamma."""
    def __init__(self, c_in, config, fixup_use_gamma):
        super().__init__()
        self.c_in = c_in
        self.epsilon = config["bnorm_epsilon"]
        self.is_using_batchnorm = False
        self.gamma_weight_decay_center_1 = config.get("gamma_weight_decay_center_1", False)
        self.scale = None
        self.gamma = None
        self.beta = torch.nn.Parameter(torch.zeros(1, c_in, 1, 1))
        if config.get("bnorm_use_gamma", False) or fixup_use_gamma:
            gamma = torch.zeros(1, c_in, 1, 1) if self.gamma_weight_decay_center_1 else torch.ones(1, c_in, 1, 1)
            self.gamma = torch.nn.Parameter(gamma)

    def set_scale(self, scale):
        self.scale = scale

    def add_reg_dict(self, groups):
        if self.gamma is not None:
            groups["normal_gamma"].append(self.gamma)
        groups["noreg"].append(self.beta)

    def set_brenorm_params(self, renorm_avg_momentum, rmax, dmax):
        pass

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        pass

    def forward(self, x, mask, mask_sum):
        gamma = self.gamma
        if gamma is not None and self.gamma_weight_decay_center_1:
            gamma = gamma + 1.0
        if self.scale is not None:
            out = x * (gamma * self.scale if gamma is not None else self.scale) + self.beta
        else:
            out = x * gamma + self.beta if gamma is not None else x + self.beta
        return out * mask


class RMSNormMask(torch.nn.Module):
    """RMS normalization over channels, optionally including on-board positions."""
    def __init__(self, c_in, spatial):
        super().__init__()
        self.c_in = c_in
        self.spatial = spatial
        self.eps = 1e-6
        if spatial:
            self.norm = None
            self.gamma = torch.nn.Parameter(torch.ones(c_in))
        else:
            self.norm = torch.nn.RMSNorm(c_in, eps=self.eps)
        self.beta = torch.nn.Parameter(torch.zeros(c_in))

    def set_scale(self, scale):
        pass

    def add_reg_dict(self, groups):
        groups["output"].append(self.gamma if self.spatial else self.norm.weight)
        groups["output"].append(self.beta)

    def set_brenorm_params(self, renorm_avg_momentum, rmax, dmax):
        pass

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        pass

    def forward(self, x, mask, mask_sum_hw, mask_sum):
        if self.spatial:
            mean_sq = torch.sum(x * x * mask, dim=(1, 2, 3), keepdim=True) / (self.c_in * mask_sum_hw + self.eps)
            out = x / torch.sqrt(mean_sq + self.eps)
            return (out * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)) * mask
        out = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return (out + self.beta.view(1, -1, 1, 1)) * mask


def rotary(x, cos, sin):
    """Rotate channel pairs, retaining the projection dtype under autocast."""
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0, x1 = pairs.unbind(dim=-1)
    cos = cos.unsqueeze(0).to(dtype=x.dtype)
    sin = sin.unsqueeze(0).to(dtype=x.dtype)
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.reshape_as(x)


class Attention(torch.nn.Module):
    def __init__(self, name, c_main, config):
        super().__init__()
        self.name = name
        self.num_heads = config["transformer_heads"]
        self.num_kv_heads = self.num_heads
        self.q_head_dim = c_main // self.num_heads
        self.v_head_dim = self.q_head_dim
        self.q_proj = torch.nn.Linear(c_main, c_main, bias=False)
        self.k_proj = torch.nn.Linear(c_main, c_main, bias=False)
        self.v_proj = torch.nn.Linear(c_main, c_main, bias=False)
        self.out_proj = torch.nn.Linear(c_main, c_main, bias=False)
        shape = (self.num_heads, self.q_head_dim // 2, 2)
        freqs = torch.exp(torch.empty(shape).uniform_(math.log(1.0 / 50.0), math.log(1.0)))
        freqs = freqs * (torch.randint(0, 2, shape) * 2 - 1).float()
        self.rope_freqs = torch.nn.Parameter(freqs)
        self.norm1 = torch.nn.RMSNorm(c_main, eps=1e-6)

    def add_reg_dict(self, groups):
        groups["noreg"].append(self.rope_freqs)
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            groups["normal_attn"].append(projection.weight)
        groups["noreg"].append(self.norm1.weight)

    def forward(self, x, mask, xy):
        b, s, _ = x.shape
        h, d = self.num_heads, self.q_head_dim
        q, k, v = F.linear(self.norm1(x), torch.cat([
            self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
        ])).chunk(3, dim=-1)
        q, k, v = [t.reshape(b, s, h, d) for t in (q, k, v)]
        angles = xy[:, 0, None, None] * self.rope_freqs[:, :, 0] + xy[:, 1, None, None] * self.rope_freqs[:, :, 1]
        cos, sin = torch.cos(angles), torch.sin(angles)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :].bool())
        return self.out_proj(out.transpose(1, 2).reshape(b, s, h * d))


class FeedForward(torch.nn.Module):
    def __init__(self, name, c_main, config):
        super().__init__()
        self.name = name
        self.c_main = c_main
        self.ffn_dim = config["transformer_ffn_channels"]
        self.ffn_linear1 = torch.nn.Linear(c_main, self.ffn_dim, bias=False)
        self.ffn_linear_gate = torch.nn.Linear(c_main, self.ffn_dim, bias=False)
        self.ffn_act = torch.nn.SiLU()
        self.ffn_linear2 = torch.nn.Linear(self.ffn_dim, c_main, bias=False)
        self.norm = torch.nn.RMSNorm(c_main, eps=1e-6)

    def add_reg_dict(self, groups):
        for projection in (self.ffn_linear1, self.ffn_linear_gate, self.ffn_linear2):
            groups["normal"].append(projection.weight)
        groups["noreg"].append(self.norm.weight)

    def forward(self, x):
        weights = torch.cat([self.ffn_linear1.weight, self.ffn_linear_gate.weight])
        value, gate = F.linear(self.norm(x), weights).split(self.ffn_dim, dim=-1)
        return self.ffn_linear2(self.ffn_act(value) * gate)


def norm_act_conv(c_in, c_out, config, activation, fixup_use_gamma):
    layer = NormActConv(c_in, c_out, None, config, activation, kernel_size=1, fixup_use_gamma=fixup_use_gamma)
    layer.norm = FixedScaleNorm(c_in, config, fixup_use_gamma)
    return layer


class NestedBlock(torch.nn.Module):
    def __init__(self, name, config, pos_len, game, packed, activation, internal_length=2):
        super().__init__()
        self.name = name
        self.norm_kind = config["norm_kind"]
        self.internal_length = internal_length
        c_main, c_mid = config["trunk_num_channels"], config["mid_num_channels"]
        self.normactconvp = norm_act_conv(c_main, c_mid, config, activation, fixup_use_gamma=False)
        self.blockstack = torch.nn.ModuleList()
        for i in range(internal_length):
            self.blockstack.append(Attention(name + ".blockstack.attn" + str(i + 1), c_mid, config))
            self.blockstack.append(FeedForward(name + ".blockstack.ffn" + str(i + 1), c_mid, config))
        self.normactconvq = norm_act_conv(c_mid, c_main, config, activation, fixup_use_gamma=True)
        self.packed = packed
        y, x = torch.meshgrid(torch.arange(pos_len), torch.arange(pos_len), indexing="ij")
        indices = torch.arange(pos_len * pos_len)
        if game == "y":
            indices = indices[(x + y < pos_len).flatten()]
        self.register_buffer("indices", indices, persistent=False)
        self.register_buffer("xy", torch.stack([x.flatten(), y.flatten()], dim=-1).float(), persistent=False)

    def initialize(self, fixup_scale):
        if self.norm_kind == "fixup":
            self.normactconvp.initialize(scale=math.pow(fixup_scale, 1.0 / (1.0 + self.internal_length)))
            self.normactconvq.initialize(scale=0.0)
        else:
            self.normactconvp.initialize(scale=1.0, norm_scale=fixup_scale)
            self.normactconvq.initialize(scale=1.0, norm_scale=1.0 / math.sqrt(self.internal_length + 1.0))

    def add_reg_dict(self, groups):
        groups = dict(groups, normal_attn=groups.get("normal_attn", groups["normal"]))
        self.normactconvp.add_reg_dict(groups)
        for block in self.blockstack:
            block.add_reg_dict(groups)
        self.normactconvq.add_reg_dict(groups)

    def set_brenorm_params(self, renorm_avg_momentum, rmax, dmax):
        pass

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        pass

    def forward(self, x, mask, mask_sum_hw, mask_sum):
        out = self.normactconvp(x, mask, mask_sum_hw, mask_sum)
        b, c, height, width = out.shape
        seq = out.flatten(2).transpose(1, 2)
        token_mask = mask.flatten(1)
        xy = self.xy
        if self.packed:
            seq = seq[:, self.indices]
            token_mask = token_mask[:, self.indices]
            xy = xy[self.indices]
        for i in range(0, len(self.blockstack), 2):
            attn, ffn = self.blockstack[i:i+2]
            seq = seq + attn(seq, token_mask, xy)
            seq = seq + ffn(seq)
        seq = seq * token_mask.unsqueeze(-1)
        if self.packed:
            seq = seq.new_zeros(b, height * width, c).index_copy(1, self.indices, seq)
        out = seq.transpose(1, 2).reshape(b, c, height, width)
        return x + self.normactconvq(out, mask, mask_sum_hw, mask_sum)
