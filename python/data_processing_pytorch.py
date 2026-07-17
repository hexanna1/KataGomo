import logging
import os

import numpy as np

import torch
import torch.nn.functional

import modelconfigs
import benty

def read_npz_training_data(
    npz_files,
    batch_size: int,
    world_size: int,
    rank: int,
    pos_len: int,
    device,
    randomize_symmetries: bool,
    model_config: modelconfigs.ModelConfig,
    board_shape: str = "y",
):
    rand = np.random.default_rng(seed=list(os.urandom(12)))
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    board_shape = normalize_board_shape(board_shape)

    for npz_file in npz_files:
        with np.load(npz_file) as npz:
            binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
            globalInputNC = npz["globalInputNC"]
            policyTargetsNCMove = npz["policyTargetsNCMove"].astype(np.float32)
            globalTargetsNC = npz["globalTargetsNC"]
            scoreDistrN = npz["scoreDistrN"].astype(np.float32)
            valueTargetsNCHW = npz["valueTargetsNCHW"].astype(np.float32)
        del npz

        binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked,axis=2)
        assert len(binaryInputNCHW.shape) == 3
        assert binaryInputNCHW.shape[2] == ((pos_len * pos_len + 7) // 8) * 8
        binaryInputNCHW = binaryInputNCHW[:,:,:pos_len*pos_len]
        binaryInputNCHW = np.reshape(binaryInputNCHW, (
            binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], pos_len, pos_len
        )).astype(np.float32)

        assert binaryInputNCHW.shape[1] == num_bin_features
        assert globalInputNC.shape[1] == num_global_features

        num_samples = binaryInputNCHW.shape[0]
        # Just discard stuff that doesn't divide evenly
        num_whole_steps = num_samples // (batch_size * world_size)

        #logging.info(f"Beginning {npz_file} with {num_whole_steps * world_size} usable batches, my rank is {rank}")
        for n in range(num_whole_steps):
            start = (n * world_size + rank) * batch_size
            end = start + batch_size

            batch_binaryInputNCHW = torch.from_numpy(binaryInputNCHW[start:end]).to(device)
            batch_globalInputNC = torch.from_numpy(globalInputNC[start:end]).to(device)
            batch_policyTargetsNCMove = torch.from_numpy(policyTargetsNCMove[start:end]).to(device)
            batch_globalTargetsNC = torch.from_numpy(globalTargetsNC[start:end]).to(device)
            batch_scoreDistrN = torch.from_numpy(scoreDistrN[start:end]).to(device)
            batch_valueTargetsNCHW = torch.from_numpy(valueTargetsNCHW[start:end]).to(device)

            if board_shape in ("y", "obtuseY", "bentY"):
                batch_board_lens = infer_board_lens_from_mask(binaryInputNCHW[start:end,0,:,:])
            else:
                batch_board_lens = None

            if randomize_symmetries:
                symm = int(rand.integers(0, 6))
                batch_binaryInputNCHW = apply_symmetry(batch_binaryInputNCHW, symm, board_shape, batch_board_lens)
                batch_policyTargetsNCMove = apply_symmetry_policy(batch_policyTargetsNCMove, symm, pos_len, board_shape, batch_board_lens)
                batch_valueTargetsNCHW = apply_symmetry(batch_valueTargetsNCHW, symm, board_shape, batch_board_lens)
            batch_binaryInputNCHW = batch_binaryInputNCHW.contiguous()
            batch_policyTargetsNCMove = batch_policyTargetsNCMove.contiguous()
            batch_valueTargetsNCHW = batch_valueTargetsNCHW.contiguous()

            batch = dict(
                binaryInputNCHW = batch_binaryInputNCHW,
                globalInputNC = batch_globalInputNC,
                policyTargetsNCMove = batch_policyTargetsNCMove,
                globalTargetsNC = batch_globalTargetsNC,
                scoreDistrN = batch_scoreDistrN,
                valueTargetsNCHW = batch_valueTargetsNCHW,
            )
            yield batch


def normalize_board_shape(board_shape):
    board_shape = board_shape.lower()
    if board_shape == "y":
        return "y"
    if board_shape == "obtusey":
        return "obtuseY"
    if board_shape == "benty":
        return "bentY"
    raise ValueError(f"Unknown board shape: {board_shape}")

def infer_board_lens_from_mask(mask_nhw):
    board_lens = []
    for mask in mask_nhw:
        ys, xs = np.nonzero(mask)
        if len(xs) <= 0:
            raise ValueError("Cannot infer board size from empty input mask")
        board_lens.append(int(max(np.max(xs), np.max(ys)) + 1))
    return board_lens

def apply_symmetry_policy(tensor, symm, pos_len, board_shape="y", board_lens=None):
    """Same as apply_symmetry but also handles the pass index"""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    tensor_without_pass = tensor[:,:,:-1].view((batch_size, channels, pos_len, pos_len))
    tensor_transformed = apply_symmetry(tensor_without_pass, symm, board_shape, board_lens)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len*pos_len),
        tensor[:,:,-1:]
    ), dim=2)

def apply_symmetry(tensor, symm, board_shape="y", board_lens=None):
    """
    Apply a shape-specific Y-board symmetry.

    Args:
        tensor (torch.Tensor): Tensor to transform. (..., W, W)
        symm (int): one of the 6 shape-coordinate symmetries.
    """
    assert tensor.shape[-1] == tensor.shape[-2]
    board_shape = normalize_board_shape(board_shape)

    if symm == 0:
        return tensor

    perms = (
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1),
        (1, 0, 2),
        (0, 2, 1),
        (2, 1, 0),
    )
    assert 0 <= symm < len(perms)

    pos_len = tensor.shape[-1]
    transformed = torch.zeros_like(tensor)
    perm = perms[symm]
    perm_inversions = sum(1 for i in range(3) for j in range(i + 1, 3) if perm[i] > perm[j])
    perm_sign = 1 if perm_inversions % 2 == 0 else -1

    def transform_group(batch_idxs, src_ys, src_xs, dst_ys, dst_xs):
        idx = torch.tensor(batch_idxs, device=tensor.device, dtype=torch.long)
        src_pos = torch.tensor(src_ys, device=tensor.device, dtype=torch.long) * pos_len + torch.tensor(src_xs, device=tensor.device, dtype=torch.long)
        dst_pos = torch.tensor(dst_ys, device=tensor.device, dtype=torch.long) * pos_len + torch.tensor(dst_xs, device=tensor.device, dtype=torch.long)
        selected = tensor.index_select(0, idx)
        src = selected.reshape(len(batch_idxs), -1, pos_len * pos_len)
        dst = torch.zeros_like(src)
        dst[:, :, dst_pos] = src[:, :, src_pos]
        transformed.index_copy_(0, idx, dst.reshape_as(selected))

    if board_shape == "y":
        if board_lens is None:
            board_lens = [pos_len] * tensor.shape[0]
        batches_by_len = {}
        for i, board_len in enumerate(board_lens):
            batches_by_len.setdefault(board_len, []).append(i)
        for board_len, batch_idxs in batches_by_len.items():
            src_ys = []
            src_xs = []
            dst_ys = []
            dst_xs = []
            for y in range(board_len):
                for x in range(board_len - y):
                    coords = (x, y, board_len - 1 - x - y)
                    sx = coords[perm[0]]
                    sy = coords[perm[1]]
                    src_ys.append(y)
                    src_xs.append(x)
                    dst_ys.append(sy)
                    dst_xs.append(sx)
            transform_group(batch_idxs, src_ys, src_xs, dst_ys, dst_xs)
    elif board_shape == "obtuseY":
        if board_lens is None:
            board_lens = [pos_len] * tensor.shape[0]
        batches_by_len = {}
        for i, board_len in enumerate(board_lens):
            batches_by_len.setdefault(board_len, []).append(i)
        for board_len, batch_idxs in batches_by_len.items():
            assert board_len % 2 == 1
            src_ys = []
            src_xs = []
            dst_ys = []
            dst_xs = []
            n = (board_len - 1) // 2
            for y in range(board_len):
                for x in range(board_len):
                    rx = board_len - 1 - x
                    ry = board_len - 1 - y
                    if (x < n and y < n and x + y < n) or (rx < n and ry < n and rx + ry < n):
                        continue
                    q = x - n
                    r = y - n
                    coords = (q, r, -q - r)
                    sx = perm_sign * coords[perm[0]] + n
                    sy = perm_sign * coords[perm[1]] + n
                    src_ys.append(y)
                    src_xs.append(x)
                    dst_ys.append(sy)
                    dst_xs.append(sx)
            transform_group(batch_idxs, src_ys, src_xs, dst_ys, dst_xs)
    elif board_shape == "bentY":
        if board_lens is None:
            board_lens = [pos_len] * tensor.shape[0]
        batches_by_len = {}
        for i, board_len in enumerate(board_lens):
            batches_by_len.setdefault(board_len, []).append(i)
        for board_len, batch_idxs in batches_by_len.items():
            topology = benty.build_topology(board_len)
            src_ys = []
            src_xs = []
            dst_ys = []
            dst_xs = []
            for pos in topology.playable:
                sym_pos = topology.sym_pos[pos][symm]
                src_ys.append(pos // board_len)
                src_xs.append(pos % board_len)
                dst_ys.append(sym_pos // board_len)
                dst_xs.append(sym_pos % board_len)
            transform_group(batch_idxs, src_ys, src_xs, dst_ys, dst_xs)
    else:
        raise AssertionError(f"Unhandled board shape: {board_shape}")
    return transformed
