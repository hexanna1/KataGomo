import logging
import os
from functools import lru_cache

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
    non_blocking = torch.device(device).type == "mps"
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

        # File tensors retain the NumPy storage while MPS copies are in flight.
        source_tensors = tuple(torch.from_numpy(array) for array in (
            binaryInputNCHW,
            globalInputNC,
            policyTargetsNCMove,
            globalTargetsNC,
            scoreDistrN,
            valueTargetsNCHW,
        ))
        try:
            for n in range(num_whole_steps):
                start = (n * world_size + rank) * batch_size
                end = start + batch_size

                (
                    batch_binaryInputNCHW,
                    batch_globalInputNC,
                    batch_policyTargetsNCMove,
                    batch_globalTargetsNC,
                    batch_scoreDistrN,
                    batch_valueTargetsNCHW,
                ) = (source[start:end].to(device, non_blocking=non_blocking) for source in source_tensors)

                batch_board_lens = None if board_shape == "hex" else infer_board_lens_from_mask(binaryInputNCHW[start:end,0,:,:])

                if randomize_symmetries:
                    num_symmetries = 2 if board_shape == "hex" else 12 if board_shape == "yavalath" else 6
                    symm = int(rand.integers(0, num_symmetries))
                    if symm != 0 and board_shape == "y":
                        indices = y_symmetry_indices(pos_len, symm, batch_board_lens, device)
                    elif symm != 0 and board_shape == "yavalath":
                        indices = yavalath_symmetry_indices(pos_len, symm, batch_board_lens, device)
                    elif symm != 0 and board_shape == "bentY":
                        indices = benty_symmetry_indices(pos_len, symm, batch_board_lens, device)
                    else:
                        indices = None
                    batch_binaryInputNCHW = apply_symmetry(batch_binaryInputNCHW, symm, board_shape, batch_board_lens, indices)
                    batch_policyTargetsNCMove = apply_symmetry_policy(batch_policyTargetsNCMove, symm, pos_len, board_shape, batch_board_lens, indices)
                    batch_valueTargetsNCHW = apply_symmetry(batch_valueTargetsNCHW, symm, board_shape, batch_board_lens, indices)
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
        finally:
            if non_blocking and num_whole_steps > 0:
                # Fence before releasing a file, including generator close or failure.
                torch.mps.synchronize()
            del source_tensors


def normalize_board_shape(board_shape):
    board_shape = board_shape.lower()
    if board_shape == "hex":
        return "hex"
    if board_shape == "y":
        return "y"
    if board_shape == "yavalath":
        return "yavalath"
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

@lru_cache(maxsize=256)
def y_symmetry_map(pos_len, board_len, symm):
    perm = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (1, 0, 2), (0, 2, 1), (2, 1, 0))[symm]
    # The extra source element is zero, so off-board outputs remain zero.
    indices = np.full(pos_len * pos_len, pos_len * pos_len, dtype=np.int64)
    for y in range(board_len):
        for x in range(board_len - y):
            coords = (x, y, board_len - 1 - x - y)
            indices[coords[perm[1]] * pos_len + coords[perm[0]]] = y * pos_len + x
    return indices


def y_symmetry_indices(pos_len, symm, board_lens, device):
    indices = np.stack([y_symmetry_map(pos_len, board_len, symm) for board_len in board_lens])
    return torch.from_numpy(indices).to(device).unsqueeze(1)


@lru_cache(maxsize=256)
def yavalath_symmetry_map(pos_len, board_len, symm):
    perm = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (1, 0, 2), (0, 2, 1), (2, 1, 0))[symm % 6]
    inversions = sum(1 for i in range(3) for j in range(i + 1, 3) if perm[i] > perm[j])
    sign = 1 if inversions % 2 == 0 else -1
    if symm >= 6:
        sign = -sign
    indices = np.full(pos_len * pos_len, pos_len * pos_len, dtype=np.int64)
    n = (board_len - 1) // 2
    for y in range(board_len):
        for x in range(board_len):
            rx = board_len - 1 - x
            ry = board_len - 1 - y
            if (x < n and y < n and x + y < n) or (rx < n and ry < n and rx + ry < n):
                continue
            coords = (x - n, y - n, 2 * n - x - y)
            sx = sign * coords[perm[0]] + n
            sy = sign * coords[perm[1]] + n
            indices[sy * pos_len + sx] = y * pos_len + x
    return indices


def yavalath_symmetry_indices(pos_len, symm, board_lens, device):
    indices = np.stack([yavalath_symmetry_map(pos_len, board_len, symm) for board_len in board_lens])
    return torch.from_numpy(indices).to(device).unsqueeze(1)


@lru_cache(maxsize=256)
def benty_symmetry_map(pos_len, board_len, symm):
    topology = benty.build_topology(board_len)
    indices = np.full(pos_len * pos_len, pos_len * pos_len, dtype=np.int64)
    for point in topology.playable:
        mapped = topology.sym_pos[point][symm]
        source = point // board_len * pos_len + point % board_len
        destination = mapped // board_len * pos_len + mapped % board_len
        indices[destination] = source
    return indices


def benty_symmetry_indices(pos_len, symm, board_lens, device):
    indices = np.stack([benty_symmetry_map(pos_len, board_len, symm) for board_len in board_lens])
    return torch.from_numpy(indices).to(device).unsqueeze(1)


def apply_symmetry_policy(tensor, symm, pos_len, board_shape="y", board_lens=None, indices=None):
    """Same as apply_symmetry but also handles the pass index"""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    tensor_without_pass = tensor[:,:,:-1].view((batch_size, channels, pos_len, pos_len))
    tensor_transformed = apply_symmetry(tensor_without_pass, symm, board_shape, board_lens, indices)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len*pos_len),
        tensor[:,:,-1:]
    ), dim=2)

def apply_symmetry(tensor, symm, board_shape="y", board_lens=None, indices=None):
    """
    Apply a shape-specific board symmetry.

    Args:
        tensor (torch.Tensor): Tensor to transform. (..., W, W)
        symm (int): one of 2 symmetries for Hex, 6 for Y/bentY, or 12 for Yavalath.
    """
    assert tensor.shape[-1] == tensor.shape[-2]
    board_shape = normalize_board_shape(board_shape)

    if symm == 0:
        return tensor
    if board_shape == "hex":
        assert symm == 1
        return tensor.flip(-1).flip(-2)

    pos_len = tensor.shape[-1]
    if board_lens is None:
        board_lens = [pos_len] * tensor.shape[0]
    if indices is None:
        make_indices = {"y": y_symmetry_indices, "yavalath": yavalath_symmetry_indices,
                        "bentY": benty_symmetry_indices}[board_shape]
        indices = make_indices(pos_len, symm, board_lens, tensor.device)
    flat = tensor.reshape(tensor.shape[0], -1, pos_len * pos_len)
    source = torch.cat((flat, flat.new_zeros((*flat.shape[:2], 1))), dim=2)
    return source.gather(2, indices.expand(-1, flat.shape[1], -1)).reshape_as(tensor)
