import logging
import os

import numpy as np

import torch
import torch.nn.functional

import modelconfigs

def read_npz_training_data(
    npz_files,
    batch_size: int,
    world_size: int,
    rank: int,
    pos_len: int,
    device,
    randomize_symmetries: bool,
    model_config: modelconfigs.ModelConfig,
):
    rand = np.random.default_rng(seed=list(os.urandom(12)))
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)

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


            if randomize_symmetries:
                symm = int(rand.integers(0, 4))
                directional_crosscuts = batch_globalInputNC[:,18] > 0.5
                tensor_board_lens = torch.sqrt(batch_binaryInputNCHW[:,0].sum(dim=(1,2))).round().to(torch.int64)
                batch_binaryInputNCHW = apply_symmetry_binary(batch_binaryInputNCHW, symm)
                batch_policyTargetsNCMove = apply_symmetry_policy(batch_policyTargetsNCMove, symm, pos_len, directional_crosscuts, tensor_board_lens)
                batch_valueTargetsNCHW = apply_symmetry(batch_valueTargetsNCHW, symm)
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


def apply_symmetry_policy(tensor, symm, pos_len, directional_crosscuts, tensor_board_lens):
    """Same as apply_symmetry but also handles the pass index"""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    tensor_without_pass = tensor[:,:,:-1].view((batch_size, channels, pos_len, pos_len))
    tensor_transformed = apply_symmetry(tensor_without_pass, symm)
    if torch.any(directional_crosscuts):
        swap_directions = bool(symm & 1) != bool(symm & 2)
        for tensor_board_len_tensor in torch.unique(tensor_board_lens[directional_crosscuts]):
            tensor_board_len = int(tensor_board_len_tensor.item())
            board_size = (tensor_board_len + 1) // 2
            rows = directional_crosscuts & (tensor_board_lens == tensor_board_len)
            y_offset = pos_len-tensor_board_len if symm & 1 else 0
            x_offset = pos_len-tensor_board_len if symm & 2 else 0
            raw_indices = []
            geometric_indices = []
            transformed_indices = []
            for row in range(board_size-1):
                for x in range(board_size-1):
                    raw_backslash = (2*row,2*x+1)
                    raw_slash = (2*row+1,2*x)
                    geom_backslash = (pos_len-1-raw_backslash[0] if symm & 1 else raw_backslash[0],pos_len-1-raw_backslash[1] if symm & 2 else raw_backslash[1])
                    geom_slash = (pos_len-1-raw_slash[0] if symm & 1 else raw_slash[0],pos_len-1-raw_slash[1] if symm & 2 else raw_slash[1])
                    sym_row = board_size-2-row if symm & 1 else row
                    sym_x = board_size-2-x if symm & 2 else x
                    sym_backslash = (y_offset+2*sym_row,x_offset+2*sym_x+1)
                    sym_slash = (y_offset+2*sym_row+1,x_offset+2*sym_x)
                    if swap_directions:
                        sym_backslash,sym_slash = sym_slash,sym_backslash
                    raw_indices.extend((raw_backslash[0]*pos_len+raw_backslash[1],raw_slash[0]*pos_len+raw_slash[1]))
                    geometric_indices.extend((geom_backslash[0]*pos_len+geom_backslash[1],geom_slash[0]*pos_len+geom_slash[1]))
                    transformed_indices.extend((sym_backslash[0]*pos_len+sym_backslash[1],sym_slash[0]*pos_len+sym_slash[1]))

            transformed_rows = tensor_transformed[rows].clone().reshape(-1,channels,pos_len*pos_len)
            raw_rows = tensor_without_pass[rows].reshape(-1,channels,pos_len*pos_len)
            transformed_rows[:,:,geometric_indices] = 0
            transformed_rows[:,:,transformed_indices] = raw_rows[:,:,raw_indices]
            tensor_transformed[rows] = transformed_rows.reshape(-1,channels,pos_len,pos_len)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len*pos_len),
        tensor[:,:,-1:]
    ), dim=2)

def apply_symmetry_binary(tensor, symm):
    tensor = apply_symmetry(tensor,symm)
    if bool(symm & 1) != bool(symm & 2):
        tensor = tensor.clone()
        backslash = tensor[:,6].clone()
        tensor[:,6] = tensor[:,7]
        tensor[:,7] = backslash
    return tensor

def apply_symmetry(tensor, symm):
    """Apply a player-preserving Quax symmetry to a square expanded lattice."""
    assert tensor.shape[-1] == tensor.shape[-2]
    if symm < 0 or symm >= 4:
        raise ValueError(f"Unknown Quax symmetry: {symm}")
    if symm & 1:
        tensor = tensor.flip(-2)
    if symm & 2:
        tensor = tensor.flip(-1)
    return tensor
