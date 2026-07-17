#!/usr/bin/env python3
"""Enlarge the storage canvas of a stopped run's selfplay, retaining replay age."""

import argparse
import os
from pathlib import Path

import numpy as np


def pad_arrays(arrays, pos_len):
    old_len = arrays['valueTargetsNCHW'].shape[-1]
    if old_len == pos_len:
        return arrays
    old_area, area = old_len * old_len, pos_len * pos_len
    pad = pos_len - old_len
    packed = arrays['binaryInputNCHWPacked']
    inputs = np.unpackbits(packed, axis=2)[:, :, :old_area].reshape(*packed.shape[:2], old_len, old_len)
    inputs = np.pad(inputs, ((0, 0), (0, 0), (0, pad), (0, pad)))
    policy = arrays['policyTargetsNCMove']
    moves = policy[:, :, :-1].reshape(*policy.shape[:2], old_len, old_len)
    moves = np.pad(moves, ((0, 0), (0, 0), (0, pad), (0, pad)))
    return dict(arrays,
                binaryInputNCHWPacked=np.packbits(inputs.reshape(*packed.shape[:2], area), axis=2),
                policyTargetsNCMove=np.concatenate((moves.reshape(*policy.shape[:2], area), policy[:, :, -1:]), axis=2),
                valueTargetsNCHW=np.pad(arrays['valueTargetsNCHW'], ((0, 0), (0, 0), (0, pad), (0, pad))),
                scoreDistrN=np.pad(arrays['scoreDistrN'], ((0, 0), (area - old_area, area - old_area))))


def pad_file(path, pos_len):
    stat = path.stat()
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    padded = pad_arrays(arrays, pos_len)
    if padded is arrays:
        return False
    temporary = path.with_name(path.stem + '_pad.npz')
    try:
        np.savez_compressed(temporary, **padded)
        os.chmod(temporary, stat.st_mode)
        os.utime(temporary, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Run data directory containing selfplay/')
    parser.add_argument('--pos-len', type=int, required=True, help='New storage canvas width')
    args = parser.parse_args()
    changed = 0
    for path in sorted((args.directory / 'selfplay').glob('*/tdata/*.npz')):
        if '_' not in path.name and pad_file(path, args.pos_len):
            changed += 1
    print(f'Padded {changed} selfplay files to {args.pos_len}×{args.pos_len}.')


if __name__ == '__main__':
    main()
