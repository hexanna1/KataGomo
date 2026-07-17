#!/usr/bin/env python3
from __future__ import annotations

import tkinter as tk

from play_common import SCRIPT_DIR, HexBoardGUI, parse_args


class YavalathGUI(HexBoardGUI):
    name = "Yavalath"

    def is_game_finished(self) -> bool:
        if not self.moves:
            return False
        move = self.moves[-1]
        if move.pos is None:
            return False
        occupied = self.occupied()
        if len(occupied) == len(self.topology.playable):
            return True
        n = self.topology.tensor_len
        x, y = move.pos % n, move.pos // n
        for dx, dy in ((1, 0), (0, 1), (1, -1)):
            length = 1
            for sign in (-1, 1):
                xx, yy = x + sign * dx, y + sign * dy
                while 0 <= xx < n and 0 <= yy < n and occupied.get(yy * n + xx) == move.player:
                    length += 1
                    xx, yy = xx + sign * dx, yy + sign * dy
            if length >= 3:
                return True
        return False


def main() -> int:
    args = parse_args(
        "Yavalath", YavalathGUI.supported_sizes, 5, "yavalath-t4",
        SCRIPT_DIR / "../build-engine-y/katago", size_label="size",
    )
    args.engine = args.engine.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    root = tk.Tk()
    YavalathGUI(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
