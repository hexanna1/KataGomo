#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import tkinter as tk

from play_common import GRID, SCRIPT_DIR, BoardGUI, parse_args


sys.path.insert(0, str((SCRIPT_DIR / "../python").resolve()))
import benty


Y_SIDE = "#525252"


def display_positions(
    topology: benty.BentYTopology,
    bow: float = 1.0,
) -> dict[int, tuple[float, float]]:
    outer = topology.boundary
    third = len(outer) // 3
    corners = (
        (math.cos(-math.pi / 2), math.sin(-math.pi / 2)),
        (math.cos(math.pi / 6), math.sin(math.pi / 6)),
        (math.cos(5 * math.pi / 6), math.sin(5 * math.pi / 6)),
    )
    positions: dict[int, tuple[float, float]] = {}
    for index, pos in enumerate(outer):
        side = min(index // third, 2)
        t = (index - side * third) / third
        ax, ay = corners[side]
        bx, by = corners[(side + 1) % 3]
        cx = (ax + bx) * (1.0 + bow) / 2
        cy = (ay + by) * (1.0 + bow) / 2
        one_minus = 1.0 - t
        positions[pos] = (
            one_minus * one_minus * ax + 2 * one_minus * t * cx + t * t * bx,
            one_minus * one_minus * ay + 2 * one_minus * t * cy + t * t * by,
        )

    fixed = set(outer)
    for pos in topology.playable:
        positions.setdefault(pos, (0.0, 0.0))
    for _ in range(2000):
        largest_move = 0.0
        for pos in topology.playable:
            if pos in fixed:
                continue
            neighbors = topology.neighbors[pos]
            next_position = (
                sum(positions[neighbor][0] for neighbor in neighbors) / len(neighbors),
                sum(positions[neighbor][1] for neighbor in neighbors) / len(neighbors),
            )
            largest_move = max(largest_move, math.dist(positions[pos], next_position))
            positions[pos] = next_position
        if largest_move < 1e-11:
            break
    return positions


class BentYGUI(BoardGUI):
    name = "Bent-Y"
    size_label = "width"
    supported_sizes = (5, 9, 13)
    display_positions = staticmethod(display_positions)

    @staticmethod
    def build_topology(size: int) -> benty.BentYTopology:
        return benty.build_topology(benty.tensor_len_for_width(size))

    def layout(self) -> tuple[dict[int, tuple[float, float]], float]:
        canvas_width = max(320, self.canvas.winfo_width())
        canvas_height = max(320, self.canvas.winfo_height())
        margin = 40
        xs = [point[0] for point in self.normalized_positions.values()]
        ys = [point[1] for point in self.normalized_positions.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        scale = min(
            (canvas_width - 2 * margin) / (max_x - min_x),
            (canvas_height - 2 * margin) / (max_y - min_y),
        )
        center_x = canvas_width / 2
        center_y = canvas_height / 2
        board_center_x = (min_x + max_x) / 2
        board_center_y = (min_y + max_y) / 2
        positions = {
            pos: (
                center_x + (x - board_center_x) * scale,
                center_y + (y - board_center_y) * scale,
            )
            for pos, (x, y) in self.normalized_positions.items()
        }
        radius = max(7.0, min(24.0, scale * 0.35 / self.size))
        return positions, radius

    def draw_board(self, positions, radius, occupied, visit_denom) -> None:
        for a, b in self.topology.edges:
            common_sides = self.topology.side_masks[a] & self.topology.side_masks[b]
            color = GRID
            line_width = max(1.0, radius * 0.12)
            if common_sides:
                color = Y_SIDE
                line_width = max(2.0, radius * 0.19)
            self.canvas.create_line(*positions[a], *positions[b], fill=color, width=line_width)

        for pos in self.topology.playable:
            x, y = positions[pos]
            player = occupied.get(pos)
            fill = self.player_colors[player] if player else self.analysis_fill(pos, visit_denom)
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline=GRID, width=1)


def main() -> int:
    args = parse_args("Bent-Y", BentYGUI.supported_sizes, 9, "by-t4", size_label="width")
    args.engine = args.engine.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    root = tk.Tk()
    BentYGUI(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
