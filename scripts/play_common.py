#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import Optional

from play_gtp import GTPClient


SCRIPT_DIR = Path(__file__).resolve().parent


GTP_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MOVE_RE = re.compile(r"^([A-Za-z]+)(\d+)$")
INFO_RE = re.compile(r"\binfo\s+move\s+(\S+)\s+(.*?)(?=\binfo\s+move\s+\S+\s+|$)", re.DOTALL)
BLACK = "#dc3c3c"
WHITE = "#2864dc"
TEXT_ON_STONE = "#fafafa"
EMPTY = (231, 221, 202)
ANALYSIS_LOW = (217, 212, 205)
ANALYSIS_HIGH = (170, 125, 210)
CANVAS_BG = "#f2efe8"
GRID = "#75664f"
ANALYSIS_WIDE_ROOT_NOISE_STEPS = (0.00, 0.01, 0.02, 0.04, 0.10, 0.20, 0.50, 1.00, 2.00)
DEFAULT_ANALYSIS_WIDE_ROOT_NOISE = 0.20


def column_letters(x: int) -> str:
    value = x + 1
    result = ""
    while value:
        value, digit = divmod(value - 1, len(GTP_ALPH))
        result = GTP_ALPH[digit] + result
    return result


def token_for_pos(pos: int, tensor_len: int) -> str:
    return f"{column_letters(pos % tensor_len)}{pos // tensor_len + 1}"


def parse_token(token: str, topology) -> Optional[int]:
    stripped = token.strip()
    if stripped.lower() in ("pass", "resign"):
        return None
    match = MOVE_RE.match(stripped)
    if match is None:
        return None
    x = 0
    for char in match.group(1).upper():
        digit = GTP_ALPH.find(char) + 1
        if digit <= 0:
            return None
        x = x * len(GTP_ALPH) + digit
    x -= 1
    y = int(match.group(2)) - 1
    pos = y * topology.tensor_len + x
    if x < 0 or x >= topology.tensor_len or y < 0 or y >= topology.tensor_len:
        return None
    return pos if pos in topology.playable else None


def field(text: str, name: str, cast):
    match = re.search(rf"\b{name}\s+([^\s]+)", text)
    if match is None:
        return None
    try:
        return cast(match.group(1))
    except Exception:
        return None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    t = clamp01(t)
    values = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def format_visits(visits: Optional[int]) -> str:
    if visits is None:
        return ""
    if visits < 1000:
        return str(visits)
    if visits < 100_000:
        return f"{visits / 1000:.1f}k"
    if visits < 1_000_000:
        return f"{visits // 1000}k"
    if visits < 10_000_000:
        return f"{visits / 1_000_000:.2f}m"
    if visits < 1_000_000_000:
        return f"{visits / 1_000_000:.1f}m"
    return f"{visits / 1_000_000:.0f}m"


def format_percent(value: Optional[float]) -> str:
    return "" if value is None else f"{value * 100:.1f}"


def format_noise(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def default_engine_path() -> Path:
    candidates = [
        SCRIPT_DIR / "../build-engine-y/katago",
        SCRIPT_DIR / "../cpp/katago",
        SCRIPT_DIR / "engine/katago",
    ]
    for candidate in candidates:
        path = candidate.resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return candidates[0].resolve()


@dataclass(frozen=True)
class Move:
    player: str
    pos: Optional[int]


@dataclass(frozen=True)
class AnalysisMove:
    token: str
    pos: Optional[int]
    order: Optional[int]
    visits: Optional[int]
    winrate: Optional[float]
    prior: Optional[float]
    pv: tuple[int, ...]


def parse_analysis_line(line: str, topology) -> list[AnalysisMove]:
    if "info move " not in line:
        return []
    result: list[AnalysisMove] = []
    for match in INFO_RE.finditer(line):
        token, rest = match.group(1), match.group(2)
        pv: tuple[int, ...] = ()
        pv_pos = rest.find(" pv ")
        if pv_pos >= 0:
            parsed_pv: list[int] = []
            for item in rest[pv_pos + 4:].split():
                parsed = parse_token(item, topology)
                if parsed is None:
                    break
                parsed_pv.append(parsed)
            pv = tuple(parsed_pv)
            rest = rest[:pv_pos]
        visits = field(rest, "visits", int)
        if visits == 0:
            continue
        result.append(AnalysisMove(
            token=token,
            pos=parse_token(token, topology),
            order=field(rest, "order", int),
            visits=visits,
            winrate=field(rest, "winrate", float),
            prior=field(rest, "prior", float),
            pv=pv,
        ))
    result.sort(key=lambda item: item.order if item.order is not None else 10**9)
    return result


class Engine(GTPClient):
    def __init__(self, engine: Path, config: Path, model: Path, size: int, interval_cs: int,
                 analysis_wide_root_noise: float, topology_factory):
        self.size = size
        self.topology_factory = topology_factory
        self.topology = topology_factory(size)
        super().__init__(
            [str(engine), "gtp", "-config", str(config), "-model", str(model),
             "-override-config", f"defaultBoardSize={self.topology.tensor_len}"],
            interval_cs, analysis_wide_root_noise,
            lambda line: parse_analysis_line(line, self.topology),
        )
        self.send(f"boardsize {self.topology.tensor_len}")
        self.send("clear_board")
        self.set_analysis_wide_root_noise(analysis_wide_root_noise)

    def sync_position(self, moves: list[Move]) -> None:
        self.stop_analysis(clear=True)
        self.send(f"boardsize {self.topology.tensor_len}")
        self.send("clear_board")
        for move in moves:
            token = "pass" if move.pos is None else token_for_pos(move.pos, self.topology.tensor_len)
            self.send(f"play {move.player} {token}")

    def set_size(self, size: int) -> None:
        self.stop_analysis(clear=True)
        self.size = size
        self.topology = self.topology_factory(size)
        self.send(f"boardsize {self.topology.tensor_len}")
        self.send("clear_board")

class BoardGUI:
    size_label = "size"
    player_colors = {"B": BLACK, "W": WHITE}
    player_text_colors = {"B": TEXT_ON_STONE, "W": TEXT_ON_STONE}
    player_names = {"B": "B", "W": "W"}

    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.size = args.size
        self.pending_size = args.size
        self.topology = self.build_topology(self.size)
        self.normalized_positions = self.display_positions(self.topology)
        self.analysis_wide_root_noise = args.analysis_wide_root_noise
        self.moves: list[Move] = []
        self.engine: Optional[Engine] = None
        self.engine_synced_moves: tuple[Move, ...] = ()
        self.analysis_enabled = False
        self.show_prior = False
        self.analysis_by_pos: dict[int, AnalysisMove] = {}
        self.status_var = tk.StringVar()
        self.engine_var = tk.StringVar()
        self.analysis_button_text = tk.StringVar(value="Analyze")

        root.title(f"{self.name} {self.size}")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<space>", self._toggle_analysis_from_space)
        root.bind("<BackSpace>", lambda _event: self.undo())
        root.bind("z", lambda _event: self.undo())
        root.bind("<KeyPress-C>", lambda _event: self.clear())
        root.bind("<KeyPress-P>", lambda _event: self.play_pass())
        root.bind("<KeyPress-comma>", lambda _event: self.play_top_ranked_move())
        root.bind("q", lambda _event: self.close())
        root.bind("<Escape>", lambda _event: self.close())
        root.bind("<Control-d>", lambda _event: self.close())
        root.bind("<KeyPress-t>", self._show_prior)
        root.bind("<KeyRelease-t>", self._hide_prior)
        root.bind("<KeyPress-equal>", lambda _event: self.change_pending_size(1))
        root.bind("<KeyPress-plus>", lambda _event: self.change_pending_size(1))
        root.bind("<KeyPress-KP_Add>", lambda _event: self.change_pending_size(1))
        root.bind("<KeyPress-minus>", lambda _event: self.change_pending_size(-1))
        root.bind("<KeyPress-underscore>", lambda _event: self.change_pending_size(-1))
        root.bind("<KeyPress-KP_Subtract>", lambda _event: self.change_pending_size(-1))
        root.bind("<KeyPress-bracketleft>", lambda _event: self.step_analysis_wide_root_noise(-1))
        root.bind("<KeyPress-bracketright>", lambda _event: self.step_analysis_wide_root_noise(1))
        root.bind("<Return>", lambda _event: self.apply_pending_size())
        root.bind("<KP_Enter>", lambda _event: self.apply_pending_size())
        self._build_widgets()
        self._try_start_engine()
        self.redraw()
        root.after(250, self._poll_analysis)

    def _build_widgets(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, textvariable=self.analysis_button_text, command=self.toggle_analysis).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=(6, 0))
        for button in toolbar.winfo_children():
            button.bind("<space>", self._toggle_analysis_from_space)
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Label(toolbar, textvariable=self.engine_var).pack(side=tk.RIGHT)
        self.canvas = tk.Canvas(self.root, width=760, height=680, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        self.canvas.bind("<Button-1>", self.click)

    def _try_start_engine(self) -> None:
        if self.args.no_engine:
            self.engine_var.set("engine: disabled")
            return
        if not self.args.engine.is_file() or not os.access(self.args.engine, os.X_OK):
            self.engine_var.set("engine: missing")
            return
        if not self.args.config.exists():
            self.engine_var.set("config: missing")
            return
        if not self.args.model.exists():
            self.engine_var.set("model: missing")
            return
        try:
            self.engine = Engine(
                self.args.engine,
                self.args.config,
                self.args.model,
                self.size,
                self.args.interval_cs,
                self.analysis_wide_root_noise,
                self.build_topology,
            )
        except Exception:
            self.engine_var.set("engine: failed")
            return
        self.engine_synced_moves = ()
        self.engine_var.set("engine: ready")

    def current_player(self) -> str:
        return "B" if len(self.moves) % 2 == 0 else "W"

    def occupied(self) -> dict[int, str]:
        return {move.pos: move.player for move in self.moves if move.pos is not None}

    def _sync_engine_position(self) -> None:
        if self.engine is None:
            return
        moves = tuple(self.moves)
        if moves == self.engine_synced_moves:
            return
        self.engine.sync_position(self.moves)
        self.engine_synced_moves = moves

    def is_game_finished(self) -> bool:
        return False

    def play(self, pos: Optional[int]) -> None:
        if self.is_game_finished() or pos in self.occupied():
            return
        self.moves.append(Move(self.current_player(), pos))
        self.analysis_by_pos.clear()
        if self.engine is not None:
            self._sync_engine_position()
            if self.analysis_enabled and not self.is_game_finished():
                self.engine.start_analysis(self.current_player())
        self.redraw()

    def play_pass(self) -> None:
        self.play(None)

    def undo(self) -> None:
        if not self.moves:
            return
        self.moves.pop()
        self.analysis_by_pos.clear()
        if self.engine is not None:
            self._sync_engine_position()
            if self.analysis_enabled and not self.is_game_finished():
                self.engine.start_analysis(self.current_player())
        self.redraw()

    def clear(self) -> None:
        self.moves.clear()
        self.analysis_by_pos.clear()
        if self.engine is not None:
            self._sync_engine_position()
            if self.analysis_enabled and not self.is_game_finished():
                self.engine.start_analysis(self.current_player())
        self.redraw()

    def _toggle_analysis_from_space(self, _event: tk.Event) -> str:
        self.toggle_analysis()
        return "break"

    def toggle_analysis(self) -> None:
        if self.engine is None:
            self._try_start_engine()
            if self.engine is None:
                return
        self.analysis_enabled = not self.analysis_enabled
        if self.analysis_enabled and not self.is_game_finished():
            self._sync_engine_position()
            self.engine.start_analysis(self.current_player(), clear=False)
        else:
            self.engine.stop_analysis()
        self.redraw()

    def top_ranked_move(self) -> Optional[int]:
        occupied = self.occupied()
        candidates = [
            item for pos, item in self.analysis_by_pos.items()
            if pos not in occupied and item.pos is not None
        ]
        if not candidates:
            return None
        best = min(candidates, key=lambda item: (
            item.order if item.order is not None else 10**9,
            -(item.visits or 0),
            item.pos or 0,
        ))
        return best.pos

    def play_top_ranked_move(self) -> None:
        pos = self.top_ranked_move()
        if pos is not None:
            self.play(pos)

    def change_pending_size(self, direction: int) -> None:
        index = self.supported_sizes.index(self.pending_size)
        index = max(0, min(len(self.supported_sizes) - 1, index + direction))
        pending_size = self.supported_sizes[index]
        if pending_size != self.pending_size:
            self.pending_size = pending_size
            self.redraw()

    def apply_pending_size(self) -> None:
        if self.pending_size == self.size:
            return
        self.size = self.pending_size
        self.topology = self.build_topology(self.size)
        self.normalized_positions = self.display_positions(self.topology)
        self.moves.clear()
        self.analysis_by_pos.clear()
        self.root.title(f"{self.name} {self.size}")
        if self.engine is not None:
            self.engine.set_size(self.size)
            self.engine_synced_moves = ()
            if self.analysis_enabled and not self.is_game_finished():
                self.engine.start_analysis(self.current_player())
        self.redraw()

    def step_analysis_wide_root_noise(self, direction: int) -> None:
        index = min(
            range(len(ANALYSIS_WIDE_ROOT_NOISE_STEPS)),
            key=lambda i: abs(ANALYSIS_WIDE_ROOT_NOISE_STEPS[i] - self.analysis_wide_root_noise),
        )
        index = max(0, min(len(ANALYSIS_WIDE_ROOT_NOISE_STEPS) - 1, index + direction))
        self.analysis_wide_root_noise = ANALYSIS_WIDE_ROOT_NOISE_STEPS[index]
        if self.engine is not None:
            self.engine.set_analysis_wide_root_noise(self.analysis_wide_root_noise)
            if self.analysis_enabled and not self.is_game_finished():
                self.engine.start_analysis(self.current_player(), clear=False)
        self.redraw()

    def _show_prior(self, _event) -> None:
        if not self.show_prior:
            self.show_prior = True
            self.redraw()

    def _hide_prior(self, _event) -> None:
        if self.show_prior:
            self.show_prior = False
            self.redraw()

    def _poll_analysis(self) -> None:
        if self.engine is not None:
            if self.engine.proc.poll() is not None and not self.engine.closed:
                self.engine_var.set("engine: stopped")
            if self.analysis_enabled and not self.is_game_finished():
                self.analysis_by_pos = {
                    item.pos: item for item in self.engine.get_analysis()
                    if item.pos is not None
                }
                self.redraw()
        self.root.after(250, self._poll_analysis)

    def analysis_fill(self, pos: int, visit_denom: float) -> str:
        analysis = self.analysis_by_pos.get(pos)
        if analysis is None:
            return lerp_rgb(EMPTY, EMPTY, 0.0)
        if self.show_prior:
            if analysis.prior is None:
                return lerp_rgb(EMPTY, EMPTY, 0.0)
            return lerp_rgb(ANALYSIS_LOW, ANALYSIS_HIGH, clamp01(analysis.prior) ** 0.9)
        if analysis.visits is not None and analysis.winrate is not None:
            value = 0.0 if visit_denom <= 0 else math.log(max(1, analysis.visits)) / visit_denom
            value = clamp01(value) ** 1.1
            value = clamp01(value + 0.35 * (analysis.winrate - 0.5) * 2.0)
            return lerp_rgb(ANALYSIS_LOW, ANALYSIS_HIGH, value)
        return lerp_rgb(EMPTY, EMPTY, 0.0)

    def redraw(self) -> None:
        self.canvas.delete("all")
        positions, radius = self.layout()
        occupied = self.occupied()
        pending = f" (pending {self.pending_size})" if self.pending_size != self.size else ""
        self.status_var.set(
            f"{self.size_label}: {self.size}{pending}    to move: {self.player_names[self.current_player()]}    moves: {len(self.moves)}    "
            f"awrn: {format_noise(self.analysis_wide_root_noise)}"
        )
        self.analysis_button_text.set("Stop" if self.analysis_enabled else "Analyze")
        top = self.top_ranked_move()
        top_analysis = self.analysis_by_pos.get(top) if top is not None else None
        visit_denom = math.log(max(2, top_analysis.visits if top_analysis and top_analysis.visits else 0))

        self.draw_board(positions, radius, occupied, visit_denom)

        stone_font = tkfont.Font(family="Helvetica", size=max(7, int(radius * 0.70)), weight="bold")
        analysis_font = tkfont.Font(family="Helvetica", size=max(6, int(radius * 0.55)), weight="bold")

        for number, move in enumerate(self.moves, start=1):
            if move.pos is None:
                continue
            x, y = positions[move.pos]
            self.canvas.create_text(x, y, text=str(number), fill=self.player_text_colors[move.player], font=stone_font)

        for pos, analysis in self.analysis_by_pos.items():
            if pos in occupied or pos not in positions:
                continue
            x, y = positions[pos]
            if self.show_prior:
                text = format_percent(analysis.prior)
                if text:
                    self.canvas.create_text(x, y, text=text, fill="#111111", font=analysis_font)
            else:
                winrate = format_percent(analysis.winrate)
                visits = format_visits(analysis.visits)
                gap = max(7, radius * 0.65)
                if winrate:
                    self.canvas.create_text(x, y - gap / 2, text=winrate, fill="#111111", font=analysis_font)
                if visits:
                    self.canvas.create_text(x, y + gap / 2, text=visits, fill="#111111", font=analysis_font)

    def click(self, event: tk.Event) -> None:
        positions, radius = self.layout()
        candidates = [
            ((event.x - x) ** 2 + (event.y - y) ** 2, pos)
            for pos, (x, y) in positions.items()
            if (event.x - x) ** 2 + (event.y - y) ** 2 <= (radius * 1.35) ** 2
        ]
        if candidates:
            self.play(min(candidates)[1])

    def close(self) -> None:
        if self.engine is not None:
            self.engine.close()
        self.root.destroy()


@dataclass(frozen=True)
class HexTopology:
    tensor_len: int
    playable: tuple[int, ...]
    side_masks: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]


def build_hex_topology(size: int) -> HexTopology:
    n = 2 * size - 1
    radius = size - 1
    playable = tuple(
        y * n + x
        for y in range(n) for x in range(n)
        if radius <= x + y <= 3 * radius
    )
    cells = set(playable)
    sides = [0] * (n * n)
    edges = []
    for pos in playable:
        x, y = pos % n, pos // n
        if y == 0 or x == n - 1:
            sides[pos] |= 1
        if x + y == radius or x == 0:
            sides[pos] |= 2
        if y == n - 1 or x + y == 3 * radius:
            sides[pos] |= 4
        for dx, dy in ((1, 0), (0, 1), (-1, 1)):
            xx, yy = x + dx, y + dy
            other = yy * n + xx
            if 0 <= xx < n and 0 <= yy < n and other in cells:
                edges.append((pos, other))
    return HexTopology(n, playable, tuple(sides), tuple(edges))


def hex_display_positions(topology: HexTopology) -> dict[int, tuple[float, float]]:
    n = topology.tensor_len
    radius = (n - 1) / 2
    return {
        pos: (
            (pos % n + 0.5 * (pos // n) - 1.5 * radius) / radius,
            math.sqrt(3) / 2 * (pos // n - radius) / radius,
        )
        for pos in topology.playable
    }


class HexBoardGUI(BoardGUI):
    supported_sizes = tuple(range(2, 12))
    build_topology = staticmethod(build_hex_topology)
    display_positions = staticmethod(hex_display_positions)

    def layout(self) -> tuple[dict[int, tuple[float, float]], float]:
        canvas_width = max(300, self.canvas.winfo_width())
        canvas_height = max(300, self.canvas.winfo_height())
        sq3 = math.sqrt(3.0)
        board_w_units = sq3 * (2 * self.size - 1)
        board_h_units = 3 * (self.size - 1) + 2
        radius = min((canvas_width - 80) / board_w_units, (canvas_height - 80) / board_h_units)
        radius = max(8.0, min(radius, 42.0))
        scale = sq3 * (self.size - 1) * radius
        positions = {
            pos: (canvas_width / 2 + x * scale, canvas_height / 2 + y * scale)
            for pos, (x, y) in self.normalized_positions.items()
        }
        return positions, radius

    def _poly(self, cx: float, cy: float, radius: float) -> list[float]:
        pts: list[float] = []
        for deg in (90, 30, -30, -90, -150, 150):
            angle = math.radians(deg)
            pts.extend([cx + radius * math.cos(angle), cy + radius * math.sin(angle)])
        return pts

    def draw_board(self, positions, radius, occupied, visit_denom) -> None:
        for pos in self.topology.playable:
            x, y = positions[pos]
            player = occupied.get(pos)
            fill = self.player_colors[player] if player else self.analysis_fill(pos, visit_denom)
            self.canvas.create_polygon(self._poly(x, y, radius), fill=fill, outline=GRID, width=1)

        self._draw_borders(positions, radius)

    def _draw_borders(self, positions, radius) -> None:
        pass


def parse_args(name, sizes, size, run, engine=None, size_label="size") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{name} board and KataGomo analysis GUI")
    parser.add_argument(f"--{size_label}", dest="size", type=int, choices=sizes, default=size)
    parser.add_argument("--engine", type=Path, default=engine or default_engine_path())
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / run / "gtp.local.cfg")
    parser.add_argument("--model", type=Path, default=SCRIPT_DIR / run / "data/latest.bin.gz")
    parser.add_argument("--interval-cs", type=int, default=20)
    parser.add_argument(
        "--analysis-wide-root-noise", "--awrn",
        dest="analysis_wide_root_noise",
        type=float,
        default=DEFAULT_ANALYSIS_WIDE_ROOT_NOISE,
    )
    parser.add_argument("--no-engine", action="store_true")
    args = parser.parse_args()
    if not ANALYSIS_WIDE_ROOT_NOISE_STEPS[0] <= args.analysis_wide_root_noise <= ANALYSIS_WIDE_ROOT_NOISE_STEPS[-1]:
        parser.error(
            f"--analysis-wide-root-noise must be between {ANALYSIS_WIDE_ROOT_NOISE_STEPS[0]} "
            f"and {ANALYSIS_WIDE_ROOT_NOISE_STEPS[-1]}"
        )
    return args
