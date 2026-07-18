#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import Callable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = (SCRIPT_DIR / "../python").resolve()
sys.path.insert(0, str(PYTHON_DIR))
import benty


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
Y_SIDE = "#525252"
SUPPORTED_WIDTHS = (5, 9, 13)
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


def parse_token(token: str, topology: benty.BentYTopology) -> Optional[int]:
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
        SCRIPT_DIR / "../build-opencl/katago",
        SCRIPT_DIR / "../cpp/katago",
        SCRIPT_DIR / "engine/katago",
    ]
    for candidate in candidates:
        path = candidate.resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return candidates[0].resolve()


def display_positions(
    topology: benty.BentYTopology,
    bow: float = 1.0,
) -> dict[int, tuple[float, float]]:
    outer = topology.layers[0]
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


def parse_analysis_line(line: str, topology: benty.BentYTopology) -> list[AnalysisMove]:
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


class Engine:
    def __init__(
        self,
        engine: Path,
        config: Path,
        model: Path,
        width: int,
        interval_cs: int,
        analysis_wide_root_noise: float,
        on_error: Callable[[str], None],
    ):
        self.width = width
        self.topology = benty.build_topology(benty.tensor_len_for_width(width))
        self.interval_cs = interval_cs
        self.analysis_wide_root_noise = analysis_wide_root_noise
        self.analysis_wide_root_noise_sent: Optional[float] = None
        self.on_error = on_error
        self.analysis: list[AnalysisMove] = []
        self.stderr_tail: deque[str] = deque(maxlen=8)
        self.lock = threading.Lock()
        self.analysis_active = False
        self.analysis_mute_until_sync = False
        self.closed = False
        self.proc = subprocess.Popen(
            [str(engine), "gtp", "-config", str(config), "-model", str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._stdout, daemon=True).start()
        threading.Thread(target=self._stderr, daemon=True).start()
        self.send(f"boardsize {self.topology.tensor_len}")
        self.send("clear_board")
        self.set_analysis_wide_root_noise(analysis_wide_root_noise)

    def send(self, command: str) -> None:
        if self.closed or self.proc.poll() is not None:
            return
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(command + "\n")
            self.proc.stdin.flush()
        except Exception as exc:
            self.on_error(str(exc))

    def _stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            with self.lock:
                if not self.analysis_active:
                    continue
                if self.analysis_mute_until_sync:
                    if line.lstrip().startswith("="):
                        self.analysis_mute_until_sync = False
                    continue
            parsed = parse_analysis_line(line, self.topology)
            if parsed:
                with self.lock:
                    if self.analysis_active and not self.analysis_mute_until_sync:
                        self.analysis = parsed

    def _stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            line = line.strip()
            if line:
                self.stderr_tail.append(line)

    def stop_analysis(self, clear: bool = False) -> None:
        with self.lock:
            self.analysis_active = False
            self.analysis_mute_until_sync = False
            if clear:
                self.analysis = []
        self.send("stop")

    def sync_position(self, moves: list[Move]) -> None:
        self.stop_analysis(clear=True)
        self.send(f"boardsize {self.topology.tensor_len}")
        self.send("clear_board")
        for move in moves:
            token = "pass" if move.pos is None else token_for_pos(move.pos, self.topology.tensor_len)
            self.send(f"play {move.player} {token}")

    def set_width(self, width: int) -> None:
        self.stop_analysis(clear=True)
        self.width = width
        self.topology = benty.build_topology(benty.tensor_len_for_width(width))
        self.send(f"boardsize {self.topology.tensor_len}")
        self.send("clear_board")

    def set_analysis_wide_root_noise(self, value: float) -> None:
        self.analysis_wide_root_noise = value
        if self.analysis_wide_root_noise_sent == value:
            return
        self.analysis_wide_root_noise_sent = value
        self.send(f"kata-set-param analysisWideRootNoise {format_noise(value)}")

    def start_analysis(self, player: str, clear: bool = True) -> None:
        self.set_analysis_wide_root_noise(self.analysis_wide_root_noise)
        self.send(f"kata-analyze {player} {self.interval_cs}")
        with self.lock:
            if clear:
                self.analysis = []
            self.analysis_active = True
            self.analysis_mute_until_sync = True

    def get_analysis(self) -> list[AnalysisMove]:
        with self.lock:
            return list(self.analysis)

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.send("quit")
        finally:
            self.closed = True
        try:
            self.proc.terminate()
        except Exception:
            pass


class BentYGUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.width = args.width
        self.pending_width = args.width
        self.topology = benty.build_topology(benty.tensor_len_for_width(self.width))
        self.normalized_positions = display_positions(self.topology)
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

        root.title(f"Bent-Y {self.width}")
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
        root.bind("<KeyPress-equal>", lambda _event: self.change_pending_width(1))
        root.bind("<KeyPress-plus>", lambda _event: self.change_pending_width(1))
        root.bind("<KeyPress-KP_Add>", lambda _event: self.change_pending_width(1))
        root.bind("<KeyPress-minus>", lambda _event: self.change_pending_width(-1))
        root.bind("<KeyPress-underscore>", lambda _event: self.change_pending_width(-1))
        root.bind("<KeyPress-KP_Subtract>", lambda _event: self.change_pending_width(-1))
        root.bind("<KeyPress-bracketleft>", lambda _event: self.step_analysis_wide_root_noise(-1))
        root.bind("<KeyPress-bracketright>", lambda _event: self.step_analysis_wide_root_noise(1))
        root.bind("<Return>", lambda _event: self.apply_pending_width())
        root.bind("<KP_Enter>", lambda _event: self.apply_pending_width())
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
                self.width,
                self.args.interval_cs,
                self.analysis_wide_root_noise,
                lambda _message: self.engine_var.set("engine: error"),
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

    def play(self, pos: int) -> None:
        if pos in self.occupied():
            return
        self.moves.append(Move(self.current_player(), pos))
        self.analysis_by_pos.clear()
        if self.engine is not None:
            self._sync_engine_position()
            if self.analysis_enabled:
                self.engine.start_analysis(self.current_player())
        self.redraw()

    def play_pass(self) -> None:
        self.moves.append(Move(self.current_player(), None))
        self.analysis_by_pos.clear()
        if self.engine is not None:
            self._sync_engine_position()
            if self.analysis_enabled:
                self.engine.start_analysis(self.current_player())
        self.redraw()

    def undo(self) -> None:
        if not self.moves:
            return
        self.moves.pop()
        self.analysis_by_pos.clear()
        if self.engine is not None:
            self._sync_engine_position()
            if self.analysis_enabled:
                self.engine.start_analysis(self.current_player())
        self.redraw()

    def clear(self) -> None:
        self.moves.clear()
        self.analysis_by_pos.clear()
        if self.engine is not None:
            self._sync_engine_position()
            if self.analysis_enabled:
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
        if self.analysis_enabled:
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

    def change_pending_width(self, direction: int) -> None:
        index = SUPPORTED_WIDTHS.index(self.pending_width)
        index = max(0, min(len(SUPPORTED_WIDTHS) - 1, index + direction))
        pending_width = SUPPORTED_WIDTHS[index]
        if pending_width != self.pending_width:
            self.pending_width = pending_width
            self.redraw()

    def apply_pending_width(self) -> None:
        if self.pending_width == self.width:
            return
        self.width = self.pending_width
        self.topology = benty.build_topology(benty.tensor_len_for_width(self.width))
        self.normalized_positions = display_positions(self.topology)
        self.moves.clear()
        self.analysis_by_pos.clear()
        self.root.title(f"Bent-Y {self.width}")
        if self.engine is not None:
            self.engine.set_width(self.width)
            self.engine_synced_moves = ()
            if self.analysis_enabled:
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
            if self.analysis_enabled:
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
            if self.analysis_enabled:
                self.analysis_by_pos = {
                    item.pos: item for item in self.engine.get_analysis()
                    if item.pos is not None
                }
                self.redraw()
        self.root.after(250, self._poll_analysis)

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
        radius = max(7.0, min(24.0, scale * 0.35 / self.width))
        return positions, radius

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
        pending = f" (pending {self.pending_width})" if self.pending_width != self.width else ""
        self.status_var.set(
            f"width: {self.width}{pending}    to move: {self.current_player()}    moves: {len(self.moves)}    "
            f"awrn: {format_noise(self.analysis_wide_root_noise)}"
        )
        self.analysis_button_text.set("Stop" if self.analysis_enabled else "Analyze")
        top = self.top_ranked_move()
        top_analysis = self.analysis_by_pos.get(top) if top is not None else None
        visit_denom = math.log(max(2, top_analysis.visits if top_analysis and top_analysis.visits else 0))

        for a, b in self.topology.edges:
            common_sides = self.topology.side_masks[a] & self.topology.side_masks[b]
            color = GRID
            line_width = max(1.0, radius * 0.12)
            if common_sides:
                color = Y_SIDE
                line_width = max(2.0, radius * 0.19)
            self.canvas.create_line(*positions[a], *positions[b], fill=color, width=line_width)

        stone_font = tkfont.Font(family="Helvetica", size=max(7, int(radius * 0.70)), weight="bold")
        analysis_font = tkfont.Font(family="Helvetica", size=max(6, int(radius * 0.55)), weight="bold")
        for pos in self.topology.playable:
            x, y = positions[pos]
            player = occupied.get(pos)
            fill = BLACK if player == "B" else WHITE if player == "W" else self.analysis_fill(pos, visit_denom)
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline=GRID, width=1)

        for number, move in enumerate(self.moves, start=1):
            if move.pos is None:
                continue
            x, y = positions[move.pos]
            self.canvas.create_text(x, y, text=str(number), fill=TEXT_ON_STONE, font=stone_font)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bent-Y board and KataGomo analysis GUI")
    parser.add_argument("--width", type=int, choices=SUPPORTED_WIDTHS, default=9)
    parser.add_argument("--engine", type=Path, default=default_engine_path())
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "by9/gtp.local.cfg")
    parser.add_argument("--model", type=Path, default=SCRIPT_DIR / "by9/data/latest.bin.gz")
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


def main() -> int:
    args = parse_args()
    args.engine = args.engine.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    root = tk.Tk()
    BentYGUI(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
