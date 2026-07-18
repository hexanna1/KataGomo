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
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import Callable, Optional


GTP_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TOKEN_RE = re.compile(r"^([A-Za-z]+)(\d+)(\*)?([\\/])?$")
INFO_REC_RE = re.compile(r"\binfo\s+move\s+(\S+)\s+(.*?)(?=\binfo\s+move\s+\S+\s+|$)", re.DOTALL)
BLACK = "#dc3c3c"
WHITE = "#2864dc"
TEXT_ON_STONE = "#fafafa"
EMPTY_OCTAGON = (231, 221, 202)
EMPTY_DIAMOND = (221, 208, 184)
ANALYSIS_LOW = (217, 212, 205)
ANALYSIS_HIGH = (170, 125, 210)
CANVAS_BG = "#f2efe8"
OUTLINE = "#8a7b63"
MIN_BOARD_SIZE = 2
MAX_BOARD_SIZE = 11
ANALYSIS_WIDE_ROOT_NOISE_STEPS = (0.00, 0.01, 0.02, 0.04, 0.10, 0.20, 0.50, 1.00, 2.00)
DEFAULT_ANALYSIS_WIDE_ROOT_NOISE = 0.20


def column_letters(x: int) -> str:
    n = x + 1
    letters = ""
    while n:
        n, digit = divmod(n - 1, len(GTP_ALPH))
        letters = GTP_ALPH[digit] + letters
    return letters


def parse_token(token: str, size: int) -> Optional[tuple[int, int, bool]]:
    stripped = token.strip()
    if stripped.lower() in ("pass", "resign"):
        return None
    match = TOKEN_RE.match(stripped)
    if not match:
        return None
    x = 0
    for char in match.group(1).upper():
        digit = GTP_ALPH.find(char) + 1
        if digit == 0:
            return None
        x = x * len(GTP_ALPH) + digit
    x -= 1
    row = int(match.group(2)) - 1
    diamond = match.group(3) is not None
    if match.group(4) is not None and not diamond:
        return None
    if row < 0 or row >= size or x < 0 or x >= size:
        return None
    if diamond and (x >= size - 1 or row >= size - 1):
        return None
    return x, row, diamond


def field(text: str, name: str, cast):
    match = re.search(rf"\b{name}\s+([^\s]+)", text)
    if not match:
        return None
    try:
        return cast(match.group(1))
    except Exception:
        return None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    t = clamp01(t)
    rgb = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def format_visits(visits: Optional[int]) -> str:
    if visits is None:
        return ""
    if visits < 1_000:
        return str(visits)
    if visits < 100_000:
        return f"{visits / 1_000:.1f}k"
    if visits < 1_000_000:
        return f"{visits // 1_000}k"
    if visits < 10_000_000:
        return f"{visits / 1_000_000:.2f}m"
    if visits < 100_000_000:
        return f"{visits / 1_000_000:.1f}m"
    return f"{visits / 1_000_000:.0f}m"


def format_percent(value: Optional[float]) -> str:
    return "" if value is None else f"{value * 100:.1f}"


def format_noise(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def default_engine_path(script_dir: Path) -> Path:
    candidates = [
        script_dir / "../build-opencl/katago",
        script_dir / "../cpp/katago",
        script_dir / "engine/katago",
    ]
    for candidate in candidates:
        path = candidate.resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return candidates[0].resolve()


@dataclass(frozen=True)
class Move:
    player: str
    x: Optional[int]
    row: Optional[int]
    diamond: bool
    direction: str = ""

    def token(self) -> str:
        if self.x is None or self.row is None:
            return "pass"
        return f"{column_letters(self.x)}{self.row + 1}{'*' if self.diamond else ''}{self.direction if self.diamond else ''}"


@dataclass(frozen=True)
class AnalysisMove:
    token: str
    x: Optional[int]
    row: Optional[int]
    diamond: bool
    direction: str
    order: Optional[int]
    visits: Optional[int]
    winrate: Optional[float]
    prior: Optional[float]
    pv: tuple[tuple[int, int, bool], ...]


def parse_analysis_line(line: str, size: int) -> list[AnalysisMove]:
    if "info move " not in line:
        return []
    result: list[AnalysisMove] = []
    for match in INFO_REC_RE.finditer(line):
        token, rest = match.group(1), match.group(2)
        pv: tuple[tuple[int, int, bool], ...] = ()
        pv_pos = rest.find(" pv ")
        if pv_pos >= 0:
            parsed_pv: list[tuple[int, int, bool]] = []
            for item in rest[pv_pos + 4:].split():
                parsed = parse_token(item, size)
                if parsed is None:
                    break
                parsed_pv.append(parsed)
            pv = tuple(parsed_pv)
            rest = rest[:pv_pos]
        visits = field(rest, "visits", int)
        if visits == 0:
            continue
        parsed = parse_token(token, size)
        token_match = TOKEN_RE.match(token)
        result.append(AnalysisMove(
            token=token,
            x=parsed[0] if parsed else None,
            row=parsed[1] if parsed else None,
            diamond=parsed[2] if parsed else False,
            direction=token_match.group(4) if token_match and token_match.group(4) else "",
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
        size: int,
        interval_cs: int,
        analysis_wide_root_noise: float,
        on_error: Callable[[str], None],
    ):
        self.size = size
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
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._stdout, daemon=True).start()
        threading.Thread(target=self._stderr, daemon=True).start()
        self.send(f"boardsize {size}")
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
            parsed = parse_analysis_line(line, self.size)
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
        self.send(f"boardsize {self.size}")
        self.send("clear_board")
        for move in moves:
            self.send(f"play {move.player} {move.token()}")

    def set_board_size(self, size: int) -> None:
        self.stop_analysis(clear=True)
        self.size = size
        self.send(f"boardsize {size}")
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
        try:
            self.send("quit")
        finally:
            self.closed = True
            try:
                self.proc.terminate()
            except Exception:
                pass


class QuaxGUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.size = args.size
        self.pending_size = args.size
        self.analysis_wide_root_noise = args.analysis_wide_root_noise
        self.moves: list[Move] = []
        self.engine: Optional[Engine] = None
        self.engine_synced_moves: tuple[Move, ...] = ()
        self.analysis_enabled = False
        self.show_prior = False
        self.analysis_by_cell: dict[tuple[int, int, bool, str], AnalysisMove] = {}
        self.status_var = tk.StringVar()
        self.engine_var = tk.StringVar()
        self.analysis_button_text = tk.StringVar(value="Analyze")

        root.title(f"Quax{self.size}")
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
        self.canvas.bind("<Button-2>", self.click)
        self.canvas.bind("<Button-3>", self.click)

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
                self.args.engine, self.args.config, self.args.model, self.size,
                self.args.interval_cs, self.analysis_wide_root_noise,
                lambda _message: self.engine_var.set("engine: error"),
            )
        except Exception:
            self.engine_var.set("engine: failed")
            return
        self.engine_synced_moves = ()
        self.engine_var.set("engine: ready")

    def current_player(self) -> str:
        return "B" if len(self.moves) % 2 == 0 else "W"

    def occupied(self) -> dict[tuple[int, int, bool], str]:
        return {
            (move.x, move.row, move.diamond): move.player
            for move in self.moves if move.x is not None and move.row is not None
        }

    def _sync_engine_position(self) -> None:
        if self.engine is None:
            return
        moves = tuple(self.moves)
        if moves == self.engine_synced_moves:
            return
        self.engine.sync_position(self.moves)
        self.engine_synced_moves = moves

    def play(self, x: int, row: int, diamond: bool, direction: str = "") -> None:
        occupied = self.occupied()
        if (x, row, diamond) in occupied:
            return
        player = self.current_player()
        if diamond:
            if direction == "\\":
                endpoints = ((x, row, False), (x + 1, row + 1, False))
            elif direction == "/":
                endpoints = ((x + 1, row, False), (x, row + 1, False))
            else:
                return
            if any(occupied.get(endpoint) != player for endpoint in endpoints):
                return
        self.moves.append(Move(player, x, row, diamond, direction))
        self.analysis_by_cell.clear()
        if self.engine is not None:
            self._sync_engine_position()
            if self.analysis_enabled:
                self.engine.start_analysis(self.current_player())
        self.redraw()

    def play_pass(self) -> None:
        self.moves.append(Move(self.current_player(), None, None, False))
        self.analysis_by_cell.clear()
        if self.engine is not None:
            self._sync_engine_position()
            if self.analysis_enabled:
                self.engine.start_analysis(self.current_player())
        self.redraw()

    def undo(self) -> None:
        if not self.moves:
            return
        self.moves.pop()
        self.analysis_by_cell.clear()
        if self.engine is not None:
            self._sync_engine_position()
            if self.analysis_enabled:
                self.engine.start_analysis(self.current_player())
        self.redraw()

    def clear(self) -> None:
        self.moves.clear()
        self.analysis_by_cell.clear()
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

    def top_ranked_move(self) -> Optional[tuple[int, int, bool, str]]:
        occupied = self.occupied()
        best = None
        for key, analysis in self.analysis_by_cell.items():
            if key[:3] in occupied:
                continue
            order = analysis.order if analysis.order is not None else 10**9
            visits = analysis.visits if analysis.visits is not None else 0
            rank = (order, -visits, key[1], key[0], key[2], key[3])
            if best is None or rank < best[0]:
                best = (rank, key)
        return None if best is None else best[1]

    def play_top_ranked_move(self) -> None:
        move = self.top_ranked_move()
        if move is not None:
            self.play(*move)

    def change_pending_size(self, delta: int) -> None:
        self.pending_size = max(MIN_BOARD_SIZE, min(MAX_BOARD_SIZE, self.pending_size + delta))
        self.redraw()

    def apply_pending_size(self) -> None:
        if self.pending_size == self.size:
            return
        self.size = self.pending_size
        self.moves.clear()
        self.analysis_by_cell.clear()
        self.root.title(f"Quax{self.size}")
        if self.engine is not None:
            self.engine.set_board_size(self.size)
            self.engine_synced_moves = ()
            if self.analysis_enabled:
                self.engine.start_analysis(self.current_player())
        self.redraw()

    def step_analysis_wide_root_noise(self, direction: int) -> None:
        idx = min(
            range(len(ANALYSIS_WIDE_ROOT_NOISE_STEPS)),
            key=lambda i: abs(ANALYSIS_WIDE_ROOT_NOISE_STEPS[i] - self.analysis_wide_root_noise),
        )
        idx = max(0, min(len(ANALYSIS_WIDE_ROOT_NOISE_STEPS) - 1, idx + direction))
        self.analysis_wide_root_noise = ANALYSIS_WIDE_ROOT_NOISE_STEPS[idx]
        if self.engine is not None:
            self.engine.set_analysis_wide_root_noise(self.analysis_wide_root_noise)
            if self.analysis_enabled:
                self.engine.start_analysis(self.current_player(), clear=False)
        self.redraw()

    def _show_prior(self, _event) -> None:
        self.show_prior = True
        self.redraw()

    def _hide_prior(self, _event) -> None:
        self.show_prior = False
        self.redraw()

    def _poll_analysis(self) -> None:
        if self.engine is not None:
            if self.engine.proc.poll() is not None and not self.engine.closed:
                self.engine_var.set("engine: stopped")
            if self.analysis_enabled:
                self.analysis_by_cell = {
                    (item.x, item.row, item.diamond, item.direction): item
                    for item in self.engine.get_analysis()
                    if item.x is not None and item.row is not None
                }
                self.redraw()
        self.root.after(250, self._poll_analysis)

    def layout(self) -> tuple[float, float, float, float, float]:
        width = max(320, self.canvas.winfo_width())
        height = max(320, self.canvas.winfo_height())
        octagon_radius_ratio = 1.0 / (2.0 * math.cos(math.pi / 8.0))
        board_units = self.size - 1 + 2.0 * octagon_radius_ratio
        spacing = max(26.0, min((width - 90) / board_units, (height - 90) / board_units, 90.0))
        octagon_radius = spacing * octagon_radius_ratio
        diamond_radius = spacing / (2.0 + math.sqrt(2.0))
        span = (self.size - 1) * spacing
        return (width - span) / 2.0, (height - span) / 2.0, spacing, octagon_radius, diamond_radius

    @staticmethod
    def center(x: int, row: int, origin_x: float, origin_y: float, spacing: float) -> tuple[float, float]:
        return origin_x + x * spacing, origin_y + row * spacing

    @staticmethod
    def diamond_center(x: int, row: int, origin_x: float, origin_y: float, spacing: float) -> tuple[float, float]:
        return origin_x + (x + 0.5) * spacing, origin_y + (row + 0.5) * spacing

    @staticmethod
    def polygon(cx: float, cy: float, radius: float, sides: int, offset: float = 0.0) -> list[float]:
        points: list[float] = []
        for i in range(sides):
            angle = offset + 2 * math.pi * i / sides
            points.extend((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        return points

    def draw_borders(self, origin_x: float, origin_y: float, spacing: float) -> None:
        half_spacing = spacing / 2.0
        left = origin_x - half_spacing
        top = origin_y - half_spacing
        right = origin_x + (self.size - 1) * spacing + half_spacing
        bottom = origin_y + (self.size - 1) * spacing + half_spacing
        width = max(4.0, spacing * 0.06)
        self.canvas.create_line(left, top, right, top, fill=BLACK, width=width)
        self.canvas.create_line(left, bottom, right, bottom, fill=BLACK, width=width)
        self.canvas.create_line(left, top, left, bottom, fill=WHITE, width=width)
        self.canvas.create_line(right, top, right, bottom, fill=WHITE, width=width)

    def analysis_fill(
        self,
        key: tuple[int, int, bool],
        empty_rgb: tuple[int, int, int],
        visit_denom: float,
    ) -> str:
        candidates = [
            analysis for analysis_key, analysis in self.analysis_by_cell.items()
            if analysis_key[:3] == key
        ]
        if not candidates:
            return lerp_rgb(empty_rgb, empty_rgb, 0.0)
        analysis = min(candidates, key=lambda item: item.order if item.order is not None else 10**9)
        if self.show_prior and analysis.prior is not None:
            return lerp_rgb(ANALYSIS_LOW, ANALYSIS_HIGH, clamp01(analysis.prior) ** 0.9)
        if analysis.visits is not None and analysis.winrate is not None:
            t = 0.0 if visit_denom <= 0 else math.log(max(1, analysis.visits)) / visit_denom
            t = clamp01(t) ** 1.1
            t = clamp01(t + 0.35 * (analysis.winrate - 0.5) * 2.0)
            return lerp_rgb(ANALYSIS_LOW, ANALYSIS_HIGH, t)
        return lerp_rgb(empty_rgb, empty_rgb, 0.0)

    def redraw(self) -> None:
        self.canvas.delete("all")
        origin_x, origin_y, spacing, octagon_radius, diamond_radius = self.layout()
        occupied = self.occupied()
        pending = f" (pending {self.pending_size})" if self.pending_size != self.size else ""
        self.status_var.set(
            f"size: {self.size}{pending}    to move: {self.current_player()}    "
            f"moves: {len(self.moves)}    awrn: {format_noise(self.analysis_wide_root_noise)}"
        )
        self.analysis_button_text.set("Stop" if self.analysis_enabled else "Analyze")
        top_cell = self.top_ranked_move()
        top_analysis = self.analysis_by_cell.get(top_cell) if top_cell is not None else None
        top_visits = top_analysis.visits if top_analysis and top_analysis.visits else 0
        visit_denom = math.log(max(2, top_visits))
        octagon_stone_font = tkfont.Font(
            family="Helvetica", size=max(7, int(octagon_radius * 0.46)), weight="bold",
        )
        diamond_stone_font = tkfont.Font(
            family="Helvetica", size=max(7, int(diamond_radius * 0.46)), weight="bold",
        )
        octagon_analysis_font = tkfont.Font(
            family="Helvetica", size=max(6, int(octagon_radius * 0.32)), weight="bold",
        )
        diamond_analysis_font = tkfont.Font(
            family="Helvetica", size=max(5, int(diamond_radius * 0.24)), weight="bold",
        )

        for row in range(self.size):
            for x in range(self.size):
                key = (x, row, False)
                cx, cy = self.center(x, row, origin_x, origin_y, spacing)
                player = occupied.get(key)
                fill = BLACK if player == "B" else WHITE if player == "W" else self.analysis_fill(
                    key, EMPTY_OCTAGON, visit_denom,
                )
                self.canvas.create_polygon(
                    self.polygon(cx, cy, octagon_radius, 8, math.pi / 8),
                    fill=fill, outline=OUTLINE, width=1,
                )
                if row == 0:
                    self.canvas.create_text(cx, cy - octagon_radius - 18, text=column_letters(x).lower(), fill="#4d463b")
                if x == 0:
                    self.canvas.create_text(cx - octagon_radius - 18, cy, text=str(row + 1), fill="#4d463b")

        for row in range(self.size - 1):
            for x in range(self.size - 1):
                key = (x, row, True)
                cx, cy = self.diamond_center(x, row, origin_x, origin_y, spacing)
                player = occupied.get(key)
                fill = BLACK if player == "B" else WHITE if player == "W" else self.analysis_fill(
                    key, EMPTY_DIAMOND, visit_denom,
                )
                self.canvas.create_polygon(
                    self.polygon(cx, cy, diamond_radius, 4),
                    fill=fill, outline=OUTLINE, width=1,
                )

        self.draw_borders(origin_x, origin_y, spacing)

        moves_by_diamond = {(move.x, move.row): move for move in self.moves if move.diamond}
        for row in range(self.size - 1):
            for x in range(self.size - 1):
                player = occupied.get((x, row, True))
                if player is None:
                    continue
                color = BLACK if player == "B" else WHITE
                move = moves_by_diamond[(x, row)]
                a, b = (((x, row), (x + 1, row + 1)) if move.direction == "\\" else ((x + 1, row), (x, row + 1)))
                if occupied.get((*a, False)) == player and occupied.get((*b, False)) == player:
                    self.canvas.create_line(
                        *self.center(*a, origin_x, origin_y, spacing),
                        *self.center(*b, origin_x, origin_y, spacing),
                        fill=color, width=max(5, spacing * 0.09),
                    )
                cx, cy = self.diamond_center(x, row, origin_x, origin_y, spacing)
                delta = diamond_radius * 0.72
                if move.direction == "\\":
                    points = (cx - delta, cy - delta, cx + delta, cy + delta)
                else:
                    points = (cx + delta, cy - delta, cx - delta, cy + delta)
                self.canvas.create_line(*points, fill=TEXT_ON_STONE, width=max(2, diamond_radius * 0.12))

        for move_number, move in enumerate(self.moves, start=1):
            if move.x is None or move.row is None:
                continue
            cx, cy = (
                self.diamond_center(move.x, move.row, origin_x, origin_y, spacing)
                if move.diamond else self.center(move.x, move.row, origin_x, origin_y, spacing)
            )
            font = diamond_stone_font if move.diamond else octagon_stone_font
            self.canvas.create_text(cx, cy, text=str(move_number), fill=TEXT_ON_STONE, font=font)

        for key, analysis in self.analysis_by_cell.items():
            if key[:3] in occupied:
                continue
            x, row, diamond, direction = key
            cx, cy = (
                self.diamond_center(x, row, origin_x, origin_y, spacing)
                if diamond else self.center(x, row, origin_x, origin_y, spacing)
            )
            radius = diamond_radius if diamond else octagon_radius
            font = diamond_analysis_font if diamond else octagon_analysis_font
            if diamond:
                cy += (-0.20 if direction == "\\" else 0.20) * radius
                prefix = direction if direction else "?"
                if self.show_prior:
                    text = format_percent(analysis.prior)
                    if text:
                        self.canvas.create_text(cx, cy, text=f"{prefix}{text}", fill="#111111", font=font)
                else:
                    winrate = format_percent(analysis.winrate)
                    visits = format_visits(analysis.visits)
                    text = " ".join(part for part in (f"{prefix}{winrate}", visits) if part)
                    if text:
                        self.canvas.create_text(cx, cy, text=text, fill="#111111", font=font)
                continue
            if self.show_prior:
                text = format_percent(analysis.prior)
                if text:
                    self.canvas.create_text(cx, cy, text=text, fill="#111111", font=font)
            else:
                winrate = format_percent(analysis.winrate)
                visits = format_visits(analysis.visits)
                gap = max(7, radius * 0.30)
                if winrate:
                    self.canvas.create_text(cx, cy - gap / 2, text=winrate, fill="#111111", font=font)
                if visits:
                    self.canvas.create_text(cx, cy + gap / 2, text=visits, fill="#111111", font=font)

    def click(self, event: tk.Event) -> None:
        origin_x, origin_y, spacing, _, diamond_radius = self.layout()
        candidates: list[tuple[float, int, int, bool]] = []
        for row in range(self.size):
            for x in range(self.size):
                cx, cy = self.center(x, row, origin_x, origin_y, spacing)
                dx, dy = abs(event.x - cx), abs(event.y - cy)
                if dx <= spacing / 2 and dy <= spacing / 2 and dx + dy <= spacing / math.sqrt(2):
                    candidates.append((dx * dx + dy * dy, x, row, False))
        for row in range(self.size - 1):
            for x in range(self.size - 1):
                cx, cy = self.diamond_center(x, row, origin_x, origin_y, spacing)
                dx, dy = abs(event.x - cx), abs(event.y - cy)
                if dx + dy <= diamond_radius:
                    candidates.append((dx * dx + dy * dy, x, row, True))
        if candidates:
            _, x, row, diamond = min(candidates)
            direction = "/" if diamond and event.num in (2, 3) else "\\" if diamond else ""
            self.play(x, row, diamond, direction)

    def close(self) -> None:
        if self.engine is not None:
            self.engine.close()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Official Quax board and KataGomo analysis GUI")
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--engine", type=Path, default=default_engine_path(script_dir))
    parser.add_argument("--config", type=Path, default=script_dir / "o11/gtp.local.cfg")
    parser.add_argument("--model", type=Path, default=script_dir / "o11/data/latest.bin.gz")
    parser.add_argument("--interval-cs", type=int, default=20)
    parser.add_argument(
        "--analysis-wide-root-noise", "--awrn",
        dest="analysis_wide_root_noise", type=float,
        default=DEFAULT_ANALYSIS_WIDE_ROOT_NOISE,
    )
    parser.add_argument("--no-engine", action="store_true")
    args = parser.parse_args()
    if not MIN_BOARD_SIZE <= args.size <= MAX_BOARD_SIZE:
        parser.error(f"--size must be between {MIN_BOARD_SIZE} and {MAX_BOARD_SIZE}")
    if not ANALYSIS_WIDE_ROOT_NOISE_STEPS[0] <= args.analysis_wide_root_noise <= ANALYSIS_WIDE_ROOT_NOISE_STEPS[-1]:
        parser.error(
            f"--analysis-wide-root-noise must be between {ANALYSIS_WIDE_ROOT_NOISE_STEPS[0]} "
            f"and {ANALYSIS_WIDE_ROOT_NOISE_STEPS[-1]}"
        )
    return args


def main() -> None:
    args = parse_args()
    args.engine = args.engine.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    root = tk.Tk()
    QuaxGUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
