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


GTP_ALPH = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
MOVE_TOKEN_RE = re.compile(r"^\s*([A-Za-z]+)\s*([0-9]+)\s*$")
INFO_REC_RE = re.compile(r"\binfo\s+move\s+(\S+)\s+(.*?)(?=\binfo\s+move\s+\S+\s+|$)", re.DOTALL)
BOARD_EMPTY = (231, 221, 202)
LEGAL_EMPTY = (207, 207, 203)
ANALYSIS_LOW = (217, 212, 205)
ANALYSIS_HIGH = (170, 125, 210)
RED_PLAYER_FILL = "#dc3c3c"
BLUE_PLAYER_FILL = "#2864dc"
TEXT_ON_PLAYER = "#fafafa"
MIN_BOARD_SIZE = 4
MAX_BOARD_SIZE = 19
ANALYSIS_WIDE_ROOT_NOISE_STEPS = (0.00, 0.01, 0.02, 0.04, 0.10, 0.20, 0.50, 1.00, 2.00)
DEFAULT_ANALYSIS_WIDE_ROOT_NOISE = 0.20


@dataclass(frozen=True)
class Move:
    pla: str
    col: Optional[int]
    row: Optional[int]


@dataclass(frozen=True)
class AnalysisMove:
    move: str
    col: Optional[int]
    row: Optional[int]
    order: Optional[int]
    visits: Optional[int]
    winrate: Optional[float]
    prior: Optional[float]
    pv: tuple[tuple[int, int], ...]


def board_to_engine_vertex(col: int, row: int) -> str:
    x = 2 * col + row - 2
    y = 2 * row - 1
    return f"({x},{y})"


def col_to_human_letters(col: int) -> str:
    out: list[str] = []
    while col > 0:
        col -= 1
        out.append(chr(ord("a") + (col % 26)))
        col //= 26
    return "".join(reversed(out))


def move_to_human_token(move: Move) -> str:
    if move.col is None or move.row is None:
        return "pass"
    return f"{col_to_human_letters(move.col)}{move.row}"


def gtp_letters_to_int(s: str) -> int:
    out = 0
    for ch in s.upper():
        idx = GTP_ALPH.find(ch)
        if idx < 0:
            return 0
        out = out * len(GTP_ALPH) + idx + 1
    return out


def parse_analysis_move_token(tok: str, board_n: int) -> Optional[tuple[int, int]]:
    stripped = tok.strip()
    if stripped.lower() in ("pass", "resign"):
        return None
    match = MOVE_TOKEN_RE.match(stripped)
    if not match:
        return None
    p = gtp_letters_to_int(match.group(1))
    num = int(match.group(2))
    if p <= 0 or num <= 0 or num % 2:
        return None
    row_from_bottom = num // 2
    row = board_n + 1 - row_from_bottom
    tmp = p - row + 1
    if tmp % 2:
        return None
    col = tmp // 2
    if not (1 <= col <= board_n and 1 <= row <= board_n):
        return None
    return col, row


def _field(rest: str, name: str, cast):
    match = re.search(rf"\b{name}\s+([^\s]+)", rest)
    if not match:
        return None
    try:
        return cast(match.group(1))
    except Exception:
        return None


def parse_kata_analyze_line(line: str, board_n: int) -> list[AnalysisMove]:
    if "info move " not in line:
        return []
    out: list[AnalysisMove] = []
    for match in INFO_REC_RE.finditer(line):
        move_token, rest = match.group(1), match.group(2)
        pv: tuple[tuple[int, int], ...] = ()
        pv_pos = rest.find(" pv ")
        if pv_pos >= 0:
            pv_items: list[tuple[int, int]] = []
            for token in rest[pv_pos + 4:].strip().split():
                coord = parse_analysis_move_token(token, board_n)
                if coord is None:
                    break
                pv_items.append(coord)
            pv = tuple(pv_items)
            rest = rest[:pv_pos]
        visits = _field(rest, "visits", int)
        if visits == 0:
            continue
        coord = parse_analysis_move_token(move_token, board_n)
        out.append(AnalysisMove(
            move=move_token,
            col=coord[0] if coord is not None else None,
            row=coord[1] if coord is not None else None,
            order=_field(rest, "order", int),
            visits=visits,
            winrate=_field(rest, "winrate", float),
            prior=_field(rest, "prior", float),
            pv=pv,
        ))
    out.sort(key=lambda item: item.order if item.order is not None else 10**9)
    return out


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = clamp01(t)
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def rgb_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def clamp_analysis_wide_root_noise(value: float) -> float:
    return max(ANALYSIS_WIDE_ROOT_NOISE_STEPS[0], min(ANALYSIS_WIDE_ROOT_NOISE_STEPS[-1], value))


def fmt_analysis_wide_root_noise(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def fmt_visits(visits: Optional[int]) -> str:
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
    if visits < 100_000_000:
        return f"{visits / 1_000_000:.1f}m"
    return f"{visits / 1_000_000:.0f}m"


def fmt_winrate(winrate: Optional[float]) -> str:
    if winrate is None:
        return ""
    return f"{winrate * 100:.1f}"


def fmt_prior(prior: Optional[float]) -> str:
    if prior is None:
        return ""
    return f"{prior * 100:.1f}"


def default_engine_path(script_dir: Path) -> Path:
    candidates = (
        script_dir / "../build-opencl/katago",
        script_dir / "../cpp/katago",
        script_dir / "engine/katago",
    )
    for candidate in candidates:
        path = candidate.resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return candidates[0].resolve()


class KataEngine:
    def __init__(
        self,
        *,
        engine: Path,
        config: Path,
        model: Path,
        board_n: int,
        interval_cs: int,
        analysis_wide_root_noise: float,
        on_error: Callable[[str], None],
    ):
        self.board_n = board_n
        self.interval_cs = interval_cs
        self.analysis_wide_root_noise = clamp_analysis_wide_root_noise(analysis_wide_root_noise)
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
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self.send(f"boardsize {board_n}")
        self.send("clear_board")
        self.set_analysis_wide_root_noise(self.analysis_wide_root_noise)

    def send(self, command: str) -> None:
        if self.closed or self.proc.poll() is not None:
            return
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(command + "\n")
            self.proc.stdin.flush()
        except Exception as exc:
            self.on_error(str(exc))

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            with self.lock:
                if not self.analysis_active:
                    continue
                if self.analysis_mute_until_sync:
                    if line.lstrip().startswith("="):
                        self.analysis_mute_until_sync = False
                    continue
            parsed = parse_kata_analyze_line(line, self.board_n)
            if parsed:
                with self.lock:
                    if self.analysis_active and not self.analysis_mute_until_sync:
                        self.analysis = parsed

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            line = line.strip()
            if line:
                self.stderr_tail.append(line)

    def sync_position(self, moves: list[Move]) -> None:
        self.stop_analysis(clear=True)
        self.send(f"boardsize {self.board_n}")
        self.send("clear_board")
        for move in moves:
            vertex = "pass" if move.col is None or move.row is None else board_to_engine_vertex(move.col, move.row)
            self.send(f"play {move.pla} {vertex}")

    def set_board_size(self, board_n: int) -> None:
        self.stop_analysis(clear=True)
        self.board_n = board_n
        self.send(f"boardsize {board_n}")
        self.send("clear_board")

    def set_analysis_wide_root_noise(self, value: float) -> None:
        self.analysis_wide_root_noise = clamp_analysis_wide_root_noise(value)
        if (
            self.analysis_wide_root_noise_sent is not None
            and abs(self.analysis_wide_root_noise_sent - self.analysis_wide_root_noise) < 1e-9
        ):
            return
        self.analysis_wide_root_noise_sent = self.analysis_wide_root_noise
        self.send(f"kata-set-param analysisWideRootNoise {fmt_analysis_wide_root_noise(self.analysis_wide_root_noise)}")

    def start_analysis(self, pla: str, *, clear: bool = True) -> None:
        self.set_analysis_wide_root_noise(self.analysis_wide_root_noise)
        self.send(f"kata-analyze {pla} {self.interval_cs}")
        with self.lock:
            if clear:
                self.analysis = []
            self.analysis_active = True
            self.analysis_mute_until_sync = True

    def stop_analysis(self, *, clear: bool = False) -> None:
        with self.lock:
            self.analysis_active = False
            self.analysis_mute_until_sync = False
            if clear:
                self.analysis = []
        self.send("stop")

    def get_analysis(self) -> list[AnalysisMove]:
        with self.lock:
            return list(self.analysis)

    def close(self) -> None:
        try:
            self.send("quit")
        except Exception:
            pass
        self.closed = True
        try:
            self.proc.terminate()
        except Exception:
            pass


class HexGui:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.board_n = args.size
        self.pending_size = self.board_n
        self.analysis_wide_root_noise = clamp_analysis_wide_root_noise(args.analysis_wide_root_noise)
        self.moves: list[Move] = []
        self.board: dict[tuple[int, int], str] = {}
        self.winner: Optional[str] = None
        self.hover: Optional[tuple[int, int]] = None
        self.analysis_enabled = False
        self.show_prior = False
        self.analysis_by_cell: dict[tuple[int, int], AnalysisMove] = {}
        self.engine: Optional[KataEngine] = None
        self.engine_synced_moves: tuple[Move, ...] = ()
        self.turn_var = tk.StringVar()
        self.engine_var = tk.StringVar(value="engine: disabled")
        self.analysis_button_text = tk.StringVar(value="Analyze")

        self.root.title(f"2v2 Hex {self.board_n}")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<space>", self._toggle_analysis_from_space)
        self.root.bind("<BackSpace>", lambda _event: self.undo())
        self.root.bind("z", lambda _event: self.undo())
        self.root.bind("<KeyPress-C>", lambda _event: self.clear())
        self.root.bind("<KeyPress-P>", lambda _event: self.play_pass())
        self.root.bind("<Control-c>", lambda _event: self.copy_hexboard_diagram())
        self.root.bind("<Command-c>", lambda _event: self.copy_hexboard_diagram())
        self.root.bind("<KeyPress-comma>", lambda _event: self.play_top_ranked_move())
        self.root.bind("q", lambda _event: self.close())
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.bind("<Control-d>", lambda _event: self.close())
        self.root.bind("<KeyPress-t>", self._show_prior)
        self.root.bind("<KeyRelease-t>", self._hide_prior)
        self.root.bind("<KeyPress-equal>", lambda _event: self.change_pending_size(1))
        self.root.bind("<KeyPress-plus>", lambda _event: self.change_pending_size(1))
        self.root.bind("<KeyPress-KP_Add>", lambda _event: self.change_pending_size(1))
        self.root.bind("<KeyPress-minus>", lambda _event: self.change_pending_size(-1))
        self.root.bind("<KeyPress-underscore>", lambda _event: self.change_pending_size(-1))
        self.root.bind("<KeyPress-KP_Subtract>", lambda _event: self.change_pending_size(-1))
        self.root.bind("<KeyPress-bracketleft>", lambda _event: self.step_analysis_wide_root_noise(-1))
        self.root.bind("<KeyPress-bracketright>", lambda _event: self.step_analysis_wide_root_noise(1))
        self.root.bind("<Return>", lambda _event: self.apply_pending_size())
        self.root.bind("<KP_Enter>", lambda _event: self.apply_pending_size())

        self._build_widgets()
        self._try_start_engine()
        self.redraw()
        self.root.after(250, self._poll_analysis)

    def _build_widgets(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, textvariable=self.analysis_button_text, command=self.toggle_analysis).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=(6, 0))
        for button in toolbar.winfo_children():
            button.bind("<space>", self._toggle_analysis_from_space)
        ttk.Label(toolbar, textvariable=self.turn_var).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Label(toolbar, textvariable=self.engine_var).pack(side=tk.RIGHT)

        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(body, width=760, height=680, bg="#f2efe8", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", lambda _event: self._set_hover(None))

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
            self.engine = KataEngine(
                engine=self.args.engine,
                config=self.args.config,
                model=self.args.model,
                board_n=self.board_n,
                interval_cs=self.args.interval_cs,
                analysis_wide_root_noise=self.analysis_wide_root_noise,
                on_error=lambda _message: self.engine_var.set("engine: error"),
            )
        except Exception:
            self.engine_var.set("engine: failed")
            return
        self.engine_synced_moves = ()
        self.engine_var.set("engine: ready")

    def _sync_engine_position(self) -> None:
        if self.engine is None:
            return
        moves = tuple(self.moves)
        if moves == self.engine_synced_moves:
            return
        self.engine.sync_position(self.moves)
        self.engine_synced_moves = moves

    @property
    def phase(self) -> int:
        return len(self.moves) % 4

    def current_player(self) -> str:
        return "B" if self.phase % 2 == 0 else "W"

    def in_region(self, col: int, row: int, phase: Optional[int] = None) -> bool:
        x, y = col - 1, row - 1
        phase = self.phase if phase is None else phase
        if phase == 0:
            return y <= x
        if phase == 1:
            return x + y >= self.board_n - 1
        if phase == 2:
            return y >= x
        return x + y <= self.board_n - 1

    def legal(self, col: int, row: int) -> bool:
        return self.winner is None and (col, row) not in self.board and self.in_region(col, row)

    def has_placement(self, phase: Optional[int] = None) -> bool:
        return any(
            (col, row) not in self.board and self.in_region(col, row, phase)
            for row in range(1, self.board_n + 1)
            for col in range(1, self.board_n + 1)
        )

    def connected(self, pla: str) -> bool:
        if pla == "B":
            frontier = [(col, 1) for col in range(1, self.board_n + 1) if self.board.get((col, 1)) == pla]
            target = lambda _col, row: row == self.board_n
        else:
            frontier = [(1, row) for row in range(1, self.board_n + 1) if self.board.get((1, row)) == pla]
            target = lambda col, _row: col == self.board_n
        seen = set(frontier)
        while frontier:
            col, row = frontier.pop()
            if target(col, row):
                return True
            for dc, dr in ((0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)):
                cell = (col + dc, row + dr)
                if cell not in seen and self.board.get(cell) == pla:
                    seen.add(cell)
                    frontier.append(cell)
        return False

    def play(self, col: int, row: int) -> None:
        if not self.legal(col, row):
            return
        pla = self.current_player()
        self.moves.append(Move(pla, col, row))
        self.board[(col, row)] = pla
        if self.connected(pla):
            self.winner = pla
        self.position_changed()

    def play_pass(self) -> None:
        if self.winner is not None:
            return
        pla = self.current_player()
        self.moves.append(Move(pla, None, None))
        self.position_changed()

    def rebuild(self) -> None:
        saved = list(self.moves)
        self.moves = []
        self.board = {}
        self.winner = None
        for move in saved:
            if self.winner is not None:
                break
            if move.col is None:
                self.moves.append(move)
            else:
                self.moves.append(move)
                self.board[(move.col, move.row)] = move.pla
                if self.connected(move.pla):
                    self.winner = move.pla

    def undo(self) -> None:
        if self.moves:
            self.moves.pop()
            self.rebuild()
            self.position_changed()

    def clear(self) -> None:
        self.moves = []
        self.board = {}
        self.winner = None
        self.position_changed()

    def position_changed(self) -> None:
        self.analysis_by_cell = {}
        if self.winner is not None:
            self.analysis_enabled = False
        if self.engine is not None:
            self._sync_engine_position()
            if self.analysis_enabled and self.winner is None:
                self.engine.start_analysis(self.current_player())
        self.redraw()

    def top_ranked_move(self) -> Optional[tuple[int, int]]:
        best: Optional[tuple[tuple[int, int, int, int], int, int]] = None
        for (col, row), analysis in self.analysis_by_cell.items():
            if not self.legal(col, row):
                continue
            order = analysis.order if analysis.order is not None else 1_000_000
            visits = analysis.visits if analysis.visits is not None else 0
            key = (order, -visits, row, col)
            if best is None or key < best[0]:
                best = (key, col, row)
        return None if best is None else (best[1], best[2])

    def play_top_ranked_move(self) -> None:
        if self.winner is None and not self.has_placement():
            self.play_pass()
            return
        move = self.top_ranked_move()
        if move is not None:
            self.play(*move)

    def set_analysis_wide_root_noise(self, value: float) -> None:
        value = clamp_analysis_wide_root_noise(value)
        if abs(self.analysis_wide_root_noise - value) < 1e-9:
            return
        self.analysis_wide_root_noise = value
        if self.engine is not None:
            self.engine.set_analysis_wide_root_noise(value)
            if self.analysis_enabled and self.winner is None:
                self.engine.start_analysis(self.current_player(), clear=False)
        self.redraw()

    def step_analysis_wide_root_noise(self, direction: int) -> None:
        idx = min(
            range(len(ANALYSIS_WIDE_ROOT_NOISE_STEPS)),
            key=lambda i: abs(ANALYSIS_WIDE_ROOT_NOISE_STEPS[i] - self.analysis_wide_root_noise),
        )
        idx = max(0, idx - 1) if direction < 0 else min(len(ANALYSIS_WIDE_ROOT_NOISE_STEPS) - 1, idx + 1)
        self.set_analysis_wide_root_noise(ANALYSIS_WIDE_ROOT_NOISE_STEPS[idx])

    def build_hexboard_diagram(self) -> str:
        rows = [
            f"{'R' if move.pla == 'B' else 'B'} {idx}:{move_to_human_token(move)}"
            for idx, move in enumerate(self.moves, 1)
        ]
        contents = ("\n" + " " * 12).join(rows)
        return f'<hexboard size="{self.board_n}x{self.board_n}"\n  contents="{contents}"\n  />'

    def copy_hexboard_diagram(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.build_hexboard_diagram())

    def _toggle_analysis_from_space(self, _event: tk.Event) -> str:
        self.toggle_analysis()
        return "break"

    def toggle_analysis(self) -> None:
        if self.winner is not None:
            return
        if self.engine is None:
            self._try_start_engine()
            if self.engine is None:
                return
        self.analysis_enabled = not self.analysis_enabled
        if self.analysis_enabled and self.winner is None:
            self._sync_engine_position()
            self.engine.start_analysis(self.current_player(), clear=False)
        else:
            self.engine.stop_analysis()
        self.redraw()

    def change_pending_size(self, delta: int) -> None:
        old = self.pending_size
        self.pending_size = max(MIN_BOARD_SIZE, min(MAX_BOARD_SIZE, self.pending_size + delta))
        if self.pending_size != old:
            self.redraw()

    def apply_pending_size(self) -> None:
        if self.pending_size == self.board_n:
            return
        self.board_n = self.pending_size
        self.moves = []
        self.board = {}
        self.winner = None
        self.analysis_by_cell = {}
        self.root.title(f"2v2 Hex {self.board_n}")
        if self.engine is not None:
            self.engine.set_board_size(self.board_n)
            self.engine_synced_moves = ()
            if self.analysis_enabled:
                self.engine.start_analysis(self.current_player())
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
                self.analysis_by_cell = {
                    (item.col, item.row): item
                    for item in self.engine.get_analysis()
                    if item.col is not None and item.row is not None
                }
                self.redraw()
        self.root.after(250, self._poll_analysis)

    def _layout(self) -> tuple[float, float, float]:
        width = max(400, self.canvas.winfo_width())
        height = max(400, self.canvas.winfo_height())
        sq3 = math.sqrt(3)
        radius = min(
            (width - 120) / (1.5 * sq3 * (self.board_n - 1) + sq3),
            (height - 120) / (1.5 * (self.board_n - 1) + 2),
        )
        radius = max(9, min(42, radius))
        board_w = radius * (1.5 * sq3 * (self.board_n - 1) + sq3)
        board_h = radius * (1.5 * (self.board_n - 1) + 2)
        return (width - board_w) / 2 + radius * sq3 / 2, (height - board_h) / 2 + radius, radius

    def _center(self, col: int, row: int, origin_x: float, origin_y: float, radius: float) -> tuple[float, float]:
        return origin_x + radius * math.sqrt(3) * (col - 1 + (row - 1) / 2), origin_y + radius * 1.5 * (row - 1)

    def _poly(self, cx: float, cy: float, radius: float) -> list[float]:
        pts: list[float] = []
        for deg in (90, 30, -30, -90, -150, 150):
            angle = math.radians(deg)
            pts.extend([cx + radius * math.cos(angle), cy + radius * math.sin(angle)])
        return pts

    def _draw_borders(self, origin_x: float, origin_y: float, radius: float) -> None:
        specs = (
            (RED_PLAYER_FILL, range(1, self.board_n + 1), lambda i: (i, 1), ((2, 3), (3, 4))),
            (RED_PLAYER_FILL, range(1, self.board_n + 1), lambda i: (i, self.board_n), ((5, 0), (0, 1))),
            (BLUE_PLAYER_FILL, range(1, self.board_n + 1), lambda i: (1, i), ((4, 5), (5, 0))),
            (BLUE_PLAYER_FILL, range(1, self.board_n + 1), lambda i: (self.board_n, i), ((1, 2), (2, 3))),
        )
        width = max(3.0, radius * 0.12)
        for color, indices, cell, pairs in specs:
            for i in indices:
                col, row = cell(i)
                cx, cy = self._center(col, row, origin_x, origin_y, radius)
                points = self._poly(cx, cy, radius)
                for first, second in pairs:
                    self.canvas.create_line(
                        points[2 * first], points[2 * first + 1],
                        points[2 * second], points[2 * second + 1],
                        fill=color,
                        width=width,
                    )

    def _draw_active_dot(self, origin_x: float, origin_y: float, radius: float) -> None:
        corners = ((self.board_n, 1), (self.board_n, self.board_n), (1, self.board_n), (1, 1))
        col, row = corners[self.phase]
        cx, cy = self._center(col, row, origin_x, origin_y, radius)
        board_cx, board_cy = self._center((self.board_n + 1) / 2, (self.board_n + 1) / 2, origin_x, origin_y, radius)
        dx, dy = cx - board_cx, cy - board_cy
        length = math.hypot(dx, dy)
        cx += dx / length * radius * 1.25
        cy += dy / length * radius * 1.25
        r = max(4, radius * 0.18)
        self.canvas.create_oval(
            cx - r,
            cy - r,
            cx + r,
            cy + r,
            fill=RED_PLAYER_FILL if self.current_player() == "B" else BLUE_PLAYER_FILL,
            outline="#ffffff",
            width=2,
        )

    def redraw(self) -> None:
        self.canvas.delete("all")
        origin_x, origin_y, radius = self._layout()
        pending_text = f" (pending {self.pending_size})" if self.pending_size != self.board_n else ""
        awrn = fmt_analysis_wide_root_noise(self.analysis_wide_root_noise)
        if self.winner is not None:
            turn_text = f"{'Red/obtuse' if self.winner == 'B' else 'Blue/acute'} wins"
        else:
            forced = " — forced pass" if not self.has_placement() else ""
            turn_text = f"to move: {self.current_player()}{forced}"
        self.turn_var.set(f"size: {self.board_n}{pending_text}    {turn_text}    moves: {len(self.moves)}    awrn: {awrn}")
        self.analysis_button_text.set("Stop" if self.analysis_enabled else "Analyze")

        top_cell = self.top_ranked_move()
        top_analysis = self.analysis_by_cell.get(top_cell) if top_cell is not None else None
        top_visits = top_analysis.visits if top_analysis is not None and top_analysis.visits is not None else 0
        denom = math.log(max(2, top_visits))
        board_font = tkfont.Font(family="Helvetica", size=max(7, int(radius * 0.42)), weight="bold")
        coord_font = tkfont.Font(family="Helvetica", size=max(8, int(radius * 0.42)))
        stone_font = tkfont.Font(family="Helvetica", size=max(8, int(radius * 0.52)), weight="bold")

        for row in range(1, self.board_n + 1):
            for col in range(1, self.board_n + 1):
                cx, cy = self._center(col, row, origin_x, origin_y, radius)
                pla = self.board.get((col, row))
                fill_rgb = BOARD_EMPTY
                if pla == "B":
                    fill = RED_PLAYER_FILL
                elif pla == "W":
                    fill = BLUE_PLAYER_FILL
                else:
                    fill = rgb_hex(LEGAL_EMPTY if self.legal(col, row) else fill_rgb)
                analysis = self.analysis_by_cell.get((col, row)) if self.legal(col, row) else None
                if analysis is not None:
                    if self.show_prior and analysis.prior is not None:
                        fill_rgb = lerp_rgb(ANALYSIS_LOW, ANALYSIS_HIGH, clamp01(analysis.prior) ** 0.9)
                        fill = rgb_hex(fill_rgb)
                    elif analysis.visits is not None and analysis.winrate is not None:
                        strength = 0.0 if denom <= 0 else math.log(max(1, analysis.visits)) / denom
                        strength = clamp01(strength) ** 1.1
                        strength = clamp01(strength + 0.35 * (analysis.winrate - 0.5) * 2.0)
                        fill_rgb = lerp_rgb(ANALYSIS_LOW, ANALYSIS_HIGH, strength)
                        fill = rgb_hex(fill_rgb)
                self.canvas.create_polygon(
                    self._poly(cx, cy, radius * 0.96),
                    fill=fill,
                    outline="#8a7b63",
                    width=1,
                    tags=(f"cell-{col}-{row}",),
                )
                if pla is not None:
                    index = next(i for i, move in enumerate(self.moves, 1) if (move.col, move.row) == (col, row))
                    self.canvas.create_text(cx, cy, text=str(index), fill=TEXT_ON_PLAYER, font=stone_font)
                    continue
                if analysis is not None:
                    if self.show_prior:
                        label = fmt_prior(analysis.prior)
                        if label:
                            self.canvas.create_text(cx, cy, text=label, fill="#111111", font=board_font)
                        continue
                    winrate = fmt_winrate(analysis.winrate)
                    visits = fmt_visits(analysis.visits)
                    gap = max(12, radius * 0.42)
                    if winrate:
                        self.canvas.create_text(cx, cy - gap / 2, text=winrate, fill="#111111", font=board_font)
                    if visits:
                        self.canvas.create_text(cx, cy + gap / 2, text=visits, fill="#111111", font=board_font)
        if self.hover is not None:
            cx, cy = self._center(*self.hover, origin_x, origin_y, radius)
            self.canvas.create_polygon(
                self._poly(cx, cy, radius * 0.82),
                fill="",
                outline="#222" if self.legal(*self.hover) else "#a02020",
                width=2,
            )
        self._draw_borders(origin_x, origin_y, radius)
        if self.winner is None:
            self._draw_active_dot(origin_x, origin_y, radius)
        for col in range(1, self.board_n + 1):
            cx, cy = self._center(col, 0, origin_x, origin_y, radius)
            self.canvas.create_text(cx, cy, text=col_to_human_letters(col), fill="#4d463b", font=coord_font)
        for row in range(1, self.board_n + 1):
            cx, cy = self._center(0, row, origin_x, origin_y, radius)
            self.canvas.create_text(cx, cy, text=str(row), fill="#4d463b", font=coord_font)

    def _nearest_cell(self, x: float, y: float) -> Optional[tuple[int, int]]:
        origin_x, origin_y, radius = self._layout()
        cells = (
            (math.hypot(x - cx, y - cy), col, row)
            for row in range(1, self.board_n + 1)
            for col in range(1, self.board_n + 1)
            for cx, cy in (self._center(col, row, origin_x, origin_y, radius),)
        )
        distance, col, row = min(cells)
        return (col, row) if distance <= radius else None

    def _on_canvas_click(self, event) -> None:
        cell = self._nearest_cell(event.x, event.y)
        if cell is not None:
            self.play(*cell)

    def _on_canvas_motion(self, event) -> None:
        self._set_hover(self._nearest_cell(event.x, event.y))

    def _set_hover(self, cell: Optional[tuple[int, int]]) -> None:
        if cell != self.hover:
            self.hover = cell
            self.redraw()

    def close(self) -> None:
        if self.engine is not None:
            self.engine.close()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="2v2 Hex board and KataGo analysis GUI")
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--engine", type=Path, default=default_engine_path(script_dir))
    parser.add_argument("--config", type=Path, default=script_dir / "2v2-11/gtp.local.cfg")
    parser.add_argument("--model", type=Path, default=script_dir / "2v2-11/data/latest.bin.gz")
    parser.add_argument("--interval-cs", type=int, default=20)
    parser.add_argument(
        "--analysis-wide-root-noise",
        "--awrn",
        dest="analysis_wide_root_noise",
        type=float,
        default=DEFAULT_ANALYSIS_WIDE_ROOT_NOISE,
    )
    parser.add_argument("--no-engine", action="store_true")
    args = parser.parse_args()
    if not MIN_BOARD_SIZE <= args.size <= MAX_BOARD_SIZE:
        parser.error(f"--size must be between {MIN_BOARD_SIZE} and {MAX_BOARD_SIZE}")
    if not ANALYSIS_WIDE_ROOT_NOISE_STEPS[0] <= args.analysis_wide_root_noise <= ANALYSIS_WIDE_ROOT_NOISE_STEPS[-1]:
        parser.error(
            f"--analysis-wide-root-noise must be between {ANALYSIS_WIDE_ROOT_NOISE_STEPS[0]} and {ANALYSIS_WIDE_ROOT_NOISE_STEPS[-1]}"
        )
    args.engine = args.engine.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    return args


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    HexGui(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
