from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class BentYTopology:
    tensor_len: int
    width: int
    frequency: int
    playable: tuple[int, ...]
    neighbors: tuple[tuple[int, ...], ...]
    side_masks: tuple[int, ...]
    layers: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int], ...]
    sym_pos: tuple[tuple[int, ...], ...]


def tensor_len_for_width(width: int) -> int:
    if width < 5 or (width - 1) % 4 != 0:
        raise ValueError(f"Unsupported Bent-Y width: {width}")
    return (3 * width - 1) // 2


def width_for_tensor_len(tensor_len: int) -> int:
    if tensor_len < 7 or (tensor_len - 1) % 6 != 0:
        raise ValueError(f"Unsupported Bent-Y tensor length: {tensor_len}")
    return (2 * tensor_len + 1) // 3


@lru_cache(maxsize=None)
def build_topology(tensor_len: int) -> BentYTopology:
    width = width_for_tensor_len(tensor_len)
    n = width - 1
    k = n // 2
    overlap_rows = k + 1
    refs: dict[tuple[int, int, int], int] = {}
    coords: list[tuple[int, int]] = []
    vertex_at: dict[tuple[int, int], int] = {}
    initial_edges: set[tuple[int, int]] = set()

    def placed(copy: int, row: int, col: int) -> tuple[int, int]:
        if copy == 0:
            return col + k, row + k
        if copy == 1:
            return col, row
        return col + k, row

    def vertex(copy: int, row: int, col: int) -> int:
        ref = (copy, row, col)
        if ref in refs:
            return refs[ref]
        point = placed(copy, row, col)
        if point not in vertex_at:
            vertex_at[point] = len(coords)
            coords.append(point)
        refs[ref] = vertex_at[point]
        return refs[ref]

    def edge(a: int, b: int) -> None:
        if a != b:
            initial_edges.add(tuple(sorted((a, b))))

    for copy in range(3):
        for row in range(n + 1):
            for col in range(row + 1):
                a = vertex(copy, row, col)
                if col < row:
                    edge(a, vertex(copy, row, col + 1))
                if row < n:
                    edge(a, vertex(copy, row + 1, col))
                    edge(a, vertex(copy, row + 1, col + 1))

    parent = list(range(len(coords)))

    def find(value: int) -> int:
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def join_keep_first(keep: int, drop: int) -> None:
        keep = find(keep)
        drop = find(drop)
        if keep != drop:
            parent[drop] = keep

    for i in range(1, k + 1):
        row = k - i
        join_keep_first(vertex(1, row, row), vertex(2, row, 0))
    for row in range(overlap_rows, n + 1):
        join_keep_first(vertex(0, row, row), vertex(2, n, row))
    for row in range(overlap_rows, n + 1):
        join_keep_first(vertex(0, row, 0), vertex(1, n, n - row))

    root_to_vertex: dict[int, int] = {}
    compact_coords: list[tuple[int, int]] = []
    old_to_compact: list[int] = []
    for old in range(len(coords)):
        root = find(old)
        if root not in root_to_vertex:
            root_to_vertex[root] = len(compact_coords)
            compact_coords.append(coords[root])
        old_to_compact.append(root_to_vertex[root])
    refs = {ref: old_to_compact[old] for ref, old in refs.items()}

    compact_edges = {
        tuple(sorted((old_to_compact[a], old_to_compact[b])))
        for a, b in initial_edges
        if old_to_compact[a] != old_to_compact[b]
    }
    vertex_pos = [y * tensor_len + x for x, y in compact_coords]
    playable = tuple(sorted(vertex_pos))
    edge_positions = tuple(sorted(tuple(sorted((vertex_pos[a], vertex_pos[b]))) for a, b in compact_edges))
    neighbor_lists: list[list[int]] = [[] for _ in range(tensor_len * tensor_len)]
    for a, b in edge_positions:
        neighbor_lists[a].append(b)
        neighbor_lists[b].append(a)
    neighbors = tuple(tuple(sorted(items)) for items in neighbor_lists)

    def ref_vertex(copy: int, row: int, col: int) -> int:
        return refs[(copy, row, col)]

    def is_hub_shell(value: int) -> bool:
        for row in range(n + 1):
            for col in range(row + 1):
                if ref_vertex(0, row, col) != value:
                    continue
                if ((row == 0 and col == 0) or (col == 0 and row <= k) or row == k or (col == row and 0 < row <= k)):
                    return True
        return False

    assigned: set[int] = set()
    vertex_layers: list[list[int]] = []

    def commit(candidates: list[int]) -> None:
        layer: list[int] = []
        seen: set[int] = set()
        for value in candidates:
            if value in seen or value in assigned:
                continue
            seen.add(value)
            assigned.add(value)
            layer.append(value)
        if layer:
            vertex_layers.append(layer)

    for grid_ring in range(k):
        ring: list[int] = []
        for row in range(grid_ring, n):
            value = ref_vertex(1, row, grid_ring)
            if not is_hub_shell(value):
                ring.append(value)
        base_row = n - grid_ring
        if base_row >= overlap_rows:
            for col in range(base_row + 1):
                value = ref_vertex(0, base_row, col)
                if not is_hub_shell(value):
                    ring.append(value)
        for row in range(n - 1, grid_ring, -1):
            value = ref_vertex(2, row, row - grid_ring)
            if not is_hub_shell(value):
                ring.append(value)
        commit(ring)

    hub_shell = [ref_vertex(0, 0, 0)]
    hub_shell.extend(ref_vertex(0, row, 0) for row in range(1, k))
    hub_shell.extend(ref_vertex(0, k, col) for col in range(k + 1))
    hub_shell.extend(ref_vertex(0, row, row) for row in range(k - 1, 0, -1))
    commit(hub_shell)

    spine_col = 1
    bottom_row = overlap_rows - 2
    while True:
        top_row = 2 * spine_col
        if top_row >= bottom_row:
            break
        ring = [ref_vertex(0, row, spine_col) for row in range(top_row, bottom_row + 1)]
        ring.extend(ref_vertex(0, bottom_row, col) for col in range(spine_col, bottom_row - spine_col + 1))
        ring.extend(ref_vertex(0, top_row + i, spine_col + i) for i in range(bottom_row - top_row, -1, -1))
        commit(ring)
        spine_col += 1
        bottom_row -= 1
    center_row = 2 * spine_col
    if center_row >= bottom_row:
        commit([ref_vertex(0, center_row, spine_col)])
    commit([value for value in range(len(compact_coords)) if value not in assigned])
    layers = tuple(tuple(vertex_pos[value] for value in layer) for layer in vertex_layers)

    side_masks = [0] * (tensor_len * tensor_len)

    def add_side(copy: int, row: int, col: int, mask: int) -> None:
        side_masks[vertex_pos[ref_vertex(copy, row, col)]] |= mask

    for row in range(n):
        add_side(1, row, 0, 1)
    add_side(0, n, 0, 1)
    for col in range(n + 1):
        add_side(0, n, col, 2)
    add_side(0, n, n, 4)
    for row in range(n - 1, 0, -1):
        add_side(2, row, row, 4)
    add_side(1, 0, 0, 4)

    corner_masks = (5, 3, 6)
    corners = []
    for mask in corner_masks:
        matches = [pos for pos in playable if side_masks[pos] == mask]
        assert len(matches) == 1
        corners.append(matches[0])

    corner_distances = [[-1, -1, -1] for _ in range(tensor_len * tensor_len)]
    for corner, start in enumerate(corners):
        corner_distances[start][corner] = 0
        pending = deque([start])
        while pending:
            pos = pending.popleft()
            for next_pos in neighbors[pos]:
                if corner_distances[next_pos][corner] < 0:
                    corner_distances[next_pos][corner] = corner_distances[pos][corner] + 1
                    pending.append(next_pos)

    pos_by_distances = {tuple(corner_distances[pos]): pos for pos in playable}
    assert len(pos_by_distances) == len(playable)
    corner_permutations = (
        (0, 1, 2), (1, 2, 0), (2, 0, 1),
        (0, 2, 1), (1, 0, 2), (2, 1, 0),
    )
    mutable_sym = [[pos] * 6 for pos in range(tensor_len * tensor_len)]
    for pos in playable:
        for symmetry, permutation in enumerate(corner_permutations):
            transformed_distances = [-1, -1, -1]
            for corner in range(3):
                transformed_distances[permutation[corner]] = corner_distances[pos][corner]
            mutable_sym[pos][symmetry] = pos_by_distances[tuple(transformed_distances)]
    sym_pos = tuple(tuple(items) for items in mutable_sym)

    m = (tensor_len - 1) // 6
    assert len(playable) == 20 * m * m + 6 * m + 1
    assert len(edge_positions) == 60 * m * m + 6 * m
    assert len(assigned) == len(playable)
    assert sum(len(neighbors[pos]) == 5 for pos in playable) == 3
    assert all(3 <= len(neighbors[pos]) <= 6 for pos in playable)
    edge_set = set(edge_positions)
    for symmetry in range(6):
        transformed = {tuple(sorted((sym_pos[a][symmetry], sym_pos[b][symmetry]))) for a, b in edge_positions}
        assert transformed == edge_set

    return BentYTopology(
        tensor_len=tensor_len,
        width=width,
        frequency=n,
        playable=playable,
        neighbors=neighbors,
        side_masks=tuple(side_masks),
        layers=layers,
        edges=edge_positions,
        sym_pos=sym_pos,
    )
