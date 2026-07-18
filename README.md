# KataHex

This branch adapts KataGomo for Hex and related variants. Select the game with
`hexVariant`: `hex` for standard Hex, `2v2` for four-seat team Hex, or `hexhex`
for multi-cell moves.

Standard Hex uses the usual rules: Black connects the top and bottom sides,
White connects the left and right sides, and stones are never captured.

In 2v2 Hex, four seats move clockwise from the top-right corner and opposite
seats form a team. The top-right and bottom-left obtuse seats play on their
inclusive sides of the long diagonal. The bottom-right and top-left acute seats
play on their inclusive sides of the short diagonal. Red occupies the obtuse
seats and moves first. Team colors still alternate, while the active seat has
period four. If a seat has no legal placement, play advances to the next seat.

Allowing Red to occupy either pair of corners, either team to move first, and
either direction of play gives eight rulesets. Board and color symmetries pair
them into four equivalence classes:

| Red-first representative | Equivalent Blue-first ruleset | Implemented |
| --- | --- | --- |
| Red at obtuse corners, clockwise | Blue at acute corners, counterclockwise | yes |
| Red at obtuse corners, counterclockwise | Blue at acute corners, clockwise | no |
| Red at acute corners, clockwise | Blue at obtuse corners, counterclockwise | no |
| Red at acute corners, counterclockwise | Blue at obtuse corners, clockwise | yes (via an initial Red pass) |

No symmetry identifies two different rows, but an initial Red pass followed by
the corresponding symmetry transform links the first and fourth rows, and
likewise the second and third. The `2v2` variant fixes the first representative;
randomized openings make the network robust to the phase change needed to play
the fourth.

In Hexhex, a move chooses an on-board center and fills every empty cell among
the center and its six neighbors. Occupied cells retain their existing colors,
and neighbors outside the board are ignored. A center is legal when its clipped
footprint contains at least one empty cell. The connection goals are the same as
in standard Hex.
