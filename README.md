# Kata-Quax

This branch adapts KataGomo for Quax. Select the ruleset with `quaxVariant`:
`double`, `single`, or `official`.

Players alternately claim octagons and diamonds, and win by connecting their
opposite sides. Black connects top to bottom and White connects left to right.
Octagons use coordinates such as `A1`; the diamond below and to the right is
`A1*`. Directional diamond moves append `\` or `/`, for example `A1*\` and
`A1*/`.

The variants differ in how diamonds create diagonal connections:

- `double`: a diamond connects both diagonal endpoint pairs and may be played
  without owning its endpoints.
- `single`: a diamond selects one diagonal direction and may be played without
  owning its endpoints; draws are possible.
- `official`: a diamond selects one diagonal direction and is legal only when
  both endpoints already contain the moving player's stones.

There is no scoring or capture.
