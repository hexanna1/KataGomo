# Kata-Y

This branch adapts KataGomo for Y on three board shapes. Select the shape with
`boardShape`: `y`, `obtuseY`, or `bentY`.

On every shape, players alternately place stones and win when one connected
group touches all three sides of the board. There is no scoring, capture, or
randomness.

The shapes differ in geometry:

- `y` is the standard triangular board;
- `obtuseY` is a hexhex board whose six boundary edges are grouped into three
  adjacent pairs, with each pair acting as one goal side;
- `bentY` is a threefold-symmetric board with three degree-five vertices.

For the standard shape, the board is the top-left triangular half of the Hex
rhombus. On an `N = 14` board, legal human-facing cells are `a1` through `n1`,
then `a2` through `m2`, and so on down to `a14`. In zero-indexed engine
coordinates, playable cells satisfy:

```text
x >= 0, y >= 0, and x + y < N
```
