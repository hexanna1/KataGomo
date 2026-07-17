#ifndef GAME_BENTY_H_
#define GAME_BENTY_H_

#include <array>
#include <utility>
#include <vector>

namespace BentY {

struct Topology {
  int tensorLen;
  int width;
  int frequency;
  std::vector<bool> playable;
  std::vector<int> sideMasks;
  std::vector<std::vector<int>> neighbors;
  std::vector<std::array<int,6>> symPos;
  std::vector<std::vector<int>> layers;
  std::vector<std::pair<int,int>> edges;

  explicit Topology(int tensorLength);
};

bool isSupportedTensorLen(int tensorLen);
int tensorLenForWidth(int width);
int widthForTensorLen(int tensorLen);
const Topology& getTopology(int tensorLen);

}

#endif
