#ifndef GAME_BENTY_H_
#define GAME_BENTY_H_

#include <array>
#include <vector>

namespace BentY {

struct Topology {
  std::vector<bool> playable;
  std::vector<int> sideMasks;
  std::vector<std::vector<int>> neighbors;
  std::vector<std::array<int,6>> symPos;

  explicit Topology(int tensorLen);
};

bool isSupportedTensorLen(int tensorLen);
const Topology& getTopology(int tensorLen);

}

#endif
