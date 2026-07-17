#include "benty.h"
#include "board.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>

using namespace std;

namespace BentY {

namespace {

struct UnionFind {
  vector<int> parent;

  explicit UnionFind(int size) : parent(size) {
    for(int i = 0; i < size; i++)
      parent[i] = i;
  }

  int find(int value) {
    if(parent[value] != value)
      parent[value] = find(parent[value]);
    return parent[value];
  }

  void joinKeepingFirst(int keep, int drop) {
    keep = find(keep);
    drop = find(drop);
    if(keep != drop)
      parent[drop] = keep;
  }
};

int refIndex(int copy, int row, int col, int n) {
  return (copy * (n + 1) + row) * (n + 1) + col;
}

}

bool isSupportedTensorLen(int tensorLen) {
  return tensorLen >= 7 && tensorLen <= Board::MAX_LEN && (tensorLen - 1) % 6 == 0;
}

int tensorLenForWidth(int width) {
  if(width < 5 || (width - 1) % 4 != 0)
    throw invalid_argument("Unsupported Bent-Y width");
  int tensorLen = (3 * width - 1) / 2;
  if(!isSupportedTensorLen(tensorLen))
    throw invalid_argument("Unsupported Bent-Y width");
  return tensorLen;
}

int widthForTensorLen(int tensorLen) {
  if(!isSupportedTensorLen(tensorLen))
    throw invalid_argument("Unsupported Bent-Y tensor length");
  return (2 * tensorLen + 1) / 3;
}

Topology::Topology(int tensorLength)
  : tensorLen(tensorLength),
    width(widthForTensorLen(tensorLength)),
    frequency(width - 1),
    playable(tensorLength * tensorLength, false),
    sideMasks(tensorLength * tensorLength, 0),
    neighbors(tensorLength * tensorLength),
    symPos(tensorLength * tensorLength),
    layers(),
    edges() {

  const int n = frequency;
  const int k = n / 2;
  const int overlapRows = k + 1;
  const int refCount = 3 * (n + 1) * (n + 1);
  vector<int> refs(refCount, -1);
  vector<pair<int,int>> vertexCoords;
  vector<int> vertexAtPos(tensorLen * tensorLen, -1);
  set<pair<int,int>> initialEdges;

  auto placedCoords = [k](int copy, int row, int col) {
    if(copy == 0)
      return make_pair(col + k, row + k);
    if(copy == 1)
      return make_pair(col, row);
    return make_pair(col + k, row);
  };

  auto getVertex = [&](int copy, int row, int col) {
    int& stored = refs[refIndex(copy, row, col, n)];
    if(stored >= 0)
      return stored;
    pair<int,int> coords = placedCoords(copy,row,col);
    int pos = coords.second * tensorLen + coords.first;
    assert(coords.first >= 0 && coords.first < tensorLen);
    assert(coords.second >= 0 && coords.second < tensorLen);
    if(vertexAtPos[pos] < 0) {
      vertexAtPos[pos] = (int)vertexCoords.size();
      vertexCoords.push_back(coords);
    }
    stored = vertexAtPos[pos];
    return stored;
  };

  auto addInitialEdge = [&](int a, int b) {
    if(a != b)
      initialEdges.insert(minmax(a, b));
  };

  for(int copy = 0; copy < 3; copy++) {
    for(int row = 0; row <= n; row++) {
      for(int col = 0; col <= row; col++) {
        int a = getVertex(copy,row,col);
        if(col < row)
          addInitialEdge(a, getVertex(copy, row, col+1));
        if(row < n) {
          addInitialEdge(a, getVertex(copy, row+1, col));
          addInitialEdge(a, getVertex(copy,row+1,col+1));
        }
      }
    }
  }

  UnionFind unions((int)vertexCoords.size());
  auto refVertex = [&](int copy, int row, int col) {
    int value = refs[refIndex(copy,row,col,n)];
    assert(value >= 0);
    return value;
  };

  for(int i = 1; i <= k; i++) {
    int row = k - i;
    unions.joinKeepingFirst(refVertex(1, row, row), refVertex(2, row, 0));
  }
  for(int row = overlapRows; row <= n; row++)
    unions.joinKeepingFirst(refVertex(0, row, row), refVertex(2, n, row));
  for(int row = overlapRows; row <= n; row++)
    unions.joinKeepingFirst(refVertex(0, row, 0), refVertex(1, n, n-row));

  vector<int> rootToVertex(vertexCoords.size(), -1);
  vector<pair<int,int>> compactCoords;
  vector<int> oldToCompact(vertexCoords.size(), -1);
  for(int old = 0; old < (int)vertexCoords.size(); old++) {
    int root = unions.find(old);
    if(rootToVertex[root] < 0) {
      rootToVertex[root] = (int)compactCoords.size();
      compactCoords.push_back(vertexCoords[root]);
    }
    oldToCompact[old] = rootToVertex[root];
  }
  for(int& ref: refs) {
    if(ref >= 0)
      ref = oldToCompact[ref];
  }

  set<pair<int,int>> compactEdges;
  for(const auto& edge: initialEdges) {
    int a = oldToCompact[edge.first];
    int b = oldToCompact[edge.second];
    if(a != b)
      compactEdges.insert(minmax(a, b));
  }

  vector<int> vertexPos(compactCoords.size(), -1);
  for(int vertex = 0; vertex < (int)compactCoords.size(); vertex++) {
    int x = compactCoords[vertex].first;
    int y = compactCoords[vertex].second;
    int pos = y * tensorLen + x;
    assert(!playable[pos]);
    playable[pos] = true;
    vertexPos[vertex] = pos;
  }

  edges.assign(compactEdges.begin(), compactEdges.end());
  for(auto& edge: edges) {
    int firstPos = vertexPos[edge.first];
    int secondPos = vertexPos[edge.second];
    edge = minmax(firstPos, secondPos);
    neighbors[edge.first].push_back(edge.second);
    neighbors[edge.second].push_back(edge.first);
  }
  int degreeFiveCount = 0;
  for(int pos = 0; pos < tensorLen * tensorLen; pos++) {
    if(playable[pos]) {
      sort(neighbors[pos].begin(), neighbors[pos].end());
      assert(neighbors[pos].size() >= 3);
      assert(neighbors[pos].size() <= 6);
      if(neighbors[pos].size() == 5)
        degreeFiveCount++;
    }
  }
  assert(degreeFiveCount == 3);

  auto compactRef = [&](int copy, int row, int col) {
    int vertex = refs[refIndex(copy,row,col,n)];
    assert(vertex >= 0);
    return vertex;
  };

  vector<bool> assigned(compactCoords.size(), false);
  auto isHubShellVertex = [&](int vertex) {
    for(int row = 0; row <= n; row++) {
      for(int col = 0; col <= row; col++) {
        if(compactRef(0,row,col) != vertex)
          continue;
        if((row == 0 && col == 0) ||
           (col == 0 && row <= k) ||
           row == k ||
           (col == row && row > 0 && row <= k))
          return true;
      }
    }
    return false;
  };

  auto commitLayer = [&](const vector<int>& candidateVertices) {
    vector<int> layer;
    vector<bool> seen(compactCoords.size(), false);
    for(int vertex: candidateVertices) {
      if(vertex < 0 || seen[vertex] || assigned[vertex])
        continue;
      seen[vertex] = true;
      assigned[vertex] = true;
      layer.push_back(vertexPos[vertex]);
    }
    if(!layer.empty())
      layers.push_back(layer);
  };

  for(int gridRing = 0; gridRing < k; gridRing++) {
    vector<int> ring;
    for(int row = gridRing; row <= n-1; row++) {
      int vertex = compactRef(1,row,gridRing);
      if(!isHubShellVertex(vertex))
        ring.push_back(vertex);
    }
    int baseRow = n - gridRing;
    if(baseRow >= overlapRows) {
      for(int col = 0; col <= baseRow; col++) {
        int vertex = compactRef(0,baseRow,col);
        if(!isHubShellVertex(vertex))
          ring.push_back(vertex);
      }
    }
    for(int row = n-1; row >= gridRing+1; row--) {
      int vertex = compactRef(2,row,row-gridRing);
      if(!isHubShellVertex(vertex))
        ring.push_back(vertex);
    }
    commitLayer(ring);
  }

  vector<int> hubShell;
  hubShell.push_back(compactRef(0,0,0));
  for(int row = 1; row < k; row++)
    hubShell.push_back(compactRef(0,row,0));
  for(int col = 0; col <= k; col++)
    hubShell.push_back(compactRef(0,k,col));
  for(int row = k-1; row >= 1; row--)
    hubShell.push_back(compactRef(0,row,row));
  commitLayer(hubShell);

  int spineCol = 1;
  int bottomRow = overlapRows - 2;
  while(true) {
    int topRow = 2 * spineCol;
    if(topRow >= bottomRow)
      break;
    vector<int> ring;
    for(int row = topRow; row <= bottomRow; row++)
      ring.push_back(compactRef(0,row,spineCol));
    for(int col = spineCol; col <= bottomRow-spineCol; col++)
      ring.push_back(compactRef(0,bottomRow,col));
    for(int i = bottomRow-topRow; i >= 0; i--)
      ring.push_back(compactRef(0,topRow+i,spineCol+i));
    commitLayer(ring);
    spineCol++;
    bottomRow--;
  }
  int centerRow = 2 * spineCol;
  if(centerRow >= bottomRow)
    commitLayer({compactRef(0,centerRow,spineCol)});

  vector<int> missing;
  for(int vertex = 0; vertex < (int)compactCoords.size(); vertex++) {
    if(!assigned[vertex])
      missing.push_back(vertex);
  }
  commitLayer(missing);

  auto addSide = [&](int copy, int row, int col, int mask) {
    sideMasks[vertexPos[compactRef(copy,row,col)]] |= mask;
  };
  for(int row = 0; row < n; row++)
    addSide(1,row,0,1);
  addSide(0,n,0,1);
  for(int col = 0; col <= n; col++)
    addSide(0,n,col,2);
  addSide(0,n,n,4);
  for(int row = n-1; row >= 1; row--)
    addSide(2,row,row,4);
  addSide(1,0,0,4);

  array<int,3> corners = {-1,-1,-1};
  const int cornerMasks[3] = {5,3,6};
  for(int pos = 0; pos < tensorLen * tensorLen; pos++) {
    symPos[pos].fill(pos);
    for(int corner = 0; corner < 3; corner++) {
      if(playable[pos] && sideMasks[pos] == cornerMasks[corner]) {
        assert(corners[corner] < 0);
        corners[corner] = pos;
      }
    }
  }

  vector<array<int,3>> cornerDistances(tensorLen * tensorLen);
  for(auto& distances: cornerDistances)
    distances.fill(-1);
  for(int corner = 0; corner < 3; corner++) {
    assert(corners[corner] >= 0);
    queue<int> pending;
    cornerDistances[corners[corner]][corner] = 0;
    pending.push(corners[corner]);
    while(!pending.empty()) {
      int pos = pending.front();
      pending.pop();
      for(int next: neighbors[pos]) {
        if(cornerDistances[next][corner] < 0) {
          cornerDistances[next][corner] = cornerDistances[pos][corner] + 1;
          pending.push(next);
        }
      }
    }
  }

  map<array<int,3>,int> posByCornerDistances;
  for(int pos = 0; pos < tensorLen * tensorLen; pos++) {
    if(!playable[pos])
      continue;
    assert(cornerDistances[pos][0] >= 0 && cornerDistances[pos][1] >= 0 && cornerDistances[pos][2] >= 0);
    bool inserted = posByCornerDistances.emplace(cornerDistances[pos],pos).second;
    assert(inserted);
  }

  static const int cornerPermutations[6][3] = {
    {0,1,2}, {1,2,0}, {2,0,1}, {0,2,1}, {1,0,2}, {2,1,0},
  };
  for(int pos = 0; pos < tensorLen * tensorLen; pos++) {
    if(!playable[pos])
      continue;
    for(int symmetry = 0; symmetry < 6; symmetry++) {
      array<int,3> transformedDistances;
      for(int corner = 0; corner < 3; corner++)
        transformedDistances[cornerPermutations[symmetry][corner]] = cornerDistances[pos][corner];
      auto match = posByCornerDistances.find(transformedDistances);
      assert(match != posByCornerDistances.end());
      symPos[pos][symmetry] = match->second;
    }
  }

  int assignedCount = 0;
  for(bool value: assigned)
    assignedCount += value ? 1 : 0;
  assert(assignedCount == (int)compactCoords.size());
  int m = (tensorLen - 1) / 6;
  assert(compactCoords.size() == (size_t)(20 * m * m + 6 * m + 1));
  assert(edges.size() == (size_t)(60 * m * m + 6 * m));

  set<pair<int,int>> edgeSet(edges.begin(),edges.end());
  for(int symmetry = 0; symmetry < 6; symmetry++) {
    set<pair<int,int>> transformedEdges;
    for(const auto& edge: edges)
      transformedEdges.insert(minmax(symPos[edge.first][symmetry], symPos[edge.second][symmetry]));
    assert(transformedEdges == edgeSet);
  }
}

const Topology& getTopology(int tensorLen) {
  if(!isSupportedTensorLen(tensorLen))
    throw invalid_argument("Unsupported Bent-Y tensor length");
  static const vector<Topology> topologies = []() {
    vector<Topology> result;
    for(int len = 7; len <= Board::MAX_LEN; len += 6)
      result.emplace_back(len);
    return result;
  }();
  return topologies[(tensorLen - 7) / 6];
}

}
