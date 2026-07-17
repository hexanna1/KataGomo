#ifndef BENTY_GRAPH_H
#define BENTY_GRAPH_H

#include <algorithm>
#include <array>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>
#include <utility>
#include <vector>

namespace BentYGraph {

struct Geometry {
  int tensorLen;
  std::vector<int> points;
  // Query-major, then key, then the ten invariant relation features.
  std::vector<float> features;
};

inline Geometry buildGeometry(int tensorLen) {
  const int n = 2 * (tensorLen - 1) / 3;
  const int k = n / 2;
  std::map<std::array<int,3>,int> refs;
  std::map<std::pair<int,int>,int> vertexAt;
  std::vector<std::pair<int,int>> coords;
  std::set<std::pair<int,int>> edges;
  auto vertex = [&](int copy, int row, int col) {
    const std::array<int,3> ref = {copy,row,col};
    auto found = refs.find(ref);
    if(found != refs.end())
      return found->second;
    const std::pair<int,int> point = copy == 0 ? std::make_pair(col+k,row+k) :
      copy == 1 ? std::make_pair(col,row) : std::make_pair(col+k,row);
    auto inserted = vertexAt.emplace(point,static_cast<int>(coords.size()));
    if(inserted.second)
      coords.push_back(point);
    refs.emplace(ref,inserted.first->second);
    return inserted.first->second;
  };
  auto edge = [&](int a, int b) {
    if(a != b)
      edges.insert(std::minmax(a,b));
  };
  for(int copy = 0; copy < 3; copy++) {
    for(int row = 0; row <= n; row++) {
      for(int col = 0; col <= row; col++) {
        int a = vertex(copy,row,col);
        if(col < row)
          edge(a,vertex(copy,row,col+1));
        if(row < n) {
          edge(a,vertex(copy,row+1,col));
          edge(a,vertex(copy,row+1,col+1));
        }
      }
    }
  }
  std::vector<int> parent(coords.size());
  std::iota(parent.begin(),parent.end(),0);
  auto find = [&](int value) {
    while(parent[value] != value) {
      parent[value] = parent[parent[value]];
      value = parent[value];
    }
    return value;
  };
  auto join = [&](int keep, int drop) {
    parent[find(drop)] = find(keep);
  };
  for(int i = 1; i <= k; i++) {
    const int row = k-i;
    join(refs.at({1,row,row}),refs.at({2,row,0}));
  }
  for(int row = k+1; row <= n; row++) {
    join(refs.at({0,row,row}),refs.at({2,n,row}));
    join(refs.at({0,row,0}),refs.at({1,n,n-row}));
  }
  std::vector<int> positions(coords.size());
  Geometry geometry;
  geometry.tensorLen = tensorLen;
  for(size_t old = 0; old < coords.size(); old++) {
    auto point = coords[find(static_cast<int>(old))];
    positions[old] = point.second*tensorLen + point.first;
    geometry.points.push_back(positions[old]);
  }
  std::sort(geometry.points.begin(),geometry.points.end());
  geometry.points.erase(std::unique(geometry.points.begin(),geometry.points.end()),geometry.points.end());
  const int count = static_cast<int>(geometry.points.size());
  std::vector<int> tokenAt(tensorLen*tensorLen,-1);
  for(int i = 0; i < count; i++)
    tokenAt[geometry.points[i]] = i;
  std::vector<std::set<int>> adjacent(count);
  for(auto endpoints: edges) {
    const int a = tokenAt[positions[endpoints.first]];
    const int b = tokenAt[positions[endpoints.second]];
    if(a != b) {
      adjacent[a].insert(b);
      adjacent[b].insert(a);
    }
  }
  std::vector<int> sides(count,0);
  auto side = [&](int copy, int row, int col, int mask) {
    sides[tokenAt[positions[refs.at({copy,row,col})]]] |= mask;
  };
  for(int row = 0; row < n; row++)
    side(1,row,0,1);
  side(0,n,0,1);
  for(int col = 0; col <= n; col++)
    side(0,n,col,2);
  side(0,n,n,4);
  for(int row = n-1; row > 0; row--)
    side(2,row,row,4);
  side(1,0,0,4);
  const std::array<int,3> cornerMasks = {5,3,6};
  std::array<int,3> corners;
  for(int corner = 0; corner < 3; corner++)
    corners[corner] = static_cast<int>(std::find(sides.begin(),sides.end(),cornerMasks[corner])-sides.begin());
  std::vector<int> distances(count*count,-1);
  for(int start = 0; start < count; start++) {
    std::queue<int> pending;
    distances[start*count+start] = 0;
    pending.push(start);
    while(!pending.empty()) {
      const int current = pending.front();
      pending.pop();
      for(int next: adjacent[current]) {
        if(distances[start*count+next] < 0) {
          distances[start*count+next] = distances[start*count+current]+1;
          pending.push(next);
        }
      }
    }
  }
  const float scale = static_cast<float>(distances[corners[0]*count+corners[1]]);
  geometry.features.reserve(count*count*10);
  for(int query = 0; query < count; query++) {
    for(int key = 0; key < count; key++) {
      const int d = distances[query*count+key];
      geometry.features.push_back(1.0f/(1.0f+d));
      geometry.features.push_back(d == 0 ? 1.0f : 0.0f);
      geometry.features.push_back(d == 1 ? 1.0f : 0.0f);
      geometry.features.push_back(d == 2 ? 1.0f : 0.0f);
      std::array<std::pair<int,int>,3> pairs;
      for(int corner = 0; corner < 3; corner++)
        pairs[corner] = {distances[corners[corner]*count+query],distances[corners[corner]*count+key]};
      std::sort(pairs.begin(),pairs.end());
      for(auto pair: pairs) {
        geometry.features.push_back(pair.first/scale);
        geometry.features.push_back(pair.second/scale);
      }
    }
  }
  return geometry;
}

inline const Geometry& geometry(int tensorLen) {
  static std::mutex mutex;
  static std::map<int,Geometry> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto found = cache.find(tensorLen);
  if(found == cache.end())
    found = cache.emplace(tensorLen,buildGeometry(tensorLen)).first;
  return found->second;
}

}

#endif
