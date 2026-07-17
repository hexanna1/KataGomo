#pragma once

#include <algorithm>
#include <map>
#include <vector>

namespace BentYGraph {

inline bool hasAttention(const std::vector<std::pair<int,unique_ptr_void>>& blocks) {
  for(const auto& block : blocks) {
    if(block.first == TRANSFORMER_ATTENTION_BLOCK_KIND &&
       static_cast<const TransformerAttentionDesc*>(block.second.get())->graphAttention)
      return true;
    if(block.first == NESTED_BOTTLENECK_BLOCK_KIND &&
       hasAttention(static_cast<const NestedBottleneckResidualBlockDesc*>(block.second.get())->blocks))
      return true;
  }
  return false;
}

inline int inputLength(const float* mask, int x, int y) {
  int length = 0;
  for(int row = 0; row < y; row++) {
    for(int col = 0; col < x; col++) {
      if(mask[row*x+col] != 0.0f)
        length = std::max(length,std::max(row,col)+1);
    }
  }
  return length;
}

// OpenCL uses size-specific handles. Metal crops homogeneous smaller
// batches and selects each row's geometry directly for mixed batches.
template<class Handle>
bool evaluateMixed(Handle& h, int n, const float* spatial, const float* global, float* output) {
  const int area = h.x*h.y;
  std::map<int,std::vector<int>> groups;
  for(int row = 0; row < n; row++)
    groups[inputLength(spatial + row*NativeTensor::NUM_SPATIAL_FEATURES*area,h.x,h.y)].push_back(row);
#ifdef USE_METAL_BACKEND
  // Metal selects each row's geometry inside one packed GPU batch.
  if(groups.size() > 1)
    return false;
#endif
  if(groups.size() == 1 && groups.begin()->first == h.x && h.x == h.y)
    return false;

  for(const auto& group : groups) {
    int length = group.first;
    auto& child = h.graphHandles[length];
    if(!child) {
      child = kg_create(h.model,length,length,h.maxBatch,h.fp16,h.device,false);
      if(!child)
        throw std::runtime_error(kg_error());
    }
    int smallArea = length*length;
    int count = static_cast<int>(group.second.size());
    std::vector<float> inputs(count*NativeTensor::NUM_SPATIAL_FEATURES*smallArea);
    std::vector<float> globals(count*NativeTensor::NUM_GLOBAL_FEATURES);
    std::vector<float> results(count*(smallArea+NativeTensor::NUM_EXTRA_OUTPUTS));
    for(int row = 0; row < count; row++) {
      int source = group.second[row];
      for(int channel = 0; channel < NativeTensor::NUM_SPATIAL_FEATURES; channel++) {
        for(int y = 0; y < length; y++)
          std::copy_n(spatial+(source*NativeTensor::NUM_SPATIAL_FEATURES+channel)*area+y*h.x,
                      length,inputs.data()+(row*NativeTensor::NUM_SPATIAL_FEATURES+channel)*smallArea+y*length);
      }
      std::copy_n(global+source*NativeTensor::NUM_GLOBAL_FEATURES,NativeTensor::NUM_GLOBAL_FEATURES,
                  globals.data()+row*NativeTensor::NUM_GLOBAL_FEATURES);
    }
    if(!kg_evaluate(child,count,inputs.data(),globals.data(),results.data()))
      throw std::runtime_error(kg_error());
    for(int row = 0; row < count; row++) {
      float* destination = output+group.second[row]*(area+NativeTensor::NUM_EXTRA_OUTPUTS);
      const float* result = results.data()+row*(smallArea+NativeTensor::NUM_EXTRA_OUTPUTS);
      std::fill_n(destination,area,-5000.0f);
      for(int y = 0; y < length; y++)
        std::copy_n(result+y*length,length,destination+y*h.x);
      std::copy_n(result+smallArea,NativeTensor::NUM_EXTRA_OUTPUTS,destination+area);
    }
  }
  return true;
}

}
