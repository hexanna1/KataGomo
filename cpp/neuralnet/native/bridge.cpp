#include "bridge.h"
#include "neuralnet/nninterface.h"
#include "neuralnet/nneval.h"

#include <memory>
#include <mutex>
#include "benty-graph-bridge.h"

// The bridge uses input buffers without constructing an upstream search engine.
NNResultBuf::NNResultBuf()
  : hasResult(false),
    includeOwnerMap(false),
    boardXSizeForServer(0),
    boardYSizeForServer(0),
    hasRowMeta(false),
    errorLogLockout(false),
    symmetry(0),
    policyOptimism(0)
{}

NNResultBuf::~NNResultBuf() = default;

namespace {
thread_local std::string error;

struct Handle {
  ComputeContext* context = nullptr;
  ComputeHandle* compute = nullptr;
  InputBuffers* buffers = nullptr;
  int x;
  int y;
  float outputScale;
  bool canPackY;
  bool graphAttention;
  void* model;
  int maxBatch;
  bool fp16;
  int device;
  std::map<int,void*> graphHandles;
  std::vector<std::unique_ptr<NNResultBuf>> rows;
  std::vector<std::unique_ptr<NNOutput>> results;
  std::vector<NNResultBuf*> inputs;
  std::vector<NNOutput*> outputs;

  ~Handle() {
    for(const auto& child : graphHandles)
      kg_free(child.second);
    if(buffers)
      NeuralNet::freeInputBuffers(buffers);
    if(compute)
      NeuralNet::freeComputeHandle(compute);
    if(context)
      NeuralNet::freeComputeContext(context);
  }
};
}

const char* kg_error() {
  return error.c_str();
}

void* kg_load(const char* path) {
  try {
    static std::once_flag initialized;
    std::call_once(initialized, NeuralNet::globalInitialize);
    return NeuralNet::loadModelFile(path, "");
  }
  catch(const std::exception& e) {
    error = e.what();
    return nullptr;
  }
}

void kg_free_model(void* m) {
  NeuralNet::freeLoadedModel(static_cast<LoadedModel*>(m));
}

const char* kg_name(void* m) {
  return NeuralNet::getModelDesc(static_cast<LoadedModel*>(m)).name.c_str();
}

void* kg_create(void* model, int x, int y, int batch, bool fp16, int device, bool packY) {
  try {
    auto h = std::make_unique<Handle>();
    h->x = x;
    h->y = y;
    auto* m = static_cast<LoadedModel*>(model);
    h->model = model;
    h->maxBatch = batch;
    h->fp16 = fp16;
    h->device = device;
    h->graphAttention = BentYGraph::hasAttention(NeuralNet::getModelDesc(m).trunk.blocks);
    ConfigParser cfg;
    cfg.overrideKey("packY",packY ? "true" : "false");
    h->context = NeuralNet::createComputeContext(
      {device},nullptr,x,y,"",fp16 ? enabled_t::Auto : enabled_t::False,m,cfg
    );
    h->compute = NeuralNet::createComputeHandle(h->context,m,nullptr,batch,false,false,device,0);
    h->canPackY = !h->graphAttention && x == y && NeuralNet::getModelDesc(m).hasAnyTransformerBlocks();
#ifdef USE_OPENCL_BACKEND
    h->canPackY = h->canPackY && !NeuralNet::isUsingFP16(h->compute);
#endif
    h->outputScale = NeuralNet::getModelDesc(m).postProcessParams.outputScaleMultiplier;
    h->buffers = NeuralNet::createInputBuffers(m,batch,x,y);
    h->outputs.reserve(batch);
    for(int i = 0; i < batch; i++) {
      auto r = std::make_unique<NNResultBuf>();
      r->boardXSizeForServer = x;
      r->boardYSizeForServer = y;
      r->rowSpatialBuf.resize(NativeTensor::NUM_SPATIAL_FEATURES * x * y);
      r->rowGlobalBuf.resize(NativeTensor::NUM_GLOBAL_FEATURES);
      h->inputs.push_back(r.get());
      h->rows.push_back(std::move(r));
      auto o = std::make_unique<NNOutput>();
      o->nnXLen = x;
      o->nnYLen = y;
      h->results.push_back(std::move(o));
    }
    return h.release();
  }
  catch(const std::exception& e) {
    error = e.what();
    return nullptr;
  }
}

void kg_free(void* handle) {
  delete static_cast<Handle*>(handle);
}

bool kg_fp16(void* handle) {
  return NeuralNet::isUsingFP16(static_cast<Handle*>(handle)->compute);
}

bool kg_can_pack_y(void* handle) {
  return static_cast<Handle*>(handle)->canPackY;
}

bool kg_evaluate(void* handle, int n, const float* spatial, const float* global, float* output) {
  try {
    auto& h = *static_cast<Handle*>(handle);
    if(h.graphAttention && BentYGraph::evaluateMixed(h,n,spatial,global,output))
      return true;
    int area = h.x*h.y;
    int batch = n;
#ifdef USE_METAL_BACKEND
    // Reuse MPSGraph shapes with at most seven padding rows above batch eight.
    batch = n <= 8 ? 1 : ((n+7)/8)*8;
    while(batch < n)
      batch *= 2;
    batch = std::min(batch,static_cast<int>(h.rows.size()));
#endif
    h.outputs.resize(batch);
    for(int i = 0; i < batch; i++) {
      int source = std::min(i,n-1);
      std::copy_n(spatial + source * NativeTensor::NUM_SPATIAL_FEATURES * area,
        NativeTensor::NUM_SPATIAL_FEATURES * area,h.rows[i]->rowSpatialBuf.data());
      std::copy_n(global + source * NativeTensor::NUM_GLOBAL_FEATURES,
        NativeTensor::NUM_GLOBAL_FEATURES,h.rows[i]->rowGlobalBuf.data());
      h.outputs[i] = h.results[i].get();
    }
    NeuralNet::getOutput(h.compute,h.buffers,batch,h.inputs.data(),h.outputs);
    for(int i = 0; i < n; i++) {
      auto& o = *h.outputs[i];
      float* dst = output + i * (area + NativeTensor::NUM_EXTRA_OUTPUTS);
      std::copy_n(o.policyProbs,area+1,dst);
      dst[area + NativeTensor::WIN] = o.whiteWinProb;
      dst[area + NativeTensor::LOSS] = o.whiteLossProb;
      dst[area + NativeTensor::NO_RESULT] = o.whiteNoResultProb;
      dst[area + NativeTensor::VAR_TIME_LEFT] = o.varTimeLeft;
      dst[area + NativeTensor::SHORTTERM_WINLOSS_ERROR] = o.shorttermWinlossError;
      // Return logits in the model's output scale for the client's postprocessing.
      for(int j = 0; j < area + NativeTensor::NUM_EXTRA_OUTPUTS; j++)
        dst[j] *= h.outputScale;
    }
    return true;
  }
  catch(const std::exception& e) {
    error = e.what();
    return false;
  }
}
