#include "../neuralnet/nninterface.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "native/bridge.h"

using namespace std;

struct LoadedModel {
  void* native;
};

struct ComputeContext {
  int x;
  int y;
  bool fp16;
};

struct ComputeHandle {
  ComputeContext* context;
  void* native;
  bool nhwc;
  void* model;
  int maxBatch;
  int device;
  Logger* logger;
  void* packed = nullptr;
};

struct InputBuffers {
  vector<float> spatial;
  vector<float> global;
  vector<float> output;
  vector<float> scratch;
};

void NeuralNet::globalInitialize() {}
void NeuralNet::globalCleanup() {}

void NeuralNet::printDevices() {
  cout << "Native " << kg_backend() << " inference" << endl;
}

LoadedModel* NeuralNet::loadModelFile(const string& file, const string&) {
  void* model = kg_load(file.c_str());
  if(!model)
    throw StringError(kg_error());
  return new LoadedModel{model};
}

void NeuralNet::freeLoadedModel(LoadedModel* m) {
  kg_free_model(m->native);
  delete m;
}

string NeuralNet::getModelName(const LoadedModel* m) {
  return kg_name(m->native);
}

int NeuralNet::getModelVersion(const LoadedModel*) {
  // The engine's input/postprocessing contract is independent of the native trunk descriptor.
  return NNModelVersion::defaultModelVersion;
}

Rules NeuralNet::getSupportedRules(const LoadedModel*, const Rules& desired, bool& supported) {
  supported = true;
  return desired;
}

ComputeContext* NeuralNet::createComputeContext(
  const vector<int>&,
  Logger*,
  int x,
  int y,
  const string&,
  const string&,
  bool,
  enabled_t fp16,
  enabled_t,
  const LoadedModel*
) {
  return new ComputeContext{x,y,fp16 != enabled_t::False};
}

void NeuralNet::freeComputeContext(ComputeContext* c) {
  delete c;
}

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* c,
  const LoadedModel* m,
  Logger* logger,
  int batch,
  bool,
  bool nhwc,
  int device,
  int
) {
  void* h = kg_create(m->native,c->x,c->y,batch,c->fp16,device,false);
  if(!h)
    throw StringError(kg_error());
  if(logger)
    logger->write("Native " + string(kg_model_backend(m->native)) + ": " + kg_name(m->native));
  return new ComputeHandle{c,h,nhwc,m->native,batch,device,logger};
}

void NeuralNet::freeComputeHandle(ComputeHandle* h) {
  if(h) {
    kg_free(h->native);
    if(h->packed)
      kg_free(h->packed);
  }
  delete h;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* h) {
  return kg_fp16(h->native);
}

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel*, int batch, int x, int y) {
  return new InputBuffers{
    vector<float>(batch * NativeTensor::NUM_SPATIAL_FEATURES * x * y),
    vector<float>(batch * NativeTensor::NUM_GLOBAL_FEATURES),
    vector<float>(batch * (x*y + NativeTensor::NUM_EXTRA_OUTPUTS)),
    vector<float>(NativeTensor::NUM_SPATIAL_FEATURES * x * y)
  };
}

void NeuralNet::freeInputBuffers(InputBuffers* b) {
  delete b;
}

void NeuralNet::getOutput(
  ComputeHandle* h,
  InputBuffers* b,
  int n,
  NNResultBuf** inputs,
  vector<NNOutput*>& outputs
) {
  int x = h->context->x;
  int y = h->context->y;
  int area = x*y;
  for(int i = 0; i < n; i++) {
    float* spatial = b->spatial.data() + i * NativeTensor::NUM_SPATIAL_FEATURES * area;
    SymmetryHelpers::copyInputsWithSymmetry(
      inputs[i]->rowSpatial,h->nhwc ? b->scratch.data() : spatial,
      1,y,x,NativeTensor::NUM_SPATIAL_FEATURES,h->nhwc,inputs[i]->symmetry,
      inputs[i]->boardShapeForServer,inputs[i]->boardXSizeForServer,inputs[i]->boardYSizeForServer
    );
    if(h->nhwc) {
      for(int c = 0; c < NativeTensor::NUM_SPATIAL_FEATURES; c++) {
        for(int j = 0; j < area; j++)
          spatial[c*area+j] = b->scratch[j*NativeTensor::NUM_SPATIAL_FEATURES+c];
      }
    }
    copy_n(inputs[i]->rowGlobal,NativeTensor::NUM_GLOBAL_FEATURES,
      b->global.data() + i * NativeTensor::NUM_GLOBAL_FEATURES);
  }
  void* inference = h->native;
  bool packY = n >= 16 && kg_can_pack_y(h->native);
  for(int i = 0; i < n; i++)
    packY = packY && inputs[i]->boardShapeForServer == BoardShape::Y;
  if(packY) {
    if(!h->packed) {
      h->packed = kg_create(h->model,x,y,h->maxBatch,h->context->fp16,h->device,true);
      if(!h->packed)
        throw StringError(kg_error());
      if(h->logger) {
        h->logger->write("Native " + string(kg_model_backend(h->model)) +
          ": Y transformer blocks use " + Global::intToString(x*(x+1)/2) +
          " triangle positions on a " + Global::intToString(x) + "x" + Global::intToString(y) +
          " grid for batches >= 16");
      }
    }
    inference = h->packed;
  }
  if(!kg_evaluate(inference,n,b->spatial.data(),b->global.data(),b->output.data()))
    throw StringError(kg_error());
  for(int i = 0; i < n; i++) {
    const float* row = b->output.data() + i * (area + NativeTensor::NUM_EXTRA_OUTPUTS);
    SymmetryHelpers::copyOutputsWithSymmetry(
      row,outputs[i]->policyProbs,1,y,x,inputs[i]->symmetry,
      inputs[i]->boardShapeForServer,inputs[i]->boardXSizeForServer,inputs[i]->boardYSizeForServer
    );
    outputs[i]->policyProbs[area] = row[area + NativeTensor::POLICY_PASS];
    outputs[i]->whiteWinProb = row[area + NativeTensor::WIN];
    outputs[i]->whiteLossProb = row[area + NativeTensor::LOSS];
    outputs[i]->whiteNoResultProb = row[area + NativeTensor::NO_RESULT];
    outputs[i]->varTimeLeft = row[area + NativeTensor::VAR_TIME_LEFT];
    outputs[i]->shorttermWinlossError = row[area + NativeTensor::SHORTTERM_WINLOSS_ERROR];
  }
}

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc*, int, int, int, bool, bool, const vector<float>&, vector<float>&
) {
  return false;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc*, int, int, int, bool, bool, const vector<float>&, const vector<float>&, vector<float>&
) {
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc*, int, int, int, bool, bool, const vector<float>&, const vector<float>&, vector<float>&
) {
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc*, int, int, int, bool, bool, const vector<float>&, const vector<float>&, vector<float>&
) {
  return false;
}
