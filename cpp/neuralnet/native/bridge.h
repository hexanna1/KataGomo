#pragma once

// Tensor-only interface; board rules and symmetries stay in the engine.
namespace NativeTensor {
constexpr int NUM_SPATIAL_FEATURES = 22;
constexpr int NUM_GLOBAL_FEATURES = 19;

// Each output row has one policy logit per cell, followed by these values.
enum ExtraOutputOffset {
  POLICY_PASS = 0,
  WIN = 1,
  LOSS = 2,
  NO_RESULT = 3,
  VAR_TIME_LEFT = 4,
  SHORTTERM_WINLOSS_ERROR = 5,
  NUM_EXTRA_OUTPUTS = 6,
};
}

#if defined(_WIN32)
#define KG_API __declspec(dllexport)
#else
#define KG_API __attribute__((visibility("default")))
#endif

extern "C" {
KG_API const char* kg_error();
KG_API const char* kg_backend();
KG_API const char* kg_model_backend(void* model);
KG_API void* kg_load(const char* path);
KG_API void kg_free_model(void* model);
KG_API const char* kg_name(void* model);
KG_API void* kg_create(void* model, int x, int y, int batch, bool fp16, int device, bool packY);
KG_API void kg_free(void* handle);
KG_API bool kg_fp16(void* handle);
KG_API bool kg_can_pack_y(void* handle);
KG_API bool kg_evaluate(void* handle, int n, const float* spatial, const float* global, float* output);
}
