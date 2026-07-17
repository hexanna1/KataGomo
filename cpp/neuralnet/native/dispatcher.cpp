#include "bridge.h"
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {
thread_local std::string error;

std::filesystem::path libraryDirectory() {
#ifdef _WIN32
  HMODULE module;
  GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    reinterpret_cast<LPCSTR>(&kg_backend), &module);
  char path[32768];
  GetModuleFileNameA(module, path, sizeof(path));
  return std::filesystem::path(path).parent_path();
#else
  Dl_info info;
  dladdr(reinterpret_cast<void*>(&kg_backend), &info);
  return std::filesystem::path(info.dli_fname).parent_path();
#endif
}

struct Backend {
  const char* name;
  void* library;
  decltype(&kg_error) lastError;
  decltype(&kg_load) load;
  decltype(&kg_free_model) freeModel;
  decltype(&kg_name) modelName;
  decltype(&kg_create) create;
  decltype(&kg_free) freeHandle;
  decltype(&kg_fp16) fp16;
  decltype(&kg_can_pack_y) canPackY;
  decltype(&kg_evaluate) evaluate;

  template<class T> T symbol(const char* name) {
#ifdef _WIN32
    return reinterpret_cast<T>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
    return reinterpret_cast<T>(dlsym(library, name));
#endif
  }

  Backend(const char* backendName, const char* filename) : name(backendName) {
    auto path = libraryDirectory() / filename;
#ifdef _WIN32
    library = LoadLibraryW(path.c_str());
    if(!library) throw std::runtime_error("Cannot load inference backend: " + path.string());
#else
    library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if(!library) throw std::runtime_error(dlerror());
#endif
    lastError = symbol<decltype(lastError)>("kg_error");
    load = symbol<decltype(load)>("kg_load");
    freeModel = symbol<decltype(freeModel)>("kg_free_model");
    modelName = symbol<decltype(modelName)>("kg_name");
    create = symbol<decltype(create)>("kg_create");
    freeHandle = symbol<decltype(freeHandle)>("kg_free");
    fp16 = symbol<decltype(fp16)>("kg_fp16");
    canPackY = symbol<decltype(canPackY)>("kg_can_pack_y");
    evaluate = symbol<decltype(evaluate)>("kg_evaluate");
  }
};

Backend& selectBackend() {
#ifdef __APPLE__
  const char* requested = std::getenv("KATAGOMO_INFERENCE_BACKEND");
  if(!requested || std::string(requested) != "opencl") {
    static Backend metal("Metal", KG_METAL_LIBRARY_NAME);
    return metal;
  }
#endif
  static Backend opencl("OpenCL", KG_OPENCL_LIBRARY_NAME);
  return opencl;
}

struct Model {
  Backend* backend = nullptr;
  void* native = nullptr;
  ~Model() { if(native) backend->freeModel(native); }
};
struct Handle {
  Backend* backend = nullptr;
  void* native = nullptr;
  ~Handle() { if(native) backend->freeHandle(native); }
};
}

const char* kg_backend() {
#ifdef __APPLE__
  return "OpenCL/Metal";
#else
  return "OpenCL";
#endif
}
const char* kg_error() { return error.c_str(); }
const char* kg_model_backend(void* model) { return static_cast<Model*>(model)->backend->name; }
void* kg_load(const char* path) {
  try {
    auto model = std::make_unique<Model>();
    model->backend = &selectBackend();
    model->native = model->backend->load(path);
    if(!model->native) { error = model->backend->lastError(); return nullptr; }
    return model.release();
  } catch(const std::exception& e) { error = e.what(); return nullptr; }
}
void kg_free_model(void* model) { delete static_cast<Model*>(model); }
const char* kg_name(void* model) {
  auto& m = *static_cast<Model*>(model);
  return m.backend->modelName(m.native);
}
void* kg_create(void* model, int x, int y, int batch, bool fp16, int device, bool packY) {
  try {
    auto& m = *static_cast<Model*>(model);
    auto handle = std::make_unique<Handle>();
    handle->backend = m.backend;
    handle->native = m.backend->create(m.native, x, y, batch, fp16, device, packY);
    if(!handle->native) { error = m.backend->lastError(); return nullptr; }
    return handle.release();
  } catch(const std::exception& e) { error = e.what(); return nullptr; }
}
void kg_free(void* handle) { delete static_cast<Handle*>(handle); }
bool kg_fp16(void* handle) {
  auto& h = *static_cast<Handle*>(handle);
  return h.backend->fp16(h.native);
}
bool kg_can_pack_y(void* handle) {
  auto& h = *static_cast<Handle*>(handle);
  return h.backend->canPackY(h.native);
}
bool kg_evaluate(void* handle, int n, const float* spatial, const float* global, float* output) {
  auto& h = *static_cast<Handle*>(handle);
  bool success = h.backend->evaluate(h.native, n, spatial, global, output);
  if(!success) error = h.backend->lastError();
  return success;
}
