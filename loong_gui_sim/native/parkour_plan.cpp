// SPDX-License-Identifier: MIT
// Native dual-input ONNX plan for the Lite26 AMP/depth policy.
#include "parkour_plan.h"
#include "vendor/onnxruntime/onnxruntime_c_api.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void error_text(char* target, size_t capacity, const char* text) noexcept {
  if (target && capacity) std::snprintf(target, capacity, "%s", text ? text : "unknown failure");
}
struct Context {
  void* library = nullptr;
  const OrtApi* api = nullptr;
  OrtEnv* env = nullptr;
  OrtSession* session = nullptr;
  OrtMemoryInfo* memory = nullptr;
  std::string output_name;
  ~Context() {
    if (api) {
      if (memory) api->ReleaseMemoryInfo(memory);
      if (session) api->ReleaseSession(session);
      if (env) api->ReleaseEnv(env);
    }
    if (library) dlclose(library);
  }
  void check(OrtStatus* status) const {
    if (!status) return;
    std::string message = api->GetErrorMessage(status);
    api->ReleaseStatus(status);
    throw std::runtime_error(message);
  }
  std::string name(bool input, size_t index) {
    OrtAllocator* allocator = nullptr;
    check(api->GetAllocatorWithDefaultOptions(&allocator));
    char* text = nullptr;
    check(input ? api->SessionGetInputName(session, index, allocator, &text)
                : api->SessionGetOutputName(session, index, allocator, &text));
    std::string value(text);
    allocator->Free(allocator, text);
    return value;
  }
  void shape(bool input, size_t index, const std::vector<int64_t>& expected) {
    OrtTypeInfo* type = nullptr;
    check(input ? api->SessionGetInputTypeInfo(session, index, &type)
                : api->SessionGetOutputTypeInfo(session, index, &type));
    try {
      const OrtTensorTypeAndShapeInfo* tensor = nullptr;
      check(api->CastTypeInfoToTensorInfo(type, &tensor));
      if (!tensor) throw std::runtime_error("Plan inputs/outputs must be tensors");
      ONNXTensorElementDataType element;
      size_t dimensions = 0;
      check(api->GetTensorElementType(tensor, &element));
      check(api->GetDimensionsCount(tensor, &dimensions));
      std::vector<int64_t> actual(dimensions);
      check(api->GetDimensions(tensor, actual.data(), dimensions));
      if (element != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || actual.size() != expected.size())
        throw std::runtime_error("Wrong float32 Lite26 tensor contract");
      for (size_t i = 0; i < dimensions; ++i)
        if (actual[i] != expected[i] && !(i == 0 && actual[i] == -1))
          throw std::runtime_error("Wrong Lite26 tensor dimensions");
    } catch (...) { api->ReleaseTypeInfo(type); throw; }
    api->ReleaseTypeInfo(type);
  }
};
bool finite(const float* values, size_t count) {
  return values && std::all_of(values, values + count, [](float x) { return std::isfinite(x); });
}
}

extern "C" unsigned loong_parkour_abi_version() { return 1; }
extern "C" const char* loong_parkour_contract() {
  return "Lite26/1:float32 obs[1,450],depth[1,8,18,32]->actions[1,26];policy_dt=0.02";
}
extern "C" void* loong_parkour_create(const char* model, const char* runtime,
                                      char* error, size_t capacity) {
  try {
    if (!model || !*model || !runtime || !*runtime) throw std::runtime_error("Model and ONNX Runtime paths are required");
    auto context = std::make_unique<Context>();
    context->library = dlopen(runtime, RTLD_NOW | RTLD_LOCAL);
    if (!context->library) throw std::runtime_error(dlerror());
    using GetApiBase = const OrtApiBase* (*)();
    auto get_api = reinterpret_cast<GetApiBase>(dlsym(context->library, "OrtGetApiBase"));
    if (!get_api) throw std::runtime_error("ORT library has no OrtGetApiBase symbol");
    context->api = get_api()->GetApi(ORT_API_VERSION);
    if (!context->api) throw std::runtime_error("ONNX Runtime API 21 or newer is required");
    auto api = context->api;
    context->check(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "loong_parkour", &context->env));
    OrtSessionOptions* options = nullptr;
    context->check(api->CreateSessionOptions(&options));
    try {
      context->check(api->SetIntraOpNumThreads(options, 1));
      context->check(api->SetInterOpNumThreads(options, 1));
      context->check(api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL));
      context->check(api->CreateSession(context->env, model, options, &context->session));
    } catch (...) { api->ReleaseSessionOptions(options); throw; }
    api->ReleaseSessionOptions(options);
    size_t inputs = 0, outputs = 0;
    context->check(api->SessionGetInputCount(context->session, &inputs));
    context->check(api->SessionGetOutputCount(context->session, &outputs));
    if (inputs != 2 || outputs != 1) throw std::runtime_error("Expected exactly two inputs and one output for Lite26");
    bool saw_obs = false, saw_depth = false;
    for (size_t i = 0; i < inputs; ++i) {
      auto name = context->name(true, i);
      if (name == "obs" && !saw_obs) { context->shape(true, i, {1,450}); saw_obs = true; }
      else if (name == "depth" && !saw_depth) { context->shape(true, i, {1,8,18,32}); saw_depth = true; }
      else throw std::runtime_error("Expected Lite26 ONNX input names obs and depth");
    }
    context->shape(false, 0, {1,26});
    context->output_name = context->name(false, 0);
    context->check(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &context->memory));
    error_text(error, capacity, "");
    return context.release();
  } catch (const std::exception& failure) { error_text(error, capacity, failure.what()); return nullptr; }
  catch (...) { error_text(error, capacity, "Unknown native plan creation error"); return nullptr; }
}

extern "C" int loong_parkour_infer(void* opaque, const float* obs, size_t obs_count,
                                   const float* depth, size_t depth_count,
                                   float* actions, size_t action_count,
                                   char* error, size_t capacity) {
  try {
    if (!opaque || !actions || obs_count != 450 || depth_count != 8*18*32 || action_count != 26)
      throw std::runtime_error("Invalid context/pointer or tensor length");
    if (!finite(obs, obs_count) || !finite(depth, depth_count))
      throw std::runtime_error("Non-finite proprioception or depth input");
    if (!std::all_of(depth, depth + depth_count, [](float x) { return x >= 0 && x <= 1; }))
      throw std::runtime_error("Depth must be training-normalized optical Z in [0,1]");
    auto& context = *static_cast<Context*>(opaque);
    auto api = context.api;
    int64_t obs_shape[] = {1,450}, depth_shape[] = {1,8,18,32};
    OrtValue* values[2] = {nullptr,nullptr};
    OrtValue* result = nullptr;
    try {
      context.check(api->CreateTensorWithDataAsOrtValue(context.memory, const_cast<float*>(obs), obs_count*sizeof(float), obs_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &values[0]));
      context.check(api->CreateTensorWithDataAsOrtValue(context.memory, const_cast<float*>(depth), depth_count*sizeof(float), depth_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &values[1]));
      const char* input_names[] = {"obs","depth"};
      const char* output_names[] = {context.output_name.c_str()};
      context.check(api->Run(context.session, nullptr, input_names, const_cast<const OrtValue* const*>(values), 2, output_names, 1, &result));
      OrtTensorTypeAndShapeInfo* shape = nullptr;
      context.check(api->GetTensorTypeAndShape(result, &shape));
      size_t count = 0;
      auto status = api->GetTensorShapeElementCount(shape, &count);
      api->ReleaseTensorTypeAndShapeInfo(shape);
      context.check(status);
      if (count != 26) throw std::runtime_error("Runtime output must have 26 actions");
      float* output = nullptr;
      context.check(api->GetTensorMutableData(result, reinterpret_cast<void**>(&output)));
      if (!finite(output,26)) throw std::runtime_error("Native policy produced non-finite actions");
      std::copy(output, output+26, actions);
    } catch (...) {
      for (auto value : values) if (value) api->ReleaseValue(value);
      if (result) api->ReleaseValue(result);
      throw;
    }
    for (auto value : values) api->ReleaseValue(value);
    api->ReleaseValue(result);
    error_text(error,capacity,"");
    return 0;
  } catch (const std::exception& failure) { error_text(error,capacity,failure.what()); return -1; }
  catch (...) { error_text(error,capacity,"Unknown native inference error"); return -1; }
}
extern "C" void loong_parkour_destroy(void* context) { delete static_cast<Context*>(context); }
