// Copyright 2026 OpenLoongOS contributors. SPDX-License-Identifier: MIT
// Portable GUI deployment of the training_dance Linglong14 1610/26 contract.
// This C API is independent of the original aarch64 Nabo plugin ABI.
#include "vendor/onnxruntime/onnxruntime_c_api.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t kObs = 1610, kAct = 26, kRef = 35, kProp = 84;
const OrtApi* api = OrtGetApiBase()->GetApi(21);
thread_local std::string error;
void check(OrtStatus* status) {
  if (status) { std::string msg = api->GetErrorMessage(status); api->ReleaseStatus(status); throw std::runtime_error(msg); }
}
constexpr float scale[kAct] = {.5f,.5f,.5f,.5f,.5f,.5f,.5f,.5f,.5f,.5f,.5f,.5f,
  .20208333333333334f,.20208333333333334f,.16875f,.16875f,.21875f,.21875f,.21875f,
  .20208333333333334f,.20208333333333334f,.16875f,.16875f,.21875f,.21875f,.21875f};
struct Plan {
  OrtEnv* env = nullptr; OrtSession* session = nullptr; OrtMemoryInfo* memory = nullptr;
  std::vector<float> reference; std::array<float,kProp*10> history{};
  std::array<float,kAct> previous{}; std::array<float,kObs> observation{};
  int frames = 0, frame = 0, start = 0; bool loop = false, initialized = false;
  ~Plan() { if(memory) api->ReleaseMemoryInfo(memory); if(session) api->ReleaseSession(session); if(env) api->ReleaseEnv(env); }
  void init(const char* model, const char* motion, int first, bool repeat) {
    std::ifstream file(motion,std::ios::binary); if(!file) throw std::runtime_error("cannot open reference");
    char magic[8]; uint32_t n=0,a=0,r=0; float period=0;
    file.read(magic,8); file.read(reinterpret_cast<char*>(&n),4); file.read(reinterpret_cast<char*>(&a),4);
    file.read(reinterpret_cast<char*>(&r),4); file.read(reinterpret_cast<char*>(&period),4);
    if(!file || std::memcmp(magic,"BMJLAB01",8) || n==0 || n>1000000 || a!=kAct || r!=kRef || std::abs(period-.02f)>1e-6f || first<0 || uint32_t(first)>=n)
      throw std::runtime_error("invalid BMJLAB01 reference/header/control period/start frame");
    reference.resize(size_t(n)*kRef); file.read(reinterpret_cast<char*>(reference.data()),reference.size()*4);
    if(!file || file.peek()!=EOF) throw std::runtime_error("invalid reference file length");
    for(float x:reference) if(!std::isfinite(x)) throw std::runtime_error("non-finite reference");
    frames=int(n); frame=start=first; loop=repeat;
    check(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,"dance_plan",&env));
    OrtSessionOptions* options=nullptr; check(api->CreateSessionOptions(&options));
    try { check(api->SetIntraOpNumThreads(options,1)); check(api->SetInterOpNumThreads(options,1));
      check(api->CreateSession(env,model,options,&session)); } catch(...) { api->ReleaseSessionOptions(options); throw; }
    api->ReleaseSessionOptions(options);
    size_t inputs=0,outputs=0; check(api->SessionGetInputCount(session,&inputs)); check(api->SessionGetOutputCount(session,&outputs));
    if(inputs!=1 || outputs!=1) throw std::runtime_error("dance requires one obs input and one actions output");
    OrtAllocator* allocator=nullptr; check(api->GetAllocatorWithDefaultOptions(&allocator));
    for(int out=0;out<2;++out) {
      char* name=nullptr; if(out) check(api->SessionGetOutputName(session,0,allocator,&name)); else check(api->SessionGetInputName(session,0,allocator,&name));
      bool correct=name && std::string(name)==(out?"actions":"obs"); allocator->Free(allocator,name);
      if(!correct) throw std::runtime_error("unexpected dance ONNX tensor names");
      OrtTypeInfo* type=nullptr; if(out) check(api->SessionGetOutputTypeInfo(session,0,&type)); else check(api->SessionGetInputTypeInfo(session,0,&type));
      const OrtTensorTypeAndShapeInfo* shape=nullptr; check(api->CastTypeInfoToTensorInfo(type,&shape));
      ONNXTensorElementDataType dtype; check(api->GetTensorElementType(shape,&dtype)); size_t count=0; check(api->GetDimensionsCount(shape,&count));
      int64_t dims[2]={}; if(count==2) check(api->GetDimensions(shape,dims,2)); api->ReleaseTypeInfo(type);
      if(dtype!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || count!=2 || (dims[0]!=1 && dims[0]!=-1) || dims[1]!=(out?26:1610))
        throw std::runtime_error("unexpected dance ONNX shape, expected float32 [1,1610] -> [1,26]");
    }
    check(api->CreateCpuMemoryInfo(OrtArenaAllocator,OrtMemTypeDefault,&memory));
  }
  void step(const float* q,const float* dq,const float* quat,const float* gyro,float* target,float* action) {
    for(size_t i=0;i<26;++i) if(!std::isfinite(q[i]) || !std::isfinite(dq[i])) throw std::runtime_error("non-finite joint state");
    double norm=0; for(int i=0;i<4;++i) { if(!std::isfinite(quat[i])) throw std::runtime_error("non-finite quaternion"); norm+=double(quat[i])*quat[i]; }
    if(norm<1e-12) throw std::runtime_error("zero quaternion");
    float w=quat[0]/std::sqrt(norm),x=quat[1]/std::sqrt(norm),y=quat[2]/std::sqrt(norm),z=quat[3]/std::sqrt(norm);
    std::array<float,kProp> prop{};
    prop[0]=2*(w*y-x*z); prop[1]=-2*(y*z+w*x); prop[2]=-(1-2*(x*x+y*y));
    for(int i=0;i<3;++i) { if(!std::isfinite(gyro[i])) throw std::runtime_error("non-finite gyroscope"); prop[3+i]=gyro[i]; }
    for(size_t i=0;i<kAct;++i) { prop[6+i]=q[i]; prop[32+i]=dq[i]; prop[58+i]=previous[i]; }
    if(!initialized) { for(int i=0;i<10;++i) std::copy(prop.begin(),prop.end(),history.begin()+i*kProp); initialized=true; }
    else { std::move(history.begin()+kProp,history.end(),history.begin()); std::copy(prop.begin(),prop.end(),history.end()-kProp); }
    size_t offset=0;
    for(int i=-10;i<=10;++i) { int index=std::clamp(frame+i,0,frames-1); std::copy_n(reference.data()+index*kRef,kRef,observation.data()+offset); offset+=kRef; }
    std::copy_n(reference.data()+frame*kRef,kRef,observation.data()+offset); offset+=kRef;
    std::copy(history.begin(),history.end(),observation.begin()+offset);
    const int64_t shape[2]={1,1610}; OrtValue* input=nullptr; OrtValue* output=nullptr;
    check(api->CreateTensorWithDataAsOrtValue(memory,observation.data(),kObs*4,shape,2,ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,&input));
    const char* inNames[]={"obs"}; const char* outNames[]={"actions"};
    try { check(api->Run(session,nullptr,inNames,&input,1,outNames,1,&output));
      float* result=nullptr; check(api->GetTensorMutableData(output,reinterpret_cast<void**>(&result)));
      for(size_t i=0;i<kAct;++i) if(!std::isfinite(result[i])) throw std::runtime_error("non-finite policy action");
      for(size_t i=0;i<kAct;++i) { action[i]=previous[i]=result[i]; target[i]=reference[frame*kRef+9+i]+scale[i]*result[i]; }
    } catch(...) { if(output) api->ReleaseValue(output); api->ReleaseValue(input); throw; }
    api->ReleaseValue(output); api->ReleaseValue(input);
    if(frame<frames-1) ++frame; else if(loop) frame=start;
  }
};
}
extern "C" {
const char* dance_error() { return error.c_str(); }
const char* dance_contract() { return "linglong14_mjlab_1610_26_v1;C_API=1;MuJoCo_order;policyDt=0.02"; }
void* dance_create(const char* model,const char* motion,int start,int loop) {
  try { error.clear(); auto plan=std::make_unique<Plan>(); plan->init(model,motion,start,loop!=0); return plan.release(); }
  catch(const std::exception& e) { error=e.what(); return nullptr; }
}
void dance_destroy(void* handle) { delete static_cast<Plan*>(handle); }
int dance_reset(void* handle) { if(!handle) return -1; auto& p=*static_cast<Plan*>(handle); p.frame=p.start; p.initialized=false; p.previous.fill(0); p.history.fill(0); return 0; }
int dance_frame(void* handle) { return handle?static_cast<Plan*>(handle)->frame:-1; }
int dance_step(void* handle,const float* q,const float* dq,const float* quat,const float* gyro,float* target,float* action) {
  try { if(!handle || !q || !dq || !quat || !gyro || !target || !action) throw std::runtime_error("null dance argument"); static_cast<Plan*>(handle)->step(q,dq,quat,gyro,target,action); return 0; }
  catch(const std::exception& e) { error=e.what(); return -1; }
}
}
