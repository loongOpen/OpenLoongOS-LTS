// SPDX-License-Identifier: MIT
#include <onnxruntime_c_api.h>
#include "deployment_values.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
namespace {
const OrtApi* api=OrtGetApiBase()->GetApi(21);
thread_local std::string failure;
void check(OrtStatus*s){if(s){std::string t=api->GetErrorMessage(s);api->ReleaseStatus(s);throw std::runtime_error(t);}}
struct Policy {
 OrtEnv* env=nullptr;OrtSession* session=nullptr;OrtMemoryInfo* memory=nullptr;
 std::array<float,700> obs{};std::array<float,20> previous{};float phase=0;bool filled=false;
 ~Policy(){if(memory)api->ReleaseMemoryInfo(memory);if(session)api->ReleaseSession(session);if(env)api->ReleaseEnv(env);}
 void init(const char*path){
   check(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,"SDKWalking",&env));OrtSessionOptions*o=nullptr;check(api->CreateSessionOptions(&o));
   try{check(api->SetIntraOpNumThreads(o,1));check(api->SetInterOpNumThreads(o,1));check(api->CreateSession(env,path,o,&session));}catch(...){api->ReleaseSessionOptions(o);throw;}
   api->ReleaseSessionOptions(o);check(api->CreateCpuMemoryInfo(OrtArenaAllocator,OrtMemTypeDefault,&memory));
 }
 void step(const float*q,const float*dq,const float*quat,const float*gyr,const float*cmd,float*target,float*actions){
   float norm=0;for(int i=0;i<4;++i)norm+=quat[i]*quat[i];if(!std::isfinite(norm)||norm<1e-9)throw std::runtime_error("invalid quaternion");norm=std::sqrt(norm);
   float w=quat[0]/norm,x=quat[1]/norm,y=quat[2]/norm,z=quat[3]/norm;
   float omega[3]={gyr[0]*.25f,gyr[1]*.25f,gyr[2]*.25f},gravity[3]={2*(w*y-x*z),-2*(y*z+w*x),-(1-2*(x*x+y*y))};
   float pos[20],vel[20];for(int i=0;i<20;++i){pos[i]=q[walking_ids[i]]-walking_q0[i];vel[i]=dq[walking_ids[i]]*.05f;}
   if(cmd[0]*cmd[0]+cmd[1]*cmd[1]+cmd[2]*cmd[2]>.01f)phase=std::fmod(phase+.02f,1.f);
   float clock=std::sin(2*3.14159265358979323846f*phase);
   const float* terms[]={omega,gravity,cmd,pos,vel,previous.data(),&clock};const int widths[]={3,3,3,20,20,20,1};int offset=0;
   for(int k=0;k<7;++k){int n=widths[k];auto begin=obs.begin()+offset;if(!filled)for(int t=0;t<10;++t)std::copy_n(terms[k],n,begin+t*n);
     else{std::move(begin+n,begin+10*n,begin);std::copy_n(terms[k],n,begin+9*n);}offset+=10*n;}
   filled=true;for(float v:obs)if(!std::isfinite(v))throw std::runtime_error("non-finite walking observation");
   OrtValue* input=nullptr;OrtValue*output=nullptr;int64_t shape[2]={1,700};
   check(api->CreateTensorWithDataAsOrtValue(memory,obs.data(),sizeof(obs),shape,2,ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,&input));
   const char*in[]={"obs"};const char*out[]={"actions"};
   try{check(api->Run(session,nullptr,in,&input,1,out,1,&output));float*data=nullptr;check(api->GetTensorMutableData(output,reinterpret_cast<void**>(&data)));
     for(int i=0;i<20;++i)if(!std::isfinite(data[i]))throw std::runtime_error("non-finite walking action");
     std::fill_n(target,26,0);for(int i=0;i<20;++i){previous[i]=actions[i]=data[i];target[walking_ids[i]]=walking_q0[i]+walking_scale[i]*data[i];}
   }catch(...){if(output)api->ReleaseValue(output);api->ReleaseValue(input);throw;}
   api->ReleaseValue(output);api->ReleaseValue(input);
 }
};
}
extern "C" {
const char* walking_error(){return failure.c_str();}
void* walking_create(const char*path){try{auto p=std::make_unique<Policy>();p->init(path);return p.release();}catch(const std::exception&e){failure=e.what();return nullptr;}}
void walking_destroy(void*p){delete static_cast<Policy*>(p);}
int walking_reset(void*p){auto&v=*static_cast<Policy*>(p);v.phase=0;v.filled=false;v.previous.fill(0);return 0;}
int walking_step(void*p,const float*q,const float*dq,const float*quat,const float*gyr,const float*cmd,float*target,float*actions){try{static_cast<Policy*>(p)->step(q,dq,quat,gyr,cmd,target,actions);return 0;}catch(const std::exception&e){failure=e.what();return -1;}}
}
