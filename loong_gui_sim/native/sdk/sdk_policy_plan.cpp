// SPDX-License-Identifier: MIT
// SDK lifecycle adapter: observations come from driver feedback; commands flow back through driver.
#include "plan.h"
#include "quill/detail/LogManager.h"
#include <cstdlib>
#include "depth_frame.h"
#include "deployment_values.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

namespace Plan {
#if POLICY_WALKING
#define UnifiedPolicyPlan SDKWalkingPlan
#elif POLICY_DANCE
#define UnifiedPolicyPlan SDKDancePlan
#else
#define UnifiedPolicyPlan SDKParkourPlan
#endif
namespace {
constexpr int slots[26]={19,20,21,22,23,24,25,26,27,28,29,30,0,1,2,3,4,5,6,7,8,9,10,11,12,13};
double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
template<class T>T symbol(void* lib,const char* n){auto p=dlsym(lib,n);if(!p)throw std::runtime_error(n);return reinterpret_cast<T>(p);}
}
class UnifiedPolicyPlan final:public basePlanClass {
 void* lib=nullptr;void* context=nullptr;const DepthFrame* depth=nullptr;
 int tick=0,period=20;bool ready=false,history_ready=false,depth_ready=false;
 std::array<float,26> q{},dq{},target{},action{};
 std::array<float,450> observation{};
 std::array<float,37*576> depth_history{};
 std::array<float,8*576> depth_input{};
 vecNf(31) gain0, damping0;
 void* (*dance_create)(const char*,const char*,int,int)=nullptr;
 int (*dance_step)(void*,const float*,const float*,const float*,const float*,float*,float*)=nullptr;
 int (*dance_reset)(void*)=nullptr;
 int (*walking_step)(void*,const float*,const float*,const float*,const float*,const float*,float*,float*)=nullptr;
 void (*destroy)(void*)=nullptr;
 const char* (*dance_error)()=nullptr;
 int (*infer)(void*,const float*,size_t,const float*,size_t,float*,size_t,char*,size_t)=nullptr;
 void resetHistory(){history_ready=depth_ready=false;action.fill(0);depth_history.fill(0);observation.fill(0);}
 bool getDepth(float* out){
   if(!depth){int fd=open("../run/depth.bin",O_RDONLY|O_CLOEXEC|O_NOFOLLOW);if(fd<0)return false;
     auto p=mmap(nullptr,sizeof(DepthFrame),PROT_READ,MAP_SHARED,fd,0);close(fd);if(p==MAP_FAILED)return false;depth=static_cast<DepthFrame*>(p);}
   for(int attempt=0;attempt<4;++attempt){auto seq=__atomic_load_n(&depth->sequence,__ATOMIC_ACQUIRE);if(seq&1)continue;
     DepthFrame copy;std::memcpy(&copy,depth,sizeof(copy));__atomic_thread_fence(__ATOMIC_ACQUIRE);
     if(seq!=__atomic_load_n(&depth->sequence,__ATOMIC_ACQUIRE))continue;
     if(!seq || copy.magic!=depth_magic || copy.version!=1 || now()-copy.monotonic_time>.2)return false;
     for(float v:copy.depth)if(!std::isfinite(v)||v<0||v>1)return false;
     std::copy_n(copy.depth,576,out);return true;
   }return false;
 }
 void updatePolicy(const Nabo::inputStruct& in){
   for(int i=0;i<26;++i){q[i]=in.sens.j[slots[i]];dq[i]=in.sens.w[slots[i]];}
   float r=in.sens.rpy[0],p=in.sens.rpy[1],y=in.sens.rpy[2];
   Eigen::Quaternionf quat(Eigen::AngleAxisf(y,Eigen::Vector3f::UnitZ())*Eigen::AngleAxisf(p,Eigen::Vector3f::UnitY())*Eigen::AngleAxisf(r,Eigen::Vector3f::UnitX()));
#if POLICY_WALKING
   float xyzw[4]={quat.w(),quat.x(),quat.y(),quat.z()},cmd[3]={in.cmd.vx,in.cmd.vy,in.cmd.wz};
   if(walking_step(context,q.data(),dq.data(),xyzw,in.sens.gyr.data(),cmd,target.data(),action.data()))throw std::runtime_error(dance_error());
#elif POLICY_DANCE
   float xyzw[4]={quat.w(),quat.x(),quat.y(),quat.z()};
   if(dance_step(context,q.data(),dq.data(),xyzw,in.sens.gyr.data(),target.data(),action.data()))throw std::runtime_error(dance_error());
#else
   float frame[576];if(!getDepth(frame))throw std::runtime_error("depth frame missing/stale: check same-world sensor bridge");
   if(!depth_ready){for(int i=0;i<37;++i)std::copy_n(frame,576,depth_history.data()+i*576);depth_ready=true;}
   else{std::move(depth_history.begin()+576,depth_history.end(),depth_history.begin());std::copy_n(frame,576,depth_history.end()-576);}
   for(int i=0;i<8;++i)std::copy_n(depth_history.data()+(1+5*i)*576,576,depth_input.data()+576*i);
   auto rot=(Eigen::AngleAxisf(-y,Eigen::Vector3f::UnitZ())*quat).toRotationMatrix();
   float tan_norm[6]={rot(0,0),rot(1,0),rot(2,0),rot(0,2),rot(1,2),rot(2,2)};
   float cmd[3]={std::clamp(in.cmd.vx,-.8f,1.5f),std::clamp(in.cmd.vy,-.5f,.5f),std::clamp(in.cmd.wz,-1.f,1.f)};
   const float* terms[]={in.sens.gyr.data(),tan_norm,cmd,q.data(),dq.data(),action.data()};
   const int widths[]={3,6,3,26,26,26};int offset=0;
   for(int k=0;k<6;++k){int w=widths[k];auto begin=observation.begin()+offset;
     if(!history_ready)for(int t=0;t<5;++t)std::copy_n(terms[k],w,begin+t*w);
     else{std::move(begin+w,begin+5*w,begin);std::copy_n(terms[k],w,begin+4*w);}offset+=5*w;}
   history_ready=true;char error[1024]={};
   if(infer(context,observation.data(),450,depth_input.data(),4608,action.data(),26,error,sizeof(error)))throw std::runtime_error(error);
   for(int i=0;i<26;++i)target[i]=parkour_q0[i]+parkour_scale[i]*action[i];
#endif
   for(float v:target)if(!std::isfinite(v))throw std::runtime_error("non-finite policy target");
 }
public:
 UnifiedPolicyPlan(){
#if POLICY_WALKING
   lib=dlopen("../../native/libwalking_plan_x64.so",RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());
   auto create=symbol<void*(*)(const char*)>(lib,"walking_create");
   walking_step=symbol<decltype(walking_step)>(lib,"walking_step");dance_reset=symbol<decltype(dance_reset)>(lib,"walking_reset");
   destroy=symbol<decltype(destroy)>(lib,"walking_destroy");dance_error=symbol<decltype(dance_error)>(lib,"walking_error");
   context=create("../model/onnx/policy.onnx");if(!context)throw std::runtime_error(dance_error());
#elif POLICY_DANCE
   lib=dlopen("../../native/libdance_plan_x64.so",RTLD_NOW|RTLD_LOCAL);
   if(!lib)throw std::runtime_error(dlerror());
   dance_create=symbol<decltype(dance_create)>(lib,"dance_create");dance_step=symbol<decltype(dance_step)>(lib,"dance_step");
   dance_reset=symbol<decltype(dance_reset)>(lib,"dance_reset");destroy=symbol<decltype(destroy)>(lib,"dance_destroy");dance_error=symbol<decltype(dance_error)>(lib,"dance_error");
   context=dance_create("../model/dance/linglong14_mjlab_dance_29999.onnx","../model/dance/linglong14_mjlab_xiao.refbin",0,0);
   if(!context)throw std::runtime_error(dance_error());
#else
   lib=dlopen("../module/parkour/libplan_parkour_x64.so",RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());
   auto create=symbol<void*(*)(const char*,const char*,char*,size_t)>(lib,"loong_parkour_create");
   infer=symbol<decltype(infer)>(lib,"loong_parkour_infer");destroy=symbol<decltype(destroy)>(lib,"loong_parkour_destroy");
   char error[1024]={};context=create("../model/parkour/policy_10200.onnx","../third_party/lib_lin_x64/libonnxruntime.so.1",error,sizeof(error));if(!context)throw std::runtime_error(error);
#endif
 }
 ~UnifiedPolicyPlan(){if(context)destroy(context);if(lib)dlclose(lib);if(depth)munmap(const_cast<DepthFrame*>(depth),sizeof(DepthFrame));}
 void init(float value) override {basePlanClass::init(value);logger.log(getPlanVersion());
   // SDK plan loggers share a backend; drain/stop it before plan teardown releases inference resources.
   std::atexit([](){quill::detail::LogManagerSingleton::instance().log_manager().stop_backend_worker();});period=std::lround(.02/value);if(period<1||std::abs(period*value-.02)>1e-6)throw std::runtime_error("policy requires an integral 20 ms control period");}
 void hey(const Nabo::inputStruct& in,Nabo::outputStruct& out) override {
   gain0=out.param.kp;damping0=out.param.kd;baseReset(in,out);resetHistory();tick=0;ready=true;
   if(dance_reset)dance_reset(context);
   for(int i=0;i<26;++i)target[i]=in.sens.j[slots[i]];
   fprintf(stderr,"[SDK %s] entered, same control chain, 26 joints / 20 ms policy\n",POLICY_WALKING?"walking":(POLICY_DANCE?"dance":"parkour"));
 }
 bool run(const Nabo::inputStruct& in,Nabo::outputStruct& out) override {
   baseRun(in,out);
   try{if(tick%period==0)updatePolicy(in);}catch(const std::exception& e){
     fprintf(stderr,"[SDK policy] stopped: %s\n",e.what());ready=false;out.ctrl.enFlag=false;return true;}
   float blend=std::clamp(tick*dt/.2f,0.f,1.f);
   outCtrl.j=j0;outCtrl.w.setZero();outCtrl.t.setZero();
   for(int i=0;i<26;++i){int slot=slots[i];
     float kp=POLICY_DANCE?dance_kp[i]:parkour_kp[i],kd=POLICY_DANCE?dance_kd[i]:parkour_kd[i];
#if POLICY_WALKING
     kp=20;kd=.1f;for(int k=0;k<20;++k)if(walking_ids[k]==i){kp=walking_kp[k];kd=walking_kd[k];}
#endif
     outCtrl.j[slot]=j0[slot]+blend*(target[i]-j0[slot]);
     out.param.kp[slot]=gain0[slot]+blend*(kp-gain0[slot]);out.param.kd[slot]=damping0[slot]+blend*(kd-damping0[slot]);
     out.param.maxTorFwd[slot]=POLICY_DANCE?dance_effort[i]:parkour_effort[i];
   }
   out.ctrl=outCtrl;++tick;return !ready;
 }
 void bye(const Nabo::inputStruct&,bool) override {ready=false;}
 void log() override {
   if(logFlag && logCnt>0 && inSens.cnt%logCnt==0){
     logger.appendTab(planName).appendTab(tim);
     logger.appendTab(inCmd.vx).appendTab(inCmd.vy).appendTab(inCmd.wz);
     for(int slot:slots)logger.appendTab(outCtrl.j[slot]);
     logger.log();
   }
 }
 string_view getPlanVersion() override {return POLICY_WALKING?"sdk-walking-700-v1":(POLICY_DANCE?"sdk-dance-1610-v1":"sdk-parkour-450-depth-v1");}
};
extern "C" basePlanClass* creat(){return new UnifiedPolicyPlan();}
}
