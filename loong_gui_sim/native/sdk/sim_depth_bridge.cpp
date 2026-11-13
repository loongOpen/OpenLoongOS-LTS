// SPDX-License-Identifier: MIT
// A sensor extension of the existing SDK simulation, with no separate physics world.
#include <mujoco/mujoco.h>
#include "depth_frame.h"
#include "deployment_values.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
DepthFrame* shared=nullptr;
const mjModel* last_model=nullptr;
double deadline=0;
std::array<double,2304*3> rays{},world_rays{};
std::array<double,2304> distances{};
std::array<int,2304> geom_ids{};
int base=-1;
bool initialized=false;
double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
void initialize(const mjModel* m) {
  if (mj_version()!=311) { fprintf(stderr,"[DEPTH] requires MuJoCo 3.1.1\n"); return; }
  base=mj_name2id(m,mjOBJ_BODY,"base_link");
  if(base<0) return;
  mkdir("../run",0700);
  int fd=open("../run/depth.bin",O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW,0600);
  if(fd<0 || ftruncate(fd,sizeof(DepthFrame))) {if(fd>=0) close(fd);return;}
  auto p=mmap(nullptr,sizeof(DepthFrame),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);close(fd);
  if(p==MAP_FAILED)return;
  shared=static_cast<DepthFrame*>(p);
  std::memset(shared,0,sizeof(*shared));shared->magic=depth_magic;shared->version=1;
  double h=2*std::tan(89.51*M_PI/360),v=2*std::tan(58.29*M_PI/360);
  for(int y=0;y<36;++y)for(int x=0;x<64;++x){
    double yy=(.5-(x+.5)/64)*h,zz=(.5-(y+.5)/36)*v,n=std::sqrt(1+yy*yy+zz*zz);
    int i=3*(y*64+x);rays[i]=1/n;rays[i+1]=yy/n;rays[i+2]=zz/n;
  }
  initialized=true;
  fprintf(stderr,"[DEPTH] same-world 64x36 camera at 50 Hz -> 32x18, shared frame ready\n");
}
int reflect(int x,int n){return x<0?-x:(x>=n?2*n-x-2:x);}
void capture(const mjModel* m,mjData* d){
  if(m!=last_model){last_model=m;deadline=0;initialized=false;}
  if(!initialized) initialize(m);
  if(!shared || base<0)return;
  if(d->time+1e-8<deadline && d->time>deadline-.1)return;
  deadline=d->time+.02;
  const double* rotation=d->xmat+base*9;
  double p[3],local_rotation[9],rotation_camera[9];
  double quat[4]={mount_quat_xyzw[3],mount_quat_xyzw[0],mount_quat_xyzw[1],mount_quat_xyzw[2]};
  mju_quat2Mat(local_rotation,quat);
  mju_mulMatMat(rotation_camera,rotation,local_rotation,3,3,3);
  mju_mulMatVec(p,rotation,mount_pos,3,3);mju_addTo3(p,d->xpos+base*3);
  for(int i=0;i<2304;++i)mju_mulMatVec(world_rays.data()+3*i,rotation_camera,rays.data()+3*i,3,3);
  const mjtByte groups[6]={1,0,1,0,0,0};
  mj_multiRay(m,d,p,world_rays.data(),groups,1,-1,geom_ids.data(),distances.data(),2304,1e6);
  float crop[18*32];
  for(int y=0;y<18;++y)for(int x=0;x<32;++x){
    int i=(y+18)*64+x+16;
    double distance=distances[i]*rays[3*i];
    if(geom_ids[i]<0 || distances[i]<0 || !std::isfinite(distance))distance=2.5;
    crop[y*32+x]=std::clamp(distance,0.,2.5);
  }
  float frame[576];const float k[3]={.27406862f,.45186276f,.27406862f};
  for(int y=0;y<18;++y)for(int x=0;x<32;++x){float value=0;
    for(int a=-1;a<=1;++a)for(int b=-1;b<=1;++b)value+=k[a+1]*k[b+1]*crop[reflect(y+a,18)*32+reflect(x+b,32)];
    frame[y*32+x]=std::clamp(value/2.5f,0.f,1.f);
  }
  __atomic_add_fetch(&shared->sequence,1,__ATOMIC_SEQ_CST);
  shared->simulation_time=d->time;shared->monotonic_time=now();
  std::memcpy(shared->depth,frame,sizeof(frame));
  std::copy_n(d->xpos+3*base,3,shared->root_position);
  std::copy_n(d->xquat+4*base,4,shared->root_quaternion);
  __atomic_add_fetch(&shared->sequence,1,__ATOMIC_SEQ_CST);
}
}
extern "C" void mj_step(const mjModel* m,mjData* d){
  static auto step=reinterpret_cast<void(*)(const mjModel*,mjData*)>(dlsym(RTLD_NEXT,"mj_step"));
  if(!step){fprintf(stderr,"[DEPTH] mj_step symbol missing\n");_exit(1);}
  // The GUI's replay helper also uses MuJoCo. It must never publish live sensor frames.
  static bool sensor_process=[](){
    char path[4096]={};ssize_t length=readlink("/proc/self/exe",path,sizeof(path)-1);
    const char* name=length>0?std::strrchr(path,'/'):nullptr;
    const char* test=std::getenv("LOONG_SIM_DEPTH_TEST");
    return (name && std::strcmp(name+1,"loong_share_sim_x64")==0) || (test && std::strcmp(test,"1")==0);
  }();
  step(m,d);if(sensor_process)capture(m,d);
}
