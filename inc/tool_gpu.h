#ifndef TOOL_GPU_H
#define TOOL_GPU_H

__global__ void split(double *x, double *y, int *index, unsigned int *regnum, int *n);
__global__ void n_body(double *x,double *y,double *mass,double *fx,double *fy,double *V,int *particle_index,unsigned int *regnum,int *n,int *region_index,int *thread_load);
__global__ void update_gpu(double *x,double *y,double *mass,double *vx,double *vy,double *fx,double *fy,double *Ek,int *n);
__global__ void energy_gpu(double *Ek,double *V,int *n,double *E);

#endif
