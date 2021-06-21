#ifndef FORCE_GPU_H
#define FORCE_GPU_H

__device__ void force_gpu(GNODE *root,double *x,double *y,double *mass,double *fx,double *fy,double *V,int *region_index,int *thread_load,unsigned int *regnum,int *particle_index);

#endif

