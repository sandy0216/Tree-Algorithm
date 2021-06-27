#ifndef FORCE_GPU_H
#define FORCE_GPU_H

__global__ void force_gpu(GNODE *root,double *x,double *y,double *mass,double *fx,double *fy,double *V,int *n);



#endif

