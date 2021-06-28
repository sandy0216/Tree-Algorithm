#ifndef TREE_GPU_H
#define TREE_GPU_H

__global__ void treeforce(double *px,double *py,double *pmass,double *fx,double *fy,double *V,int *rn,int *region_index,int *thread_load,int *n);

#endif
