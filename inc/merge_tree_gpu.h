#ifndef MERGE_TREE_GPU_H
#define MERGE_TREE_GPU_H

__global__ void merge_gpu(double *x, double *y, double *mass, int *particle_index, unsigned int *regnum, int *n, int *region_index,int *thread_load, double *rx, double *ry, double *rmass, int *rn, int *flag);
__global__ void merge_top(double *x, double *y, double *mass, int *num, int *region_index, int *flag);

#endif
