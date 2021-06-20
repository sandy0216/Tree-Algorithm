#ifndef MERGE_TREE_GPU_H
#define MERGE_TREE_GPU_H

__global__ void merge_gpu(double *x, double *y, double *mass, int *particle_index, unsigned int *regnum, int *n, int *side, double *boxsize, NODE *local, int *region_index,int *thread_load, int *flag);


#endif
