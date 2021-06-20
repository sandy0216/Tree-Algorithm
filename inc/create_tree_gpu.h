#ifndef CREATE_TREE_GPU_H
#define CREATE_TREE_GPU_H

__global__ void split(double *x, double *y, int *index, unsigned int *regnum, int *n, int *side, double *boxsize);
__global__ void tree(double *x, double *y, double *mass, int *particle_index , unsigned int *regnum, int *n, int *side, double* boxsize,NODE *local,int* region_index, int* thread_load, int *flag);

#endif
