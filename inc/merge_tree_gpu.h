#ifndef MERGE_TREE_GPU_H
#define MERGE_TREE_GPU_H

__global__ void merge_bottom(double *x, double *y, double *mass, int *particle_index, unsigned int *regnum, int *n, int *region_index,int *thread_load, double *rx, double *ry, double *rmass, int *rn);
__global__ void merge_top(double *px,double *py,double *pmass, double *x, double *y, double *mass, int *num, int *region_index, int *thread_load, unsigned int *regnum, int *particle_index, double *fx, double *fy, double *V,GNODE *root,double *flag);

#endif
