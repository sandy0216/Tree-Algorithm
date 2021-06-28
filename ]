#ifndef MERGE_TREE_GPU_H
#define MERGE_TREE_GPU_H

//__global__ void merge_bottom(double *x, double *mass, unsigned int *regnum, int *n, int *region_index,int *thread_load, double *rx, double *ry, double *rmass, int *rn);
//__global__ void merge_bottom(double *x, double *y, double *mass, unsigned int *regnum, int *n, int *region_index,int *thread_load, double *rx, double *ry, double *rmass, int *rn);

//__global__ void merge_top(double *px,double *py,double *pmass, double *x, double *y, double *mass, int *num,  int *thread_load, unsigned int *regnum, double *fx, double *fy, double *V,int *block_index,GNODE *root,double *flag);
//__global__ void merge_global(double *px,double *py,double *pmass, int *thread_load, unsigned int *regnum, double *fx, double *fy, double *V,int *block_index,GNODE *root);
__global__ void merge_bottom(double *x, double *y, double *mass, int *index,double *rx, double *ry, double *rmass, int *rn, int *n);
__global__ void merge_bottom2(double *rx, double *ry, double *rmass);
__global__ void merge_top1(double *rx,double *ry,double *rmass,int *rn,int *morton_index,GNODE *root);
__global__ void merge_top2(GNODE *root);


#endif
