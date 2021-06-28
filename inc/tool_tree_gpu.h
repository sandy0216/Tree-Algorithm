#ifndef TOOL_TREE_H
#define TOOL_TREE_H

//__device__ void create_node_gpu(NODE *head,double cx, double cy, double x, double y, double mass, double side);
//__device__ int region_gpu(double x, double y, double cx, double cy);
__device__ void finest_grid_gpu(NODE *current, double x, double y, double mass,int *flag);
//__device__ void add_particle_gpu(NODE *head, double x, double y, double mass,int *flag);



#endif
