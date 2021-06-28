#ifndef TREE_GPU_TOOL_H
#define TREE_GPU_TOOL_H

__device__ void create_node_gpu(TNODE *head,double cmx,double cmy,double mass,double side);
__device__ void add_particle_gpu(TNODE *head,double cx,double cy,double x,double y,double mass);


#endif
