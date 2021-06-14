#ifndef TOOL_TREE_H
#define TOOL_TREE_H

__device__ void create_node(NODE *head,double cx, double cy, double x, double y, double mass, double side);
__device__ int region(double x, double y, double cx, double cy);
__device__ void finest_grid(NODE *current, double x, double y, double mass);



#endif
