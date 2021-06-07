#ifndef TOOL_TREE_H
#define TOOL_TREE_H

void create_node(NODE *head,double cx, double cy, double x, double y, double mass, double side);
int region(double x, double y, double cx, double cy);
void finest_grid(NODE *current, double x, double y, double mass);



#endif
