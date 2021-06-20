#ifndef TOOL_MAIN_H
#define TOOL_MAIN_H

void delete_par(double *x, double *y, double *mass, int index, unsigned long *n);
void check_boundary(double *x,double *y,double *mass,unsigned long *n);
void update(double *x, double *y, double *vx, double *vy, unsigned long n);
void block(int n,int stx,int sty,int *region_index,int *which_region);

#endif

