#ifndef TOOL_MAIN_H
#define TOOL_MAIN_H

void delete_par(double *x, double *y, double *mass, int index);
void check_boundary(double *x,double *y,double *mass,int *n,double boxsize);
void update(double *x, double *y, double *vx, double *vy, int n,double dt);

#endif

