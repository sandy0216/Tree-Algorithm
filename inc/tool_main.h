#ifndef TOOL_MAIN_H
#define TOOL_MAIN_H

void delete_par(double *x, double *y, double *mass, int index, int *n);
void check_boundary(double *x,double *y,double *mass,int *n);
void update(double *x, double *y, double *vx, double *vy, int n);

#endif

