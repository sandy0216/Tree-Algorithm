#ifndef FORCE_H
#define FORCE_H

void force(NODE* head, double *x, double *y, double *mass, double *fx, double *fy, int n);
void direct_nbody( double px, double py, double pmass, double *x, double *y, double *mass, int n, double *fx, double *fy);
void potential(NODE* head, double *x, double *y, double *mass, double *v, int n);

#endif
