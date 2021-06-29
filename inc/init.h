#ifndef INIT_H
#define INIT_H

void init( double *x, double *y, double *mass, const unsigned long n, const double region, const double boxsize, const double maxmass);
void init_binary(double *x,double *y,double *mass, double *vx, double *vy, const unsigned long n, const double boxsize);
#endif
