//--------------------------------
//
// This function generate a 2D random initial consition with 2D normal distribution
//
// -------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>

void init( double *x, double *y, double *mass, const int n, const double boxsize, const double maxmass)
{
	for( long i=0;i<n;i++ ){
		x[i] = (double) rand()/(float)RAND_MAX*boxsize;
		y[i] = (double) rand()/(float)RAND_MAX*boxsize;
		mass[i] = (double) rand()/(float)RAND_MAX*maxmass;
	}
}
