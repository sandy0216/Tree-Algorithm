//--------------------------------
//
// This function generate a 2D random initial consition with 2D normal distribution
//
// -------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>

void init( double *x, double *y, double *mass, const unsigned long n, const double region, const double boxsize, const double maxmass)
{
	for( long i=0;i<n;i++ ){
		x[i] = (double) rand()/(float)RAND_MAX*region+boxsize/2-region/2;
		y[i] = (double) rand()/(float)RAND_MAX*region+boxsize/2-region/2;
		mass[i] = (double) rand()/(float)RAND_MAX*maxmass;
	}
}
