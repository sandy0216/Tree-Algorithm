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

void init_binary(double *x,double *y,double *mass, double *vx, double *vy, const unsigned long n, const double boxsize)
{
	double r,ang;
	for( long i=0;i<n/2;i++ ){
		r = (double) rand()/(float)RAND_MAX*20;
		ang = (double) rand()/(float)RAND_MAX*2*3.141592;
		x[i] = 35+ r*cos(ang);
		y[i] = 35+ r*sin(ang);
		mass[i] = 10;
		vx[i] = -sqrt(5*n/30)*rand()/(float)RAND_MAX;
		vy[i] = sqrt(5*n/30)*rand()/(float)RAND_MAX;
	}
	for( long i=n/2;i<n;i++ ){
		r = (double) rand()/(float)RAND_MAX*20;
		ang = (double) rand()/(float)RAND_MAX*2*3.141592;
		x[i] = 65+ r*cos(ang);
		y[i] = 65+ r*sin(ang);
		mass[i] = 10;
		vx[i] = sqrt(5*n/30)*rand()/(float)RAND_MAX;
		vy[i] = -sqrt(5*n/30)*rand()/(float)RAND_MAX;
	}
}


