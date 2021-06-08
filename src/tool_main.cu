#include <cstdio>


void delete_par(double *x, double *y, double *mass, int index, int *n)
{
	for( int i=index;i<*n-1;i++ ){
		x[i]=x[i+1];
		y[i]=y[i+1];
		mass[i]=mass[i+1];
	}
}

void check_boundary(double *x,double *y,double *mass,int *n,double boxsize)
{
	int del=0;
	for( int i=0;i<*n;i++ ){
		if( x[i]>boxsize ){
			delete_par(x,y,mass,i,n);
			del++;
		}else if( x[i]<0 ){
			delete_par(x,y,mass,i,n);
			del++;
		}else if( y[i]>boxsize ){
			delete_par(x,y,mass,i,n);
			del++;
		}else if( y[i]<0 ){
			delete_par(x,y,mass,i,n);
			del++;
		}
	}
	*n = *n-del;
}

void update(double *x, double *y, double *vx, double *vy, int n,double dt)
{
	for( int i=0;i<n;i++ ){
		x[i] += vx[i]*dt;
		y[i] += vy[i]*dt;
	}
}


