#include <cstdio>
#include "../inc/param.h"


void delete_par(double *x, double *y, double *mass, int index, unsigned long *n)
{
	for( int i=index;i<*n-1;i++ ){
		x[i]=x[i+1];
		y[i]=y[i+1];
		mass[i]=mass[i+1];
	}
}

void check_boundary(double *x,double *y,double *mass,unsigned long *n)
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

void update(double *x, double *y, double *vx, double *vy, unsigned long n)
{
	for( int i=0;i<n;i++ ){
		x[i] += vx[i]*dt;
		y[i] += vy[i]*dt;
	}
}

void block(int n,int stx,int sty,int *region_index,int *which_region)
{
	int ix,iy;
	if( n==2 ){
		//printf("record region %d,%d\n",stx,sty);
		//printf("which_region = %d\n",*which_region);
		for( int i=0;i<4;i++ ){
		ix = stx;
		iy = sty;
		if( i==1 || i==3 ){ ix+=1; }
		if( i==2 || i==3 ){ iy+=1; }
		region_index[*which_region] = (int)ix+nx*iy;
		*which_region += 1;
		}
	}else{
		for( int i=0;i<2;i++ ){
		for( int j=0;j<2;j++ ){
			block(n/2,(int)stx+n/2*i,(int)sty+n/2*j,region_index,which_region);
		}
		}
	}
}
		
