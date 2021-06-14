#include <cstdio>
#include <cstring>
#include "../inc/def_node.h"
#include "../inc/print_tree.h"
#include "../inc/tool_tree.h"
#include "../inc/param.h"


__global__ void split(double *x, double *y, double *mass)
{
	int num = initial_n;
	int nx = blockDim.x*gridDim.x;
	int ny = blockDim.y*gridDim.y;
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	double xlen = boxsize/nx;
	double ylen = boxsize/ny;
	
	for( int i=0;i<num;i++ ){
	if( x[i]>xlen*ix && x[i]<xlen*(ix+1) && y[i]<ylen*(iy+1) && y[i]>ylen*iy ){
		
	}
	}
	
}
