#include <cstdio>
#include <cstring>
#include <omp.h>
#include <vector>
#include "../inc/def_node.h"
#include "../inc/tool_tree_gpu.h"
#include "../inc/param.h"

__global__ void split(double *x, double *y, int *index, unsigned int *regnum, int *n, int *side, double *boxsize)
{
	int nx = blockDim.x*gridDim.x;
	int ny = blockDim.y*gridDim.y;
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_id = ix+iy*nx;
	int a,b;
	while( thread_id<*n ){
		a = x[thread_id]/(*boxsize/ *side);
		b = y[thread_id]/(*boxsize/ *side);
		index[thread_id] = a+b*(*side);
		atomicAdd(&regnum[a+b*(*side)],1);
		thread_id += nx*ny;
	}
}


__global__ void tree(double *x, double *y, double *mass, int *particle_index , unsigned int *regnum, int *n, int *side, double* boxsize, NODE *local)
{
	int nx = blockDim.x*gridDim.x;
	int ny = blockDim.y*gridDim.y;
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_id = ix+iy*nx;
	NODE  *root;
	int localn;
	int start, pindex;
	int n_work = (*side)*(*side);	
	while( thread_id<n_work ){
		root = new NODE();
		if( thread_id==0 ){ localn = regnum[thread_id]; }
		else{ localn =regnum[thread_id]-regnum[thread_id-1]; }
		
		if( localn!= 0 ){
			if( thread_id==0 ){ start=0; }
			else{ start=regnum[thread_id-1]; }
			pindex = particle_index[start];
			create_node_gpu(root,*boxsize/2,*boxsize/2,x[pindex],y[pindex],mass[pindex],*boxsize);
			for( int i=0;i<localn;i++ ){
				if( thread_id==0 ){ start=0; }
				else{ start=regnum[thread_id-1]; }
				pindex = particle_index[start+i];
				add_particle_gpu(root,x[pindex],y[pindex],mass[pindex]);
			}
		}
	}
		thread_id += nx*ny;
}
	

