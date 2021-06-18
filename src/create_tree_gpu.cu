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


__global__ void tree(double *x, double *y, double *mass, int *particle_index , unsigned int *regnum, int *n, int *side, double* boxsize, NODE *local,int* flag)
{
	int nx = blockDim.x*gridDim.x;
	int ny = blockDim.y*gridDim.y;
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_id = ix+iy*nx;
	NODE  *root,*copy;
	int localn;
	int start, pindex;
	int n_work = (*side)*(*side);	
	int a,b;
	double length = *boxsize/(*side);
	while( thread_id<n_work ){
		root = new NODE(); //&local[thread_id];
		if( thread_id==0 ){ localn = regnum[thread_id]; }
		else{ localn = regnum[thread_id]-regnum[thread_id-1]; }
		
		a = thread_id%(*side);
		b = thread_id/(*side);
		if( localn!= 0 ){
			if( thread_id==0 ){ start=0; }
			else{ start=regnum[thread_id-1]; }
			a = thread_id%(*side);
			b = thread_id/(*side);
			pindex = particle_index[start];
			create_node_gpu(root,length*((double)a+0.5),length*((double)b+0.5),x[pindex],y[pindex],mass[pindex],length);
			if( localn>1 ){
				for( int i=1;i<localn;i++ ){
					pindex = particle_index[start+i];
					add_particle_gpu(root,x[pindex],y[pindex],mass[pindex],&flag[thread_id]);
				}
			}
		}else{
			root->center[0] = length*(a+1/2);
			root->center[1] = length*(b+1/2);
			root->side = length;
			root->num = 0;
			root->leaf = 0;
			for( int i=0;i<4;i++ ){
				root->next[i] = NULL;
			}
		}
		copy = &local[thread_id];
		copy->center[0] = root->center[0];
		copy->center[1] = root->center[1];
		copy->side = root->side;
		copy->num  = root->num;
		copy->leaf = root->leaf;
		copy->centerofmass[0] = root->centerofmass[0];
		copy->centerofmass[1] = root->centerofmass[1];
		thread_id += nx*ny;
	}
}
	
/*__global__ merge_gpu(NODE *local)
{
	int nx = blockDim.x*gridDim.x;
	int ny = blockDim.y*gridDim.y;
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_id = ix+iy*nx;
	NODE  *root;


}*/
