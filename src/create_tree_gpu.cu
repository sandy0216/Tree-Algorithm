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


__global__ void tree(double *x, double *y, double *mass, int *particle_index , unsigned int *regnum, int *n, int *side, double* boxsize, NODE *local,int *region_index, int *thread_load, int* flag)
{
	int nx = blockDim.x*gridDim.x;
	int ny = blockDim.y*gridDim.y;
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_id = ix+iy*nx;
	NODE  *root,*copy;
	int localn;		// Number of particles in each region
	int start, pindex;
	int a,b;
	int region_id;
	double length = *boxsize/(*side);
	int start_r;
	if( thread_id==0 ){ start_r = 0;            }
	else{	start_r = thread_load[thread_id-1]; }
	flag[ix+iy*nx] = 0;
	for( int i=start_r;i<thread_load[thread_id];i++ ){
		root = new NODE();
		region_id = region_index[i];
		if( region_id==0 ){ localn = regnum[region_id]; }
		else{ localn = regnum[region_id]-regnum[region_id-1]; }
		a = region_id%(*side);
		b = region_id/(*side);
		if( localn!= 0 ){ // If there do exist particle in a region
			if( region_id==0 ){ start=0; }
			else{ start=regnum[region_id-1]; }
			pindex = particle_index[start];
			create_node_gpu(root,length*((double)a+0.5),length*((double)b+0.5),x[pindex],y[pindex],mass[pindex],length);
			flag[ix+iy*nx] += 1;
			if( localn>1 ){ // if there are more than one particle in the region
				for( int j=1;j<localn;j++ ){
					pindex = particle_index[start+j];
					add_particle_gpu(root,x[pindex],y[pindex],mass[pindex],&flag[ix+iy*nx]);
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
		copy = &local[region_id];
		copy->num = root->num;
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
