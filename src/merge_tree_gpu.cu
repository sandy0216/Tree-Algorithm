#include <cstdio>
#include <cstring>
#include <omp.h>
#include <vector>
#include "../inc/def_node.h"
#include "../inc/tool_tree_gpu.h"
#include "../inc/param.h"
#include "../inc/force_gpu.h"


extern __shared__ GNODE cache[];

__global__ void merge_bottom(double *x, double *y, double *mass, int *index,double *rx, double *ry, double *rmass, int *rn, int *n)
{
	int nx = blockDim.x*gridDim.x;
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int id = ix+iy*nx;
	int reg_id;
	while( id<*n ){
		reg_id = index[id];
		atomicAdd(&rx[reg_id],x[id]*mass[id]);
		atomicAdd(&ry[reg_id],y[id]*mass[id]);
		atomicAdd(&rmass[reg_id],mass[id]);
		atomicAdd(&rn[reg_id],1);
		id += nx*nx;
	}
}

__global__ void merge_bottom2(double *rx,double *ry,double *rmass){
	int nx = blockDim.x*gridDim.x;
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int id = ix+iy*nx;
	while( id<d_n_work ){
		if( rmass[id]!= 0 ){
			rx[id] = rx[id]/rmass[id];
			ry[id] = ry[id]/rmass[id];
		}
		id += nx*nx;
	}
}
	
__global__ void merge_top1(double *rx, double *ry, double *rmass, int *rn,int *morton_index, GNODE *root)
{
	int side_per_grid = d_side;
	int reg_per_grid  = d_side*d_side;
	int nx = blockDim.x*gridDim.x;
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_id = ix+iy*nx;
	int global_id;
	int id;
	while( thread_id<reg_per_grid ){
		id = morton_index[thread_id];
		global_id = global_node-reg_per_grid+thread_id;
		root[global_id].centerofmass[0]=rx[id];
		root[global_id].centerofmass[1]=ry[id];
		root[global_id].mass = rmass[id];
		root[global_id].side = d_boxsize/d_side;
		root[global_id].num = rn[id];
		root[global_id].leaf = id;
		thread_id += nx*nx;
	}
}

__global__ void merge_top2(GNODE *root)
{
	int side_per_grid = d_side;
	int reg_per_grid = d_side*d_side;
	int id,global_id;
	double cmx,cmy,tm;
	int cn;
	if( blockIdx.x==0 && blockIdx.y==0 ){
	while( side_per_grid !=0 ){
		id = threadIdx.x+threadIdx.y*blockDim.x;
		side_per_grid /= 2;
		reg_per_grid += pow(side_per_grid,2);
		while( id<pow(side_per_grid,2) ){
			global_id = global_node-reg_per_grid+id;
			cmx = cmy = tm = 0.0;
			cn = 0;
			for( int i=1;i<5;i++ ){
				cmx += root[4*global_id+i].centerofmass[0]*root[4*global_id+i].mass;
				cmy += root[4*global_id+i].centerofmass[1]*root[4*global_id+i].mass;
				tm += root[4*global_id+i].mass;
				cn += root[4*global_id+i].num;
			}
			if( cn==0 ){
				root[global_id].centerofmass[0] = 0;
				root[global_id].centerofmass[1] = 0;
			}else{
				root[global_id].centerofmass[0] = cmx/tm;
				root[global_id].centerofmass[1] = cmy/tm;
			}
			root[global_id].mass = tm;
			root[global_id].num = cn;
			root[global_id].leaf = 0;
			root[global_id].side = root[4*global_id+1].side*2;
			id += blockDim.x*blockDim.y;
		}
		__syncthreads();
	}
	}
}

















