#include <cstdio>
#include <cstring>
#include <omp.h>
#include <vector>
#include "../inc/def_node.h"
#include "../inc/tool_tree_gpu.h"
#include "../inc/param.h"


__global__ void merge_gpu(double *x, double *y, double *mass, int *particle_index, unsigned int *regnum, int *n, int *region_index,int *thread_load, double *rx, double *ry, double *rmass, int *rn, int *flag)
{
	int nx = blockDim.x*gridDim.x;
	int ny = blockDim.y*gridDim.y;
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_id = ix+iy*nx;
	double totalmass,xcm,ycm;
	int st_reg, st_par;
	int reg_id, par_id;
	int par_n;
	if( thread_id==0 ){ st_reg = 0; }
	else{ st_reg = thread_load[thread_id-1]; }
	for( int i=st_reg;i<thread_load[thread_id];i++ ){
		reg_id = region_index[i];
		if( reg_id==0 ){ 
			st_par=0;
			par_n = regnum[reg_id];
	       	}else{ 
			st_par = regnum[reg_id-1]; 
			par_n = regnum[reg_id]-regnum[reg_id-1];
		}
		if( par_n!=0 ){
			xcm = ycm = totalmass = 0.0;
			for( int j=0;j<par_n;j++ ){
				par_id = particle_index[st_par+j];
				xcm += x[par_id]*mass[par_id];
				ycm += y[par_id]*mass[par_id];
				totalmass += mass[par_id];
			}
			*flag = 1;
			rx[reg_id] = xcm/totalmass;
			ry[reg_id] = ycm/totalmass;
			rmass[reg_id] = totalmass;
			rn[reg_id] = par_n;
		}else{
			rn[reg_id] = 0;
		}
	}
}

__global__ void merge_top(double *x, double *y, double *mass, int *num, int *region_index, int *flag){
	extern __shared__ GNODE cache[];
	extern __device__ int dn_gnode;
	int reg_per_block = (d_side/bx)*(d_side/bx);
	int side_per_block = d_side/bx;
	int reg_start = (blockIdx.x+blockIdx.y*gridDim.x)*reg_per_block;
	int thread_id = threadIdx.x+threadIdx.y*blockDim.x;
	int reg_id;
	int cache_id;
	while( thread_id<reg_per_block ){
		reg_id = region_index[reg_start+thread_id];
		cache_id = dn_gnode-reg_per_block+thread_id;
		if( blockIdx.x==0 && blockIdx.y==0 ){
		flag[cache_id] = num[reg_id];
		}
		cache[cache_id].centerofmass[0]=x[reg_id];
		cache[cache_id].centerofmass[1]=y[reg_id];
		cache[cache_id].mass = mass[reg_id];
		cache[cache_id].side = d_boxsize/d_side;
		cache[cache_id].num = num[reg_id];
		cache[cache_id].leaf = 1;
		thread_id += blockDim.x*blockDim.y;
	}
	__syncthreads();
	double cmx,cmy,tm;
	int cn;

	while(side_per_block != 0 ){
		thread_id = threadIdx.x+threadIdx.y*blockDim.x;
		side_per_block /= 2;
		reg_per_block += side_per_block*side_per_block;
		while( thread_id<side_per_block*side_per_block ){
			cache_id = dn_gnode-reg_per_block+thread_id;
			cmx = cmy = tm = 0.0;
			cn = 0;
			for( int i=1;i<5;i++ ){
				cmx += cache[4*cache_id+i].centerofmass[0]*cache[4*cache_id+i].mass;
				cmy += cache[4*cache_id+i].centerofmass[1]*cache[4*cache_id+i].mass;
				tm += cache[4*cache_id+i].mass;
				cn += cache[4*cache_id+i].num;
			}
			cache[cache_id].centerofmass[0] = cmx/tm;
			cache[cache_id].centerofmass[1] = cmy/tm;
			cache[cache_id].mass = tm;
			cache[cache_id].num = cn;
			cache[cache_id].leaf = 0;
			if( blockIdx.x==0 && blockIdx.y==0 ){
				flag[cache_id] = cache[cache_id].num;
			}

			thread_id += blockDim.x*blockDim.y;
		}
		__syncthreads();
	}// end of 'while side_per_block != 1'


			



}












