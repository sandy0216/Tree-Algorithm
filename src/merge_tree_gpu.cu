#include <cstdio>
#include <cstring>
#include <omp.h>
#include <vector>
#include "../inc/def_node.h"
#include "../inc/tool_tree_gpu.h"
#include "../inc/param.h"
#include "../inc/force_gpu.h"


extern __shared__ GNODE cache[];

__global__ void merge_bottom(double *x, double *y, double *mass, int *particle_index, unsigned int *regnum, int *n, int *region_index,int *thread_load, double *rx, double *ry, double *rmass, int *rn)
{
	int nx = blockDim.x*gridDim.x;
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
			rx[reg_id] = xcm/totalmass;
			ry[reg_id] = ycm/totalmass;
			rmass[reg_id] = totalmass;
			rn[reg_id] = par_n;
		}else{
			rn[reg_id] = 0;
			rx[reg_id] = 0;
			ry[reg_id] = 0;
			rmass[reg_id] = 0;
		}
	}
}

__global__ void merge_top(double *px,double *py,double *pmass, double *x, double *y, double *mass, int *num, int *region_index, int *thread_load, unsigned int *regnum, int *particle_index, double *fx, double *fy, double *V, GNODE *root,double *flag)
{
	int reg_per_block = (d_side/bx)*(d_side/bx);
	int side_per_block = d_side/bx;
	int block_id  = blockIdx.x+blockIdx.y*gridDim.x;
	int thread_id = threadIdx.x+threadIdx.y*blockDim.x;
	int reg_start = block_id*reg_per_block;
	int reg_id;
	int cache_id;
	while( thread_id<reg_per_block ){
		reg_id = region_index[reg_start+thread_id];
		cache_id = share_node-reg_per_block+thread_id;
		/*if( blockIdx.x==0 && blockIdx.y==0 ){
		flag[cache_id] = x[reg_id];
		}*/
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
			cache_id = share_node-reg_per_block+thread_id;
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
			cache[cache_id].side = cache[cache_id*4+1].side*2;
			/*if( blockIdx.x==0 && blockIdx.y==0 ){
				flag[cache_id] = cache[cache_id].centerofmass[0];
			}*/

			thread_id += blockDim.x*blockDim.y;
		}
		__syncthreads();
	}// end of 'while side_per_block != 1'
	
	int global_id;
	int side_per_grid = bx;
	int block_per_grid = bx*bx;
	if( threadIdx.x==0 && threadIdx.y==0 ){
		global_id = global_node-bx*by+block_id;
		root[global_id].num = cache[0].num;
		root[global_id].centerofmass[0] = cache[0].centerofmass[0];
		root[global_id].centerofmass[1] = cache[0].centerofmass[1];
		root[global_id].mass = cache[0].mass;
		root[global_id].side = cache[0].side;
		flag[global_id] = root[global_id].side;
	}
	if( blockIdx.x==0 && blockIdx.y==0 ){
		while( side_per_grid != 0 ){
			block_id = threadIdx.x+threadIdx.y*blockDim.x;
			side_per_grid /= 2;
			block_per_grid += side_per_grid*side_per_grid;
			while( block_id<side_per_grid*side_per_grid ){
				global_id = global_node-block_per_grid+block_id;
				cmx=cmy=tm=0.0;
				cn = 0;
				for( int i=1;i<5;i++ ){
					cmx += root[4*global_id+i].centerofmass[0]*root[4*global_id+i].mass;
					cmy += root[4*global_id+i].centerofmass[1]*root[4*global_id+i].mass;
					tm += root[4*global_id+i].mass;
					cn += root[4*global_id+i].num;
				}
				root[global_id].centerofmass[0] = cmx/tm;
				root[global_id].centerofmass[1] = cmy/tm;
				root[global_id].mass = tm;
				root[global_id].num = cn;
				root[global_id].leaf = 0;
				root[global_id].side = root[4*global_id+1].side*2;
				flag[global_id] = root[global_id].side;
				block_id += blockDim.x*blockDim.y;
			}
			__syncthreads();
		}
	}
	//====================Calculate force===========================
	force_gpu(root,px,py,pmass,fx,fy,V,region_index,thread_load,regnum,particle_index);	

}












