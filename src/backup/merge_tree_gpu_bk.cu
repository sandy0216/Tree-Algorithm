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


//__global__ void merge_bottom(double *x, double *mass, unsigned int *regnum, int *n, int *region_index,int *thread_load, double *rx, double *ry, double *rmass, int *rn)
/*__global__ void merge_bottom(double *x, double *y, double *mass, unsigned int *regnum, int *n, int *region_index,int *thread_load, double *rx, double *ry, double *rmass, int *rn)

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
				par_id = st_par+j;
				xcm += x[par_id]*mass[par_id];
				ycm += y[par_id]*mass[par_id];
				totalmass += mass[par_id];
			}
			rx[reg_id] = x[];//xcm/totalmass;
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
}*/

__global__ void merge_top(double *px,double *py,double *pmass, double *x, double *y, double *mass, int *num, int *thread_load, unsigned int *regnum, double *fx, double *fy, double *V, int *block_index,GNODE *root,double *flag)
{
	int reg_per_block = (d_side/d_bx)*(d_side/d_bx);
	int side_per_block = d_side/d_bx;
	int block_id  = blockIdx.x+blockIdx.y*gridDim.x;
	int thread_id = threadIdx.x+threadIdx.y*blockDim.x;
	int reg_start = block_id*reg_per_block;
	int reg_id;
	int cache_id;
	while( thread_id<reg_per_block ){
		reg_id = reg_start+thread_id;
		cache_id = share_node-reg_per_block+thread_id;
		if( blockIdx.x==bx/2 && blockIdx.y==bx/2 ){
			flag[cache_id] = (double)x[reg_start+thread_id];
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
			cache_id = share_node-reg_per_block+thread_id;
			cmx = cmy = tm = 0.0;
			cn = 0;
			for( int i=1;i<5;i++ ){
				cmx += cache[4*cache_id+i].centerofmass[0]*cache[4*cache_id+i].mass;
				cmy += cache[4*cache_id+i].centerofmass[1]*cache[4*cache_id+i].mass;
				tm += cache[4*cache_id+i].mass;
				cn += cache[4*cache_id+i].num;
			}
			if( cn==0 ){
				cache[cache_id].centerofmass[0] = 0;
				cache[cache_id].centerofmass[1] = 0;
			}else{
				cache[cache_id].centerofmass[0] = cmx/tm;
				cache[cache_id].centerofmass[1] = cmy/tm;
			}
			cache[cache_id].mass = tm;
			cache[cache_id].num = cn;
			cache[cache_id].leaf = 0;
			cache[cache_id].side = cache[cache_id*4+1].side*2;
			if( blockIdx.x==bx/2 && blockIdx.y==bx/2 ){
				flag[cache_id] = (double)cache[cache_id].centerofmass[0];
			}

			thread_id += blockDim.x*blockDim.y;
		}
		__syncthreads();
	}// end of 'while side_per_block != 1'
	int nx = blockDim.x*gridDim.x;
	int ix = blockDim.x*blockIdx.x+threadIdx.x;
	int iy = blockDim.y*blockIdx.y+threadIdx.y;
	thread_id = ix+iy*nx;
	
	int st_reg, st_par;
	int par_id;
	int par_n;
	double loc_fx,loc_fy,loc_v;
	if( thread_id==0 ){ st_reg=0; }
	else{ st_reg = thread_load[thread_id]-1; }
	for( int i=st_reg;i<thread_load[thread_id];i++ ){
		if( i==0 ){
			st_par = 0;
			par_n = regnum[i];
		}else{
			st_par = regnum[i-1];
			par_n = regnum[i]-regnum[i-1];
		}
		if( par_n!=0 ){
			for( int j=0;j<par_n;j++ ){
				par_id = st_par+j;
				loc_fx = 0;
				loc_fy = 0;
				loc_v = 0;
				node_share_force_gpu(0,px[par_id],py[par_id],pmass[par_id],&loc_fx,&loc_fy,&loc_v);
				fx[par_id] = loc_fx;
				fy[par_id] = loc_fy;
				V[par_id] = loc_v;
			}
		}
	}
	
	int global_id;
	int side_per_grid = gridDim.x;
	int block_per_grid = gridDim.x*gridDim.x;
	if( threadIdx.x==0 && threadIdx.y==0 ){
		global_id = global_node-gridDim.x*gridDim.y+block_index[block_id];
		root[global_id].num = cache[0].num;
		root[global_id].centerofmass[0] = cache[0].centerofmass[0];
		root[global_id].centerofmass[1] = cache[0].centerofmass[1];
		root[global_id].mass = cache[0].mass;
		root[global_id].side = cache[0].side;
		//flag[global_id] = root[global_id].num;
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
				//flag[global_id] = root[global_id].num;
				block_id += blockDim.x*blockDim.y;
			}
			__syncthreads();
		}
	}
	//====================Calculate force===========================
	//force_gpu(root,px,py,pmass,fx,fy,V,region_index,thread_load,regnum);	

}


__global__ void merge_global(double *px,double *py,double *pmass, int *thread_load, unsigned int *regnum, double *fx, double *fy, double *V, int *block_index,GNODE *root)
{
	int block_id = threadIdx.x+threadIdx.y*blockDim.x;
	int side_per_grid = gridDim.x;
	int block_per_grid = gridDim.x*gridDim.x;
	int global_id;
	double cmx,cmy,tm;
	int cn;
	if( blockIdx.x==0 && blockIdx.y==0 ){
	while( side_per_grid != 0 ){
		side_per_grid /= 2;
		block_id = threadIdx.x+threadIdx.y*blockDim.x;
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
			//flag[global_id] = root[global_id].num;
			block_id += blockDim.x*blockDim.y;
		}
		__syncthreads();
	}
	}
}










