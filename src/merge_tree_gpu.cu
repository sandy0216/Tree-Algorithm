#include <cstdio>
#include <cstring>
#include <omp.h>
#include <vector>
#include "../inc/def_node.h"
#include "../inc/tool_tree_gpu.h"
#include "../inc/param.h"


__global__ void merge_gpu(double *x, double *y, double *mass, int *particle_index, unsigned int *regnum, int *n, int *side, double *boxsize, NODE *local, int *region_index,int *thread_load, int *flag)
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
	int n_len = *side;
	int a,b;
	double length = *boxsize/(*side);
	NODE* root;
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
		a = reg_id%n_len;
		b = reg_id/n_len;
		root = &local[reg_id];
		if( par_n!=0 ){
			xcm = ycm = totalmass = 0.0;
			for( int j=0;j<par_n;j++ ){
				par_id = particle_index[st_par+j];
				xcm += x[par_id]*mass[par_id];
				ycm += y[par_id]*mass[par_id];
				totalmass += mass[par_id];
			}
			*flag = 1;
			root->centerofmass[0] = xcm/totalmass;
			root->centerofmass[1] = ycm/totalmass;
			root->mass = totalmass;
			root->num = par_n;
		}else{
			root->num = 0;
		}
	}
}

__device__ void merge(NODE *root, NODE *next){
	NODE *lroot,*lnext;
	double cmx = 0.0;
	double cmy = 0.0;
	double tm  = 0.0;
	int num = 0;
	for( int i=0;i<4;i++ ){
		lnext = &next[i];	
	       	tm += lnext->mass;
		cmx += lnext->centerofmass[0]*(lnext->mass);
		cmy += lnext->centerofmass[1]*(lnext->mass);
		num += lnext->num;
		lroot->next[i] = lnext;
	}
	lroot->num = num;
	lroot->mass = tm;
	lroot->centerofmass[0] = cmx;
	lroot->centerofmass[1] = cmy;
}







