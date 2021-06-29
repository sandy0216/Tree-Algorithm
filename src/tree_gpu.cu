#include <cstdio>
#include <cstring>
#include "../inc/def_node.h"
#include "../inc/param.h"
#include "../inc/tree_gpu_tool.h"

__global__ void treeforce(double *px,double *py,double *pmass,double *fx,double *fy,double *V,int *rn,int *region_index,int *thread_load,int *n)
{
	int nx = blockDim.x*gridDim.x;
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_id = ix+iy*nx;
	int st_reg, st_par;
	int reg_id, par_id, pid;
	int par_n;
	double r,cfx,cfy,cv,x,y,m,side,temp,temp_v;
	if( thread_id==0 ){ st_reg = 0; }
	else{ st_reg = thread_load[thread_id-1]; }
#ifdef BALANCE
	for( int i=st_reg;i<thread_load[thread_id];i++ ){
		reg_id = region_index[i];
#else
	while( thread_id<d_n_work ){
		reg_id = thread_id;
#endif
		if( reg_id==0 ){
			st_par = 0;
			par_n = rn[reg_id];
		}else{
			st_par = rn[reg_id-1];
			par_n = rn[reg_id]-rn[reg_id-1];
		}
		cfx = 0;
		cfy = 0;
		temp_v = 0;
		for( int j=0;j<par_n;j++ ){
			par_id = st_par + j;
			x = px[par_id];
			y = py[par_id];
			m = pmass[par_id];
			for( int k=j+1;k<par_n;k++ ){
				pid = st_par+k;
				r = sqrt(pow(x-px[pid],2)+pow(y-py[pid],2));
				if( r>d_eplison ){
					cfx = m*pmass[pid]/pow(r,3)*(px[pid]-x);
					cfy = m*pmass[pid]/pow(r,3)*(py[pid]-y);
					cv  = -m*pmass[pid]/r;
				}else{
					temp = sqrt(pow(r,2)+pow(d_eplison,2));
					cfx = m*pmass[pid]/pow(temp,3)*(px[pid]-x);
					cfy = m*pmass[pid]/pow(temp,3)*(py[pid]-y);
					cv = -m*pmass[pid]/temp;
				}
				fx[par_id] += cfx;
				fy[par_id] += cfy;
				fx[pid] -= cfx;
				fy[pid] -= cfy;
				temp_v += cv;
			}
		}
		atomicAdd(V,temp_v);
#ifdef BALANCE

#else
		thread_id += nx*nx;
#endif	

	}		
}



