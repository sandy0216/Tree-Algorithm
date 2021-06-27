#include <cstdio>
#include <cstring>
#include <omp.h>
#include <vector>
#include "../inc/def_node.h"
#include "../inc/tool_tree_gpu.h"
#include "../inc/param.h"

__global__ void split(double *x, double *y, int *index, int *n)
{
	int nx = blockDim.x*gridDim.x;
	int ny = blockDim.y*gridDim.y;
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_id = ix+iy*nx;
	int a,b;
	while( thread_id<*n ){
		a = x[thread_id]/(d_boxsize/ d_side);
		b = y[thread_id]/(d_boxsize/ d_side);
		index[thread_id] = a+b*d_side;
		//atomicAdd(&regnum[a+b*(d_side)],1);
		thread_id += nx*ny;
	}
}

__global__ void spread_par(double *oldx,double *oldy,double *oldmass,double *newx,double *newy,double *newmass,int *sp_index, int *n){
//__global__ void spread_reg(double *oldx,double *newx,int *sp_index){
	int nx = blockDim.x*gridDim.x;
	int ny = blockDim.y*gridDim.y;
	int ix = blockDim.x*blockIdx.x+threadIdx.x;
	int iy = blockDim.y*blockIdx.y+threadIdx.y;
	int id = ix+iy*nx;
	int sp_id;
	while(id<*n){
		sp_id = sp_index[id];
		newx[id] = oldx[sp_id];
		newy[id] = oldy[sp_id];
		newmass[id] = oldmass[sp_id];
		id += nx*ny;
	}
}

__global__ void spread_reg(double *oldx,double *oldy,double *oldmass,int *oldn,double *newx,double *newy,double *newmass,int *newn,int *sp_index){
//__global__ void spread_reg(double *oldx,double *newx,int *sp_index){
	int nx = blockDim.x*gridDim.x;
	int ny = blockDim.y*gridDim.y;
	int ix = blockDim.x*blockIdx.x+threadIdx.x;
	int iy = blockDim.y*blockIdx.y+threadIdx.y;
	int id = ix+iy*nx;
	int sp_id;
	while(id<d_n_work){
		sp_id = sp_index[id];
		newx[id] = oldx[sp_id];
		newy[id] = oldy[sp_id];
		newmass[id] = oldmass[sp_id];
		newn[id] = oldn[sp_id];
		id += nx*ny;
	}
}

__global__ void n_body(double *x,double *y,double *mass,double *fx,double *fy,double *V,int *particle_index,unsigned int *regnum,int *n,int *region_index,int *thread_load)
{
	int nx = blockDim.x*gridDim.x;
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_id = ix+iy*nx;
	int st_reg, st_par;
	int reg_id, par_id;
	int par_n;
	int parp_id;
	double r,cfx,cfy,cv;
	if( thread_id==0 ){ st_reg = 0; }
	else{ st_reg = thread_load[thread_id-1]; }
	for( int i=st_reg;i<thread_load[thread_id];i++ ){
		reg_id = region_index[i];
		if( reg_id==0 ){
			st_par = 0;
			par_n  = regnum[reg_id];
		}else{
			st_par = regnum[reg_id-1];
			par_n  = regnum[reg_id]-regnum[reg_id-1];
		}
		for( int j=0;j<par_n;j++ ){
			par_id = particle_index[st_par+j];
			for( int k=j+1;k<par_n;k++ ){
				parp_id = particle_index[st_par+k];
				r = sqrt(pow(x[par_id]-x[parp_id],2)+pow(y[par_id]-x[parp_id],2));
				if( r>d_eplison ){
					cfx = mass[par_id]*mass[parp_id]/pow(r,3)*(x[parp_id]-x[par_id]);
					cfy = mass[par_id]*mass[parp_id]/pow(r,3)*(y[parp_id]-y[par_id]);
					cv  = -mass[par_id]*mass[parp_id]/r;
				}else{
					cfx = mass[par_id]*mass[parp_id]/sqrt(pow(pow(r,2)+pow(d_eplison,2),3))*(x[parp_id]-x[par_id]);
					cfy = mass[par_id]*mass[parp_id]/sqrt(pow(pow(r,2)+pow(d_eplison,2),3))*(y[parp_id]-y[par_id]);
					cv  = -mass[par_id]*mass[parp_id]/sqrt(pow(r,2)+pow(d_eplison,2));
				}
				fx[par_id] += cfx;
				fx[parp_id] -= cfx;
				fy[par_id] += cfy;
				fy[parp_id] -= cfy;
				V[par_id] += cv;
				V[parp_id] += cv;
			}
		}
	}
	
}

__global__ void update_gpu(double *x,double *y,double *mass,double *vx,double *vy,double *fx,double *fy,double *Ek,int *n)
{
	int nx = blockDim.x*gridDim.x;
	int ny = blockDim.y*gridDim.y;
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int thread_id = ix+iy*nx;
	while( thread_id<*n ){
		x[thread_id] += d_dt*vx[thread_id];
		y[thread_id] += d_dt*vy[thread_id];
		vx[thread_id] += d_dt*fx[thread_id];
		vy[thread_id] += d_dt*fy[thread_id];
		Ek[thread_id] = 0.5*mass[thread_id]*(pow(vx[thread_id],2)+pow(vy[thread_id],2));
		thread_id += nx*ny;
	}
}

__global__ void energy_gpu(double *Ek,double *V,int *n,double *E)
{
	extern __shared__ double cache[];
	
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int cacheIndex = threadIdx.x;

	double temp = Ek[i]+V[i];
	i += blockDim.x*gridDim.x;
	while( i<*n ){
		temp += Ek[i]+V[i];
		i += blockDim.x*gridDim.x;
	}
	cache[cacheIndex] = temp;
	__syncthreads();

	int ib = blockDim.x/2;
	while( ib!=0 ){
		if(cacheIndex<ib){
			cache[cacheIndex] += cache[cacheIndex]+ib;
		}
		__syncthreads();
		ib/=2;
	}
	if( cacheIndex==0 ){
		E[blockIdx.x] = cache[0];
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
	
