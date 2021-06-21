#include <cstdio>
#include <cstring>
#include "../inc/def_node.h"
#include "../inc/tool_tree_gpu.h"
#include "../inc/param.h"


__device__ void node_force_gpu(GNODE *root,int global_id,double x,double y,double mass,double *fx,double *fy,double *v);
__device__ void node_force_gpup(GNODE *root,int global_id,double x,double y,double mass,double *fx,double *fy,double *v);
__device__ void node_share_force_gpu(int cache_id,double x,double y,double mass,double *fx,double *fy,double *v);
__device__ void node_share_force_gpup(int cache_id,double x,double y,double mass,double *fx,double *fy,double *v);
extern __shared__ GNODE cache[];

__device__ void force_gpu(GNODE *root,double *x,double *y,double *mass,double *fx,double *fy,double *V,int *region_index,int *thread_load,unsigned int *regnum,int *particle_index)
{
	int nx = blockDim.x*gridDim.x;
	int ix = blockDim.x*blockIdx.x+threadIdx.x;
	int iy = blockDim.y*blockIdx.y+threadIdx.y;
	int thread_id = ix+iy*nx;
	
	int st_reg, st_par;
	int reg_id, par_id;
	int par_n;

	double loc_fx,loc_fy,loc_v;
	if( thread_id==0 ){ st_reg=0; }
	else{ st_reg = thread_load[thread_id]-1; }
	for( int i=st_reg;i<thread_load[thread_id];i++ ){
		reg_id = region_index[i];
		if( reg_id==0 ){
			st_par = 0;
			par_n = regnum[reg_id];
		}else{
			st_par = regnum[reg_id-1];
			par_n = regnum[reg_id]-regnum[reg_id-1];
		}
		if( par_n!=0 ){
		for( int j=0;j<par_n;j++ ){
			par_id = particle_index[st_par+j];
			loc_fx = 0;
			loc_fy = 0;
			loc_v = 0;
			node_force_gpu(root,0,x[par_id],y[par_id],mass[par_id],&loc_fx,&loc_fy,&loc_v);
			fx[par_id] = loc_fx;
			fy[par_id] = loc_fy;
			V[par_id] = loc_v;
		}
		
		}
		}
}

__device__ void node_force_gpu(GNODE *root,int global_id,double x,double y,double mass,double *fx,double *fy,double *v)
{
	double cmx,cmy,r,side,m;
	cmx = root[global_id].centerofmass[0];
	cmy = root[global_id].centerofmass[1];
	side = root[global_id].side;
	m = root[global_id].mass;
	r = sqrt(pow(cmx-x,2)+pow(cmy-y,2));
	if( side/r<d_theta || global_id>global_node-bx*bx ){
		if( global_id-(global_node-bx*bx)==blockIdx.x+blockIdx.y*gridDim.x ){
			node_share_force_gpu(0,x,y,mass,fx,fy,v);
		}else{
			if( r>d_eplison ){
				*fx += mass*m/pow(r,3)*(cmx-x);
				*fy += mass*m/pow(r,3)*(cmy-y);
				*v  += -mass*m/r;
			}else{
				*fx += mass*m/sqrt(pow(pow(r,2)+pow(d_eplison,2),3))*(cmx-x);
				*fy += mass*m/sqrt(pow(pow(r,2)+pow(d_eplison,2),3))*(cmy-y);
				*v  += -mass*m/sqrt(pow(r,2)+pow(d_eplison,2));
			}
		}
	}else{
		for( int i=1;i<5;i++ ){
			node_force_gpup(root,4*global_id+i,x,y,mass,fx,fy,v);
		}
	}
}

__device__ void node_share_force_gpu(int cache_id,double x,double y,double mass,double *fx,double *fy,double *v)
{
	double cmx,cmy,r,side,m;
	cmx = cache[cache_id].centerofmass[0];
	cmy = cache[cache_id].centerofmass[1];
	side = cache[cache_id].side;
	m = cache[cache_id].mass;
	r = sqrt(pow(cmx-x,2)+pow(cmy-y,2));
	if( side/r>d_theta || cache_id>share_node-(d_side/bx*d_side/bx) ){
		if( r>d_eplison ){
			*fx += mass*m/pow(r,3)*(cmx-x);
			*fy += mass*m/pow(r,3)*(cmy-y);
			*v  += -mass*m/r;
		}else{
			*fx += mass*m/sqrt(pow(pow(r,2)+pow(d_eplison,2),3))*(cmx-x);
			*fy += mass*m/sqrt(pow(pow(r,2)+pow(d_eplison,2),3))*(cmy-y);
			*v  += -mass*m/sqrt(pow(r,2)+pow(d_eplison,2));
		}
	}else{
		for( int i=1;i<5;i++ ){
			node_share_force_gpup(4*cache_id+i,x,y,mass,fx,fy,v);
		}
	}
}


__device__ void node_force_gpup(GNODE *root,int global_id,double x,double y,double mass,double *fx,double *fy,double *v)
{
	double cmx,cmy,r,side,m;
	cmx = root[global_id].centerofmass[0];
	cmy = root[global_id].centerofmass[1];
	side = root[global_id].side;
	m = root[global_id].mass;
	r = sqrt(pow(cmx-x,2)+pow(cmy-y,2));
	if( side/r<d_theta || global_id>global_node-bx*bx ){
		if( global_id-(global_node-bx*bx)==blockIdx.x+blockIdx.y*gridDim.x ){ 
			node_share_force_gpu(0,x,y,mass,fx,fy,v);
		}else{
			if( r>d_eplison ){
				*fx += mass*m/pow(r,3)*(cmx-x);
				*fy += mass*m/pow(r,3)*(cmy-y);
				*v  += -mass*m/r;
			}else{
				*fx += mass*m/sqrt(pow(pow(r,2)+pow(d_eplison,2),3))*(cmx-x);
				*fy += mass*m/sqrt(pow(pow(r,2)+pow(d_eplison,2),3))*(cmy-y);
				*v  += -mass*m/sqrt(pow(r,2)+pow(d_eplison,2));
			}
		}
	}else{
		for( int i=1;i<5;i++ ){
			node_force_gpu(root,4*global_id+i,x,y,mass,fx,fy,v);
		}
	}
}

__device__ void node_share_force_gpup(int cache_id,double x,double y,double mass,double *fx,double *fy,double *v)
{
	double cmx,cmy,r,side,m;
	cmx = cache[cache_id].centerofmass[0];
	cmy = cache[cache_id].centerofmass[1];
	side = cache[cache_id].side;
	m = cache[cache_id].mass;
	r = sqrt(pow(cmx-x,2)+pow(cmy-y,2));
	if( side/r>d_theta || cache_id>share_node-d_side/bx*d_side/bx ){
		if( r>d_eplison ){
			*fx += mass*m/pow(r,3)*(cmx-x);
			*fy += mass*m/pow(r,3)*(cmy-y);
			*v  += mass*m/r;
		}else{
			*fx += mass*m/sqrt(pow(pow(r,2)+pow(d_eplison,2),3))*(cmx-x);
			*fy += mass*m/sqrt(pow(pow(r,2)+pow(d_eplison,2),3))*(cmy-y);
			*v += mass*m/sqrt(pow(r,2)+pow(d_eplison,2));
		}
	}else{
		for( int i=1;i<5;i++ ){
			node_share_force_gpu(4*cache_id+i,x,y,mass,fx,fy,v);
		}
	}
}
