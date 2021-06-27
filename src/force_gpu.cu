#include <cstdio>
#include <cstring>
#include "../inc/def_node.h"
#include "../inc/tool_tree_gpu.h"
#include "../inc/param.h"


__device__ void node_force_gpu(GNODE *root,int global_id,double x,double y,double mass,double *fx,double *fy,double *v);
__device__ void node_force_gpup(GNODE *root,int global_id,double x,double y,double mass,double *fx,double *fy,double *v);

__global__ void force_gpu(GNODE *root,double *x,double *y,double *mass,double *fx,double *fy,double *V,int *n)
{
	int nx = blockDim.x*gridDim.x;
	int ix = blockDim.x*blockIdx.x+threadIdx.x;
	int iy = blockDim.y*blockIdx.y+threadIdx.y;
	int id = ix+iy*nx;
	
	double cmx,cmy,r,side,m,temp;
	double loc_fx,loc_fy,loc_v;
	double px,py,pm;
	int global_id;
	int quene[1000];
	int q_len,a,b;
	int st=global_node-d_n_work;
	while( id<*n ){
		loc_fx = 0;
		loc_fy = 0;
		loc_v  = 0;
		px = x[id];
		py = y[id];
		pm = mass[id];
		a = px/(d_boxsize/d_side);
		b = py/(d_boxsize/d_side);
		q_len = 1;
		quene[0] = 0;
		while( q_len!=0 ){
			global_id = quene[q_len-1];
			if( global_id>global_node){
				loc_fx = 200;
				break;
			}
			q_len -= 1;
			cmx = root[global_id].centerofmass[0];
			cmy = root[global_id].centerofmass[1];
			side = root[global_id].side;
			m = root[global_id].mass;	
			r = sqrt((cmx-px)*(cmx-px)+(cmy-py)*(cmy-py));
			if( side/r<d_theta || global_id>=global_node-d_side*d_side ){
				if( (int)a+b*d_side != root[global_id].leaf ){
					if( r>d_eplison ){
						loc_fx += m*pm/(r*r*r)*(cmx-px);
						loc_fy += m*pm/(r*r*r)*(cmy-py);
						loc_v  += -m*pm/r;
					}else{
						temp = r*r+d_eplison*d_eplison;
						loc_fx += m*pm/sqrt(temp*temp*temp)*(cmx-px);
						loc_fy += m*pm/sqrt(temp*temp*temp)*(cmy-py);
						loc_v  += -m*pm/sqrt(temp*temp);
					}
				}
			}else{
				for( int i=0;i<4;i++ ){
					quene[q_len+i] = 4*global_id+i+1;
				}
				q_len += 4;
				if( q_len>1000 ){
					loc_fx = 100;
					break;
				}
			}
		}
		/*for( int i=0;io<d_n_work;i++ ){
			cmx = root[st+i].centerofmass[0];
			cmy = root[st+i].centerofmass[1];
			m = root[st+i].mass;
			r = sqrt((cmx-px)*(cmx-px)+(cmy-py)*(cmy-py));
			loc_fx += m*pm/(r*r*r)*(cmx-px);
			loc_fy += m*pm/(r*r*r)*(cmy-py);
			loc_v  += -m*pm/r;
		}*/
		fx[id] = loc_fx;
		fy[id] = loc_fy;
		atomicAdd(V,loc_v);
		id += nx*nx;
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
			node_force_gpup(root,4*global_id+i,x,y,mass,fx,fy,v);
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
		//*fx += 100;
		for( int i=1;i<5;i++ ){
			node_force_gpu(root,4*global_id+i,x,y,mass,fx,fy,v);
		}
	}
}

