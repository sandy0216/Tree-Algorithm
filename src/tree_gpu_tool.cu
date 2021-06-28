#include "../inc/def_node.h"

__device__ void finest_grid_gpu(TNODE *head,double cx,double cy,double x,double y,double mass);

__device__ void create_node_gpu(TNODE *head,double cmx,double cmy,double mass,double side)
{
	head->centerofmass[0] = cmx;
	head->centerofmass[1] = cmy;
	head->mass = mass;
	head->side = side;
	head->num = 1;
	for( int i=0;i<4;i++ ){
		head->next[i] = NULL;
	}
}

__device__ int region_gpu(double x, double y, double cx, double cy){
	if( x>=cx && y>=cy ){
		return 0;
	}else if( x<cx && y>=cy ){
		return 1;
	}else if( x<cx && y<cy ){
		return 2;
	}else if( x>=cx && y<cy ){
		return 3;
	}
	//printf("Error in deciding region!!!");
	return 5;
}

__device__ void add_particle_gpu(TNODE *head,double cx,double cy,double x,double y,double mass)
{
	TNODE *temp;
	int reg;
	double cmx,cmy,m,side;
	double cxs[4],cys[4];
	while(true){
		cmx = head->centerofmass[0];
		cmy = head->centerofmass[1];
		side = head->side;
		m = head->mass;
		cxs[0]=cxs[3]=cx+side/4;
		cxs[1]=cxs[2]=cx-side/4;
		cys[0]=cys[1]=cy+side/4;
		cys[2]=cys[3]=cy-side/4;
		if( head->num != 1 ){
			head->num = head->num+1;
			head->centerofmass[0] = (cmx*m+x*mass)/(m+mass);
			head->centerofmass[1] = (cmy*m+y*mass)/(m+mass);
			head->mass = m+mass;

			reg = region_gpu(x,y,cx,cy);
			if( head->next[reg] == NULL ){
				temp = new TNODE();
				head->next[reg] = temp;
				create_node_gpu(temp,x,y,mass,side/2);
				break;
			}else{
				cx = cxs[reg];
				cy = cys[reg];
				head = head->next[reg];
			}
		}else{
			finest_grid_gpu(head,cx,cy,x,y,mass);
			break;
		}
	}
}

__device__ void finest_grid_gpu(TNODE *head,double cx,double cy,double x,double y,double mass)
{
	double cxs[4],cys[4];
	double cmx,cmy,side,m;
	int reg1,reg2;
	TNODE *nextnode;
	while( head->num == 1 ){
		// input parameters
		cmx = head->centerofmass[0];
		cmy = head->centerofmass[1];
		side = head->side;
		m = head->mass;
		
		// information for the sub-grids
		cxs[0]=cxs[3]=cx+side/4;
		cxs[1]=cxs[2]=cx-side/4;
		cys[0]=cys[1]=cy+side/4;
		cys[2]=cys[3]=cy-side/4;
		
		reg1 = region_gpu(cmx,cmy,cx,cy);
		reg2 = region_gpu(x,y,cx,cy);
		head->num = head->num + 1;
		head->centerofmass[0] = (cmx*m+x*mass)/(m+mass);
		head->centerofmass[1] = (cmy*m+y*mass)/(m+mass);
		head->mass = m+mass;
		if( reg1 != reg2 ){
			nextnode = new TNODE();
			create_node_gpu(nextnode,cmx,cmy,m,side/2);
			head->next[reg1] = nextnode;
			nextnode = new TNODE();
			create_node_gpu(nextnode,x,y,mass,side/2);
			head->next[reg2] = nextnode;
			break;
		}else{
			cx = cxs[reg1];
			cy = cys[reg1];
			nextnode = new TNODE();
			cmx = head->centerofmass[0];
			cmy = head->centerofmass[1];
			m = head->mass;
			create_node_gpu(nextnode,cmx,cmy,m,side/2);
			nextnode->num = 2;
			head->next[reg1] = nextnode;
			head = head->next[reg1];
		}
	}
}




		

