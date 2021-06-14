#include <cstdio>
#include <cstring>
#include "../inc/def_node.h"
#include "../inc/print_tree.h"
#include "../inc/param.h"

__device__ void create_node(NODE *head,double cx, double cy, double x, double y, double mass, double side)
{
	head->center[0]       = cx;
	head->center[1]       = cy;
	head->centerofmass[0] = x;
	head->centerofmass[1] = y;
	head->mass = mass;
	head->side = side;
	head->num  = 1;
	head->leaf = 0;
	for( int i=0;i<4;i++ ){
		head->next[i] = NULL;
	}
}

__device__ int region(double x, double y, double cx, double cy){
	if( x>=cx && y>=cy ){
		return 0;
	}else if( x<cx && y>=cy ){
		return 1;
	}else if( x<cx && y<cy ){
		return 2;
	}else if( x>=cx && y<cy ){
		return 3;
	}
	return 5;
}

__device__ void finest_grid(NODE *current, double x, double y, double mass)
{
	double cxs[4],cys[4];
	double cx,cy,cmx,cmy,side,m;
	int reg;
	NODE* nextnode;
	if( current->leaf==0 ){
	//while(true){
		//input parameters
		cx = current->center[0];
		cy = current->center[1];
		side = current->side;
		cmx = current->centerofmass[0];
		cmy = current->centerofmass[1];
		m = current->mass;

		//Check if the two particle is too close
		if( sqrt(pow(cx-x,2)+pow(cy-y,2))<1e-7 ){
			printf("Two particles are too close!!!\n");
			printf("Old particle is at %.3f, %.3f\n",cmx,cmy);
			printf("New particle is at %.3f, %.3f\n",x,y);
		}

		//information for the sub-grids
		cxs[0]=cxs[3]=cx+side/4;
		cxs[1]=cxs[2]=cx-side/4;
		cys[0]=cys[1]=cy+side/4;
		cys[2]=cys[3]=cy-side/4;
	
		//split a block for the current particle
		reg = region(cmx,cmy,cx,cy);
		if( current->next[reg]!=NULL ){ printf("Already exist subgrid, exist bug!!!\n"); }
		else{
			nextnode = new NODE();
			current->next[reg] = nextnode;
#ifdef DEBUG
			printf("[Parent]Create node at Quad %d\n",reg);
			printf("[Parent]Grid cneter %.3f, %.3f\n",cx,cy);
			printf("[Parent]Particle position %.3f, %.3f\n",cmx,cmy);
#endif
			create_node(current->next[reg],cxs[reg],cys[reg],cmx,cmy,m,side/2);
		}
		
		//update information of current grid
		current->leaf = 1;
		current->num = current->num + 1;
		current->centerofmass[0] = (cmx*m+x*mass)/(m+mass);
		current->centerofmass[1] = (cmy*m+y*mass)/(m+mass);
		current->mass = m+mass;

		
		reg = region(x,y,cx,cy);
		if( current->next[reg]!=NULL ){
			//printf("oh no\n");
			finest_grid(current->next[reg],x,y,mass);
		}else{
			nextnode = new NODE();
			current->next[reg] = nextnode;
#ifdef DEBUG
			printf("[Children]Create node at Quad %d\n",reg);
			printf("[Children]Grid cneter %.3f, %.3f\n",cx,cy);
			printf("[Children]Particle position %.3f, %.3f\n",x,y);
#endif 
			create_node(current->next[reg],cxs[reg],cys[reg],x,y,mass,side/2);
			if( current == NULL ){ printf("Creation of subnode failed.\n"); }
		}
	}//end of if
}//end of function

	

	

	
