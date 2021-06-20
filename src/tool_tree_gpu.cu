#include "../inc/def_node.h"

__device__ void create_node_gpu(NODE *head, double cx, double cy, double x, double y, double mass, double side)
{
	head->center[0] = cx;
	head->center[1] = cy;
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
	return 5;
}

__device__ void finest_grid_gpup(NODE *current, double x, double y, double mass,int *flag);

__device__ void finest_grid_gpu(NODE *current, double x, double y, double mass,int *flag)
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
		/*if( sqrt(pow(cx-x,2)+pow(cy-y,2))<1e-7 ){
			printf("Two particles are too close!!!\n");
			printf("Old particle is at %.3f, %.3f\n",cmx,cmy);
			printf("New particle is at %.3f, %.3f\n",x,y);
		}*/

		//information for the sub-grids
		cxs[0]=cxs[3]=cx+side/4;
		cxs[1]=cxs[2]=cx-side/4;
		cys[0]=cys[1]=cy+side/4;
		cys[2]=cys[3]=cy-side/4;
		
		//split a block for the current particle
		reg = region_gpu(cmx,cmy,cx,cy);
		//if( current->next[reg]!=NULL ){ printf("Already exist subgrid, exist bug!!!\n"); }
		if( current->next[reg] == NULL ){	
			nextnode = new NODE();
			current->next[reg] = nextnode;
			create_node_gpu(current->next[reg],cxs[reg],cys[reg],cmx,cmy,m,side/2);
			*flag += 1;
		}
			
		//update information of current grid
		current->leaf = 1;
		current->num = current->num + 1;
		current->centerofmass[0] = (cmx*m+x*mass)/(m+mass);
		current->centerofmass[1] = (cmy*m+y*mass)/(m+mass);
		current->mass = m+mass;

		
		reg = region_gpu(x,y,cx,cy);
		if( current->next[reg]!=NULL ){
			//printf("oh no\n");
			finest_grid_gpup(current->next[reg],x,y,mass,flag);
		}else{
			nextnode = new NODE();
			current->next[reg] = nextnode;
			create_node_gpu(current->next[reg],cxs[reg],cys[reg],x,y,mass,side/2);
			*flag += 1;
//			if( current == NULL ){ printf("Creation of subnode failed.\n"); }
		}
	}//end of if
}//end of function


__device__ void finest_grid_gpup(NODE *current, double x, double y, double mass,int *flag)
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
		/*if( sqrt(pow(cx-x,2)+pow(cy-y,2))<1e-7 ){
			printf("Two particles are too close!!!\n");
			printf("Old particle is at %.3f, %.3f\n",cmx,cmy);
			printf("New particle is at %.3f, %.3f\n",x,y);
		}*/

		//information for the sub-grids
		cxs[0]=cxs[3]=cx+side/4;
		cxs[1]=cxs[2]=cx-side/4;
		cys[0]=cys[1]=cy+side/4;
		cys[2]=cys[3]=cy-side/4;
		
		//split a block for the current particle
		reg = region_gpu(cmx,cmy,cx,cy);
		//if( current->next[reg]!=NULL ){ printf("Already exist subgrid, exist bug!!!\n"); }
		if( current->next[reg] == NULL ){	
			nextnode = new NODE();
			current->next[reg] = nextnode;
			create_node_gpu(current->next[reg],cxs[reg],cys[reg],cmx,cmy,m,side/2);
			*flag += 1;
		}
			
		//update information of current grid
		current->leaf = 1;
		current->num = current->num + 1;
		current->centerofmass[0] = (cmx*m+x*mass)/(m+mass);
		current->centerofmass[1] = (cmy*m+y*mass)/(m+mass);
		current->mass = m+mass;

		
		reg = region_gpu(x,y,cx,cy);
		if( current->next[reg]!=NULL ){
			//printf("oh no\n");
			finest_grid_gpu(current->next[reg],x,y,mass,flag);
		}else{
			nextnode = new NODE();
			current->next[reg] = nextnode;
			create_node_gpu(current->next[reg],cxs[reg],cys[reg],x,y,mass,side/2);
			*flag += 1;
//			if( current == NULL ){ printf("Creation of subnode failed.\n"); }
		}
	}//end of if
}//end of function




__device__ void add_particle_gpu(NODE *head, double x, double y, double mass,int *flag)
{
	NODE *current = head;
	NODE *temp;
	int reg;
	double cx,cy,cmx,cmy,m,side;
	double cxs[4],cys[4];
	while(true){
		//input parameters
		cx = current->center[0];
		cy = current->center[1];
		cmx = current->centerofmass[0];
		cmy = current->centerofmass[1];
		side = current->side;
		m = current->mass;
		
		if( current->num != 1){ //not the smallest grid
			//update the information
			current->num = current->num+1;
			current->centerofmass[0] = (cmx*m+x*mass)/(m+mass);
			current->centerofmass[1] = (cmy*m+y*mass)/(m+mass);
			current->mass = m+mass;

			//move to next grid
			reg = region_gpu(x,y,cx,cy);
			if(current->next[reg] == NULL){
				cxs[0]=cxs[3]=cx+side/4;
				cxs[1]=cxs[2]=cx-side/4;
				cys[0]=cys[1]=cy+side/4;
				cys[2]=cys[3]=cy-side/4;
				temp = new NODE();
				current->next[reg] = temp;
				create_node_gpu(temp,cxs[reg],cys[reg],x,y,mass,side/2);
				*flag += 1;
				break;
			}else{
				current = current->next[reg];
			}
		}else{	//reach the smallest grid which contains only one particle
			finest_grid_gpu(current,x,y,mass,flag);	
			break;
		}//End of if-else while reaching the smallest grid
		
	}
}

	
/*__device__ void create_tree_gpu(NODE *head, vector<double> *x, vector<double> *y, vector<double> *mass, int n, double boxsize)
{
	create_node_gpu(head,boxsize/2,boxsize/2,x[0],y[0],mass[0],boxsize);
	for( int i=1;i<n;i++ ){
		add_particle(head,x[i],y[i],mass[i]);
	}
	NODE* current=head->next[0];
	if( current == NULL ){
		//printf("[tree]creation failed\n");
	}
}*/
	

	
