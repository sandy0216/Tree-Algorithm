#include <cstdio>
#include <cstring>
#include "../inc/def_node.h"
#include "../inc/print_tree.h"
#include "../inc/tool_tree.h"

void add_particle(NODE *head, double x, double y, double mass)
{
	NODE *current = head;
	int reg;
	double cx,cy,cmx,cmy,m;
	while(true){
		//input parameters
		cx = current->center[0];
		cy = current->center[1];
		cmx = current->centerofmass[0];
		cmy = current->centerofmass[1];
		m = current->mass;
		
		if( current->num != 1){ //not the smallest grid
			//update the information
			current->num = current->num+1;
			current->centerofmass[0] = (cmx*m+x*mass)/(m+mass);
			current->centerofmass[1] = (cmy*m+y*mass)/(m+mass);
			current->mass = m+mass;

			//move to next grid
			reg = region(x,y,cx,cy);
			current = current->next[reg];
			if(current == NULL){  printf("No subgrid, exist bug!!!\n");  }
		}else{	//reach the smallest grid which contains only one particle
			reg = finest_grid(current,x,y,mass);	
		}//End of if-else while reaching the smallest grid
		
	}
}


void create_tree(NODE *head, double *x, double *y, double *mass, const double boxsize)
{
	create_node(head,boxsize/2,boxsize/2,x[0],y[0],mass[0],boxsize);
	print_tree(head);
	printf("=====Add particle=====\n");
	add_particle(head,x[1],y[1],mass[1]);
	//add_particle(head,x[2],y[2],mass[2]);
	NODE* current=head->next[0];
	if( current == NULL ){
		printf("creation failed\n");
	}
	print_tree(head);
	//add_particle(head, x[1], y[1], mass[0]);

}
