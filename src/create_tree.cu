#include <cstdio>
#include <cstring>
#include "../inc/def_node.h"
#include "../inc/print_tree.h"
#include "../inc/tool_tree.h"
#include "../inc/param.h"

//#define DEBUG
void add_particle(NODE *head, double x, double y, double mass)
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
			reg = region(x,y,cx,cy);
			if(current->next[reg] == NULL){
				cxs[0]=cxs[3]=cx+side/4;
				cxs[1]=cxs[2]=cx-side/4;
				cys[0]=cys[1]=cy+side/4;
				cys[2]=cys[3]=cy-side/4;
				temp = new NODE();
				current->next[reg] = temp;
				create_node(temp,cxs[reg],cys[reg],x,y,mass,side/2);
				break;
			}else{
				current = current->next[reg];
			}
		}else{	//reach the smallest grid which contains only one particle
#ifdef DEBUG
			printf("Find the finest grid\n");
			printf("New particle position:%.3f,%.3f\n",x,y);
			printf("Old particle position:%.3f,%.3f\n",cmx,cmy);
#endif
			finest_grid(current,x,y,mass);	
			break;
		}//End of if-else while reaching the smallest grid
		
	}
}


void create_tree(NODE *head, double *x, double *y, double *mass, int n)
{
	create_node(head,boxsize/2,boxsize/2,x[0],y[0],mass[0],boxsize);
	//print_tree(head);
	//printf("=====Add particle=====\n");
	for( int i=1;i<n;i++ ){
		add_particle(head,x[i],y[i],mass[i]);
		//printf("add %d\n",i);
	}
	//add_particle(head,x[2],y[2],mass[2]);
	NODE* current=head->next[0];
	if( current == NULL ){
		printf("[tree]creation failed\n");
	}
	//print_tree(head);
	//add_particle(head, x[1], y[1], mass[0]);

}
