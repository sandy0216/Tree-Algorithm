#include <cstdio>
#include <cstring>
#include "../inc/def_node.h"
#include "../inc/print_tree.h"

void create_node(NODE *head,double cx, double cy, double x, double y, double mass, double side)
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

int region(double x, double y, double cx, double cy){
	if( x>=cx && y>=cy ){
		return 0;
	}else if( x<cx && y>=cy ){
		return 1;
	}else if( x<cx && y<cy ){
		return 2;
	}else if( x>=cx && y<cy ){
		return 3;
	}
	printf("Error in deciding region!!!");
	return 5;
}

void add_particle(NODE *head, double x, double y, double mass)
{
	NODE *current = head;
	int reg;
	double cx,cy,side;
	double cxs[4],cys[4];
	while(true){
		cx = current->center[0];
		cy = current->center[1];
		side = current->side;
		cxs[0]=cxs[3]=cx+side/4;
		cxs[1]=cxs[2]=cx-side/4;
		cys[0]=cys[1]=cy+side/4;
		cys[2]=cys[3]=cy-side/4;
		reg = region(x,y,cx,cy);
		current = current->next[reg];
		if( current == NULL ){
			printf("Create node at Quad %d\n",reg);
			current = new NODE();
			if( current == NULL ){
				printf("Creation failed\n");
			}
			create_node(current,cxs[reg],cys[reg],x,y,mass,side/2);
			break;
		}
		
	}

}


void create_tree(NODE *head, double *x, double *y, double *mass, const double boxsize)
{
	create_node(head,boxsize/2,boxsize/2,x[0],y[0],mass[0],boxsize);
	print_tree(head);
	printf("=====Add particle=====\n");
	add_particle(head,x[1],y[1],mass[1]);
	NODE* current=head->next[0];
	if( current == NULL ){
		printf("creation failed\n");
	}


	print_tree(head);
	//add_particle(head, x[1], y[1], mass[0]);

}
