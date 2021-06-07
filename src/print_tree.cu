#include <cstdio>
#include "../inc/def_node.h"

void print_node(NODE *current);

void print_tree(NODE *head)
{
	NODE* current;
	if( head != NULL ){
		printf("====root====\n");
		print_node(head);
		for( int i=0;i<4;i++ ){
			current = head->next[i];
			if( current != NULL ){
				printf("===Node %d===\n",i);
				print_node(current);
				printf("Kids of this node\n");
				print_tree(current);	
			}
		}
	}else{
		//printf("No nodes inside\n");
	}
}

void print_node(NODE *current){
	if( current != NULL ){
	printf("Center of the box=(%.3f,%.3f)\n",current->center[0],current->center[1]);
	printf("Number of particles=%d\n",current->num);
	printf("Side length=%.3f\n",current->side);
	printf("Center of mass=%.3f, %.3f\n",current->centerofmass[0],current->centerofmass[1]);
	}else{
		printf("No node here\n");
	}
}



