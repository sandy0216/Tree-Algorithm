#include <cstdio>
#include "../inc/def_node.h"

void print_node(NODE *current);

void print_tree(NODE *head)
{
	int layer = 0;
	int node_a = 0;
	NODE *parent = head;
	NODE *current;
	printf("=============root===============\n");
	print_node(parent);
	//while( parent->leaf != 0 ){
	for( int i=0;i<4;i++ ){
		print_node(parent->next[i]);
	}
	//layer++;
	//}
	print_node(parent->next[0]);
}

void print_node(NODE *current){
	if( current != NULL ){
	printf("=====================================\n");
	printf("Center of the box=(%.3f,%.3f)\n",current->center[0],current->center[1]);
	printf("Number of particles=%d\n",current->num);
	printf("Side length=%.3f\n",current->side);
	}
}



