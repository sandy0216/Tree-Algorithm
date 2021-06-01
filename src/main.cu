#include <cstdio>
#include <vector>
#include "../inc/init.h"
#include "../inc/def_node.h"
#include "../inc/create_tree.h"

using namespace std;


int main( int argc, char* argv[] )
{
	double *x, *y, *mass;
	double boxsize = 10.0;
	double maxmass = 100.0;

	int n=10;

	x = (double *)malloc(n*sizeof(double));
	y = (double *)malloc(n*sizeof(double));
	mass = (double *)malloc(n*sizeof(double));

	// Create initial conditions
	init(x, y, mass, n, boxsize, maxmass);
	FILE *initfile;
	initfile = fopen("./input/init.dat","w");
	fprintf(initfile, "index\tx\ty\tmass\n");
	for( int i=0;i<n;i++ ){
		fprintf(initfile, "%d\t%.3f\t%.3f\t%.3f\n",i,x[i],y[i],mass[i]);
	}
	// End of creating intial conditions
	
	NODE *head = new NODE();
	create_tree(head, x, y, mass, boxsize);


	//NODE *current = head;
	/*for( int i=1;i<n;i++ ){
		current->next = new NODE();
		current = current->next;
		current->mass = mass[i];
		current->next = NULL;
	}*/

	/*printf("Test:\n");
	current = head;
	while( current != NULL ){
		printf("%.3f\t%.3f\n",current->center[0],current->center[1]);
		for( int i=0;i<4;i++ ){
		current = head->next[i];
		if( current != NULL ){
		printf("%.3f\t%.3f\n",current->center[0],current->center[1]);
		}
		}
		current = current->next[0];	
	}*/
	
	


	return 0;
}
