#include <cstdio>
#include <vector>
#include "../inc/init.h"

using namespace std;

struct NODE{
	double center[2];
	double centerofmass[2];
	double totalmass;
	double side;
	NODE *next1;
	NODE *next2;
	NODE *next3;
	NODE *next4;
};

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
	head->center[0] = boxsize/2;
	head->center[1] = boxsize/2;
	head->next1 = NULL;
	head->next2 = NULL;
	head->next3 = NULL;
	head->next4 = NULL;

	NODE *current = head;
	/*for( int i=1;i<n;i++ ){
		current->next = new NODE();
		current = current->next;
		current->mass = mass[i];
		current->next = NULL;
	}*/

	printf("Test:\n");
	current = head;
	while( current != NULL ){
		printf("%.3f\t%.3f\n",current->center[0],current->center[1]);
		current = current->next1;
	}
	
	


	return 0;
}
