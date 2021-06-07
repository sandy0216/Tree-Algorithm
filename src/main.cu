#include <cstdio>
#include <cstring>
#include <vector>
#include "../inc/init.h"
#include "../inc/def_node.h"
#include "../inc/create_tree.h"
#include "../inc/force.h"
#include "../inc/print_tree.h"

using namespace std;


int main( int argc, char* argv[] )
{
	double *x, *y, *mass;
	double *vx, *vy;
	double *fx,*fy;
	double boxsize = 10.0;
	double maxmass = 100.0;
	double theta = 0.8;

	double dt = 0.01;
	double endtime = 1;

	int n=50;

	x = (double *)malloc(n*sizeof(double));
	y = (double *)malloc(n*sizeof(double));
	mass = (double *)malloc(n*sizeof(double));
	vx = (double *)malloc(n*sizeof(double));
	vy = (double *)malloc(n*sizeof(double));
	fx = (double *)malloc(n*sizeof(double));
	fy = (double *)malloc(n*sizeof(double));

	//==================initial conditions========================
	// Create initial conditions
	init(x, y, mass, n, boxsize, maxmass);
	for( int i=0;i<n;i++ ){
		vx[i]=vy[i]=0.0;
	}
	
	// Record the initial conditions
	FILE *initfile;
	initfile = fopen("./input/init.dat","w");
	fprintf(initfile, "index\tx\ty\tmass\n");
	for( int i=0;i<n;i++ ){
		fprintf(initfile, "%d\t%.3f\t%.3f\t%.3f\n",i,x[i],y[i],mass[i]);
	}
	fclose(initfile);
	// End of creating intial conditions
	
	//=================Evolution===============================
	double t=0.0;
	int step=0;
	
	char preffix[15] = "./output/snap_";
	char number[10];
	char suffix[5] = ".dat";
	//char *filename=NULL;
	
	while( t<endtime ){
		printf("[Step %d] T=%.3f\n",step,t);

		// Create tree
		NODE *head = new NODE();
		create_tree(head, x, y, mass, boxsize,n);
		printf("End creating tree...\n");

		// Calculate force for each particles
		force(head, x, y, mass, fx, fy, theta,n);
		printf("Finish calculating force...\n");
		
		// Iterate steps
		for( int i=0;i<n;i++ ){
			x[i] += vx[i]*dt;
			y[i] += vy[i]*dt;
			vx[i] += fx[i]*dt;
			vy[i] += fy[i]*dt;
			if( x[i]>boxsize ){
				x[i] -= boxsize;
			}else if( x[i]<0 ){
				x[i] += boxsize;
			}
			if( y[i]>boxsize ){
				y[i] -= boxsize;
			}else if( y[i]<0 ){
				y[i] += boxsize;
			}
		}
		printf("Finish moving particles...\n");
	
		/*
		//Output snapshots
		sprintf(number,"%d",step);
		strcat( preffix, number);
		printf("%s\n",preffix);
		//filename = str_contact(preffix,number);
		//filename = str_contact(preffix,suffix);
		printf("here");
		FILE *outfile;
		outfile = fopen(filename,"w");
		fprintf(initfile, "index\tx\ty\n");
		for( int i=0;i<n;i++ ){
			fprintf(outfile, "%d\t%.3f\t%.3f\n",i,x[i],y[i]);
		}
		printf("Finish output positions...\n");
		fclose(outfile);
*/
		// Move to next step
		t = t+dt;
		step = step+1;
		free(head);
	}
	

	return 0;
}
