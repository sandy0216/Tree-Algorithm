#include <cstdio>
#include <cstring>
#include <vector>
#include "../inc/init.h"
#include "../inc/def_node.h"
#include "../inc/create_tree.h"
#include "../inc/force.h"
#include "../inc/print_tree.h"
#include "../inc/tool_main.h"

using namespace std;


int main( int argc, char* argv[] )
{
	double *x, *y, *mass;
	double *vx, *vy;
	double *fx,*fy;
	double *V;
	double E,Ek;
	double boxsize = 100.0;
	double region = 20.0;  // restrict position of the initial particles
	double maxmass = 100.0;
	double theta = 0.8;

	double dt = 1e-5;
	double endtime = 5e-5;

	int n=100;

	x = (double *)malloc(n*sizeof(double));
	y = (double *)malloc(n*sizeof(double));
	mass = (double *)malloc(n*sizeof(double));
	vx = (double *)malloc(n*sizeof(double));
	vy = (double *)malloc(n*sizeof(double));
	fx = (double *)malloc(n*sizeof(double));
	fy = (double *)malloc(n*sizeof(double));
	V  = (double *)malloc(n*sizeof(double));

	//==================initial conditions========================
	// Create initial conditions
	init(x, y, mass, n, region, boxsize, maxmass);
	for( int i=0;i<n;i++ ){
		vx[i]=vy[i]=0.0;
		//mass[i]=10;
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
	int file=0;
	
	char preffix[15] = "./output/snap_";
	char number[5];
	char suffix[5] = ".dat";
	int length;
	
	while( t<endtime ){
		// Create tree
		NODE *head = new NODE();
		create_tree(head, x, y, mass, boxsize,n);
		//printf("End creating tree...\n");

		// Calculate force for each particles
		force(head, x, y, mass, fx, fy, theta,n);
		//printf("Finish calculating force...\n");

		update(x,y,vx,vy,n,dt);
		update(vx,vy,fx,fy,n,dt);
		potential(head,x,y,mass,V,theta,n);

		// Verification
		//if( step%10==0 ){
			printf("[Step %d] T=%.3e\n",step,t);
			E = 0;
			Ek = 0;
			for( int i=0;i<n;i++ ){
				E += V[i]+0.5*mass[i]*(pow(vx[i],2)+pow(vy[i],2));
				Ek += 0.5*mass[i]*(pow(vx[i],2)+pow(vy[i],2));
			}
			printf("Particle number remains:%d\n",n);
			printf("Energy conservation:%.3e\n",E);
			printf("Kinetic energy:%.3e\n",Ek);
		//}
	
		if( step%1000==0 ){
			//Output snapshots
			sprintf(number,"%d",file);
			length = snprintf(NULL, 0, "%s%s%s",preffix,number,suffix);
			char concated[length+1];
			snprintf(concated,sizeof(concated),"%s%s%s",preffix,number,suffix);
			FILE *outfile;
			outfile = fopen(concated,"w");
			fprintf(outfile, "index\tx\ty\n");
			for( int i=0;i<n;i++ ){
				fprintf(outfile, "%d\t%.3f\t%.3f\n",i,x[i],y[i]);
			}
			printf("Record position ...\n");
			fclose(outfile);
			file += 1;
		}

		// Move to next step
		t = t+dt;
		step = step+1;
	}
	

	return 0;
}
