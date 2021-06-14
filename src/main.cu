#include <cstdio>
#include <cstring>
#include <vector>
#include "../inc/init.h"
#include "../inc/def_node.h"
#include "../inc/create_tree.h"
#include "../inc/create_tree_gpu.h"
#include "../inc/force.h"
#include "../inc/print_tree.h"
#include "../inc/tool_main.h"
#include "../inc/param.h"

using namespace std;


int main( int argc, char* argv[] )
{
	double *x, *y, *mass;
	double *d_x, *d_y, *d_mass;
	double *vx, *vy;
	double *fx,*fy;
	double *V;
	double E,Ek;
	double region = 80.0;  // restrict position of the initial particles
	double maxmass = 100.0;

	int    n  = initial_n;
	int    *d_index;
	int    *d_num;

	double endtime = dt*1;

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
	printf("Finsih creating initial condition...\n");
	
	// Record the initial conditions
	FILE *initfile;
	initfile = fopen("./input/init.dat","w");
	fprintf(initfile, "index\tx\ty\tmass\n");
	for( int i=0;i<n;i++ ){
		fprintf(initfile, "%d\t%.3f\t%.3f\t%.3f\n",i,x[i],y[i],mass[i]);
	}
	fclose(initfile);
	// End of creating intial conditions

	//==================GPU settings==============================
	/*int gid = 0;
	if( cudaSetDevice(gid) != cudaSuccess ){
		printf("!!! Cannot select GPU \n");
		exit(1);
	}
	cudaSetDevice(gid);

	int tx,ty; // Number of threads per block
	tx = 4;
	ty = 4;
	if( tx*ty>1024 ){
		printf("Number of threads per block must < 1024!!\n");
		exit(0);
	}
	dim3 threads(tx,ty);
	int bx,by; // Number of blocks per grid
	bx = 4;
	by = 4;
	if( bx>65535 || by>65535 ){
		printf("The grid size exceeds the limit!\n");
		exit(0);
	}
	dim3 blocks(bx,by);

	int n_thread = tx*ty;
	int n_block  = bx*by;
	int n_work   = n_thread*n_block;
	
	cudaMalloc((void**)&d_x, n*sizeof(double));
	cudaMalloc((void**)&d_y, n*sizeof(double));
	cudaMalloc((void**)&d_mass, n*sizeof(double));
	cudaMalloc((void**)&d_index, n*sizeof(int));
	
	
	cudaMemcpy( d_x, x, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy( d_y, y, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy( d_mass, mass, n*sizeof(double), cudaMemcpyHostToDevice);

	split<<<blocks,threads>>>(d_x,d_y,d_mass);	
	
	
	
	
	printf("well done\n");*/






	
	//=================Evolution===============================
	double t=0.0;
	int step=0;
	int file=0;
	
	char preffix[15] = "./output/snap_";
	char number[5];
	char suffix[5] = ".dat";
	int length;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float t_tree, t_force, t_update, t_estimate;
	
	while( t<endtime ){
		printf("[Step %d] T=%.3e\n",step,t);
		cudaEventRecord(start,0);
		// Create tree
		NODE *head = new NODE();
		create_tree(head, x, y, mass,n);
		//printf("End creating tree...\n");
		if( step == 0 ){
			potential(head,x,y,mass,V,n);
			E = 0;
			for( int i=0;i<n;i++ ){
				E += V[i];
			}
			printf("Initial energy:%.3e\n",E);
		}
		cudaEventRecord(stop,0);
		cudaEventElapsedTime(&t_tree, start, stop);
		printf("End creating tree, time=%.5f(ms)\n",t_tree);


		// Calculate force for each particles
		cudaEventRecord(start,0);
		force(head, x, y, mass, fx, fy,n);
		//printf("Finish calculating force...\n");
		cudaEventRecord(stop,0);
		cudaEventElapsedTime(&t_force,start,stop);
		printf("End calculating force, time=%.5f(ms)\n",t_force);


		cudaEventRecord(start,0);
		update(x,y,vx,vy,n);
		update(vx,vy,fx,fy,n);
		check_boundary(x,y,mass,&n);
		cudaEventRecord(stop,0);
		cudaEventElapsedTime(&t_update,start,stop);
		printf("End updating particle, time=%.5f(ms)\n",t_update);


		//check_boundary(x,y,mass,&n);

		// Verification
		//if( step%100==0 ){
			cudaEventRecord(start,0);
			potential(head,x,y,mass,V,n);
			//printf("[Step %d] T=%.3e\n",step,t);
			E = 0;
			Ek = 0;
			for( int i=0;i<n;i++ ){
				E += V[i]+0.5*mass[i]*(pow(vx[i],2)+pow(vy[i],2));
				Ek += 0.5*mass[i]*(pow(vx[i],2)+pow(vy[i],2));
			}
			cudaEventRecord(stop,0);
			cudaEventElapsedTime(&t_estimate,start,stop);
			printf("Particle number remains:%d\n",n);
			printf("Energy conservation:%.3e\n",E);
			printf("Kinetic energy:%.3e\n",Ek);
			printf("End estimate energy, time=%.3f(ms)\n",t_estimate);

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
