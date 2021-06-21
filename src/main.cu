#include <cstdio>
#include <cstring>
#include <vector>
#include <omp.h>
#include "../inc/init.h"
#include "../inc/def_node.h"
#include "../inc/create_tree.h"
#include "../inc/tool_gpu.h"
#include "../inc/force.h"
#include "../inc/print_tree.h"
#include "../inc/tool_main.h"
#include "../inc/param.h"
#include "../inc/heap.h"
#include "../inc/cuapi.h"
#include "../inc/merge_tree_gpu.h"

using namespace std;

__device__ double d_boxsize;
__device__ double d_theta;
__device__ double d_eplison;
__device__ double d_dt;
__device__ int d_n_work;
__device__ int d_n_thread;
__device__ int d_side;// =nx=ny
__device__ int share_node;
__device__ int global_node;



int main( int argc, char* argv[] )
{
	double *x, *y, *mass;
	double *vx, *vy;
	double *fx,*fy;
	double *V;
	double region = 100.0;  // restrict position of the initial particles
	double maxmass = 100.0;

	unsigned long    n  = initial_n;

	double endtime = dt*1;

	float testtime;
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

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
	/*FILE *initfile;
	initfile = fopen("./input/init.dat","w");
	fprintf(initfile, "index\tx\ty\tmass\n");
	for( int i=0;i<n;i++ ){
		fprintf(initfile, "%d\t%.3f\t%.3f\t%.3f\n",i,x[i],y[i],mass[i]);
	}
	fclose(initfile);*/
	// End of creating intial conditions
	
	//================================================================================//
	//		The following code is for GPU parallelization			  //
	//================================================================================//

	
	//====================GPU blocks & gird settings====================
	int gid = 0;
	if( cudaSetDevice(gid) != cudaSuccess ){
		printf("!!! Cannot select GPU \n");
		exit(1);
	}
	cudaSetDevice(gid);

	if( tx*ty>1024 ){
		printf("Number of threads per block must < 1024!!\n");
		exit(0);
	}
	dim3 threads(tx,ty);
	if( bx>65535 || by>65535 ){
		printf("The grid size exceeds the limit!\n");
		exit(0);
	}
	dim3 blocks(bx,by);
	//====================End of blocks & grid settings==================

	//====================Set basic parameters of GPU====================
	// Set global parameter
	cudaMemcpyToSymbol( d_boxsize, &boxsize, sizeof(double));
	cudaMemcpyToSymbol( d_side, &nx, sizeof(int));
	cudaMemcpyToSymbol( d_n_work, &n_work, sizeof(int));
	cudaMemcpyToSymbol( d_n_thread, &n_thread, sizeof(int));
	
	// Deliver the information of each particle to GPU
	int   *d_n;
	cudaMalloc((void**)&d_n, sizeof(int));
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	double *d_x,*d_y,*d_mass,*d_vx,*d_vy;
	cudaMalloc((void**)&d_x, n*sizeof(double));
	cudaMalloc((void**)&d_y, n*sizeof(double));
	cudaMalloc((void**)&d_mass, n*sizeof(double));
	cudaMalloc((void**)&d_vx, n*sizeof(double));
	cudaMalloc((void**)&d_vy, n*sizeof(double));
	cudaMemcpy(d_vx, vx, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vy, vy, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, n*sizeof(double), cudaMemcpyHostToDevice);
	// Allocate memory for force
	double *d_fx,*d_fy,*d_V,*d_Ek;
	cudaMalloc((void**)&d_fx,n*sizeof(double));
	cudaMalloc((void**)&d_fy,n*sizeof(double));
	cudaMalloc((void**)&d_V,n*sizeof(double));
	cudaMalloc((void**)&d_Ek,n*sizeof(double));
	//====================End of setting basic parameters=====================

	//====================Split the particle into different subregion=====================
	// Record the region index of each particle
	int *index, *d_index;   
	index = (int *)malloc(n*sizeof(int));
	cudaMalloc((void**)&d_index, n*sizeof(int));	
	
	// Record number of particle in each region
	unsigned int *regnum, *d_regnum;      
	regnum   = (unsigned int *)malloc(n_work*sizeof(unsigned int));
	cudaMalloc((void**)&d_regnum, n_work*sizeof(unsigned int));
	for( int i=0;i<n_work;i++ ){ regnum[i]=0; }
	cudaMemcpy(d_regnum, regnum, n_work*sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	// Call kernel function :
	// Input  : parameters, postion fo the particles
	// Output : region index of each particles, number of particles in each region
	cudaEventRecord(start,0);
	split<<<threads,blocks>>>(d_x,d_y,d_index,d_regnum,d_n);
	cudaMemcpy(regnum,d_regnum,n_work*sizeof(unsigned int),cudaMemcpyDeviceToHost);
	cudaMemcpy(index,d_index,n*sizeof(int),cudaMemcpyDeviceToHost);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("Split&copy out the result takes %.3e(ms)\n",testtime);
	//===================End of splitting particles into different subregion============

	//=============================Do Load Balance=================================
	// Define region index
	int *region_index,*d_region_index,*thread_load,*d_thread_load;
	region_index = (int *)malloc(n_work*sizeof(int));	// Order of the region
	thread_load = (int *)malloc(n_thread*sizeof(int));  	// How many region for each thread
	cudaMalloc((void**)&d_region_index,n_work*sizeof(int));
	cudaMalloc((void**)&d_thread_load,n_thread*sizeof(int));
	int which_region=0;
	int which_thread=0;
	int reg_id;
	int current_thread_load = 0;
	
	// Calculate the Morton ordering of the sub-regions
	block(nx,0,0,region_index,&which_region);
	
	// Calculate load for each thread
	for( int i=0;i<n_thread;i++ ){ thread_load[i]=0; }
	for( int i=0;i<n_work;i++ ){
		reg_id = region_index[i];
		current_thread_load += regnum[reg_id];
		thread_load[which_thread] += 1;
		if( current_thread_load>n/n_thread-regnum[reg_id]/2 && which_thread<n_thread-1 ){
			printf("thread %d,load %d,take %d region\n",which_thread,current_thread_load,thread_load[which_thread]);
			which_thread+=1;
			current_thread_load = 0;
		}
		if( current_thread_load>n/n_thread*2 ){
			printf("Load no balance!!!\n");
			exit(1);
		}
	}

	// Check to region is left and cumsum the load of each thread
	int check = 0;
	for( int i=1;i<n_thread;i++ ){
		check += thread_load[i];
		thread_load[i] += thread_load[i-1];
	}
	if( check+thread_load[0] != n_work ){
		printf("Error no load balance!!!\n");
		exit(1);
	}
	
	cudaMemcpy(d_region_index,region_index,n_work*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_thread_load,thread_load,n_thread*sizeof(int),cudaMemcpyHostToDevice);
	//=====================End fo load balance=======================

	//=====================Allocate each particles===================	
	// Cumsum of the # of particle in each region
	for( int i=1;i<n_work;i++ ){
		regnum[i] += regnum[i-1];
	}
	cudaMemcpy(d_regnum,regnum,n_work*sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	// Sort the particle by the region index
	int *particle_index,*d_particle_index;
	particle_index = (int *)malloc(n*sizeof(int));
	cudaMalloc((void**)&d_particle_index,n*sizeof(int));
	for( int i=0;i<n;i++ ){
		particle_index[i] = i;
	}
	HeapSort(index,particle_index,n);
	cudaMemcpy(d_particle_index,particle_index,n*sizeof(int),cudaMemcpyHostToDevice);
	//====================End of allocate each particles=====================

	//==================='Merge' particles in different subregion into a particle================
	double *d_flag;
	// Treat the particles in each region as a new particle
	double *rx, *ry, *rmass, *d_rx, *d_ry, *d_rmass;
	int *rn, *d_rn;
	rx = (double *)malloc(n_work*sizeof(double));
	ry = (double *)malloc(n_work*sizeof(double));
	rmass = (double *)malloc(n_work*sizeof(double));
	rn = (int *)malloc(n_work*sizeof(int));
	cudaMalloc((void**)&d_rx, n_work*sizeof(double));
	cudaMalloc((void**)&d_ry, n_work*sizeof(double));
	cudaMalloc((void**)&d_rmass, n_work*sizeof(double));
	cudaMalloc((void**)&d_rn, n_work*sizeof(int));

	// Call kernel function:
	// Input:origin information of each particles, particle index, load of each region, total number of particle, 
	//	region index, load of each thread
	// Output:information of output 'particles', and how many number they contain.
	cudaEventRecord(start,0);
	merge_bottom<<<threads,blocks>>>(d_x,d_y,d_mass,d_particle_index,d_regnum,d_n,d_region_index,d_thread_load,d_rx,d_ry,d_rmass,d_rn);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("First step of merge and copy out the result takes %.3f(ms)\n",testtime);
	//===================End of 'Merge' particles in different region=======================

	//==================='Merge' particles with shared and global memory====================
	// Calculate memory for shared and global memory
	int sh_node = (pow(nx/bx,2)-1)*4/3+1;
	cudaMemcpyToSymbol( share_node, &sh_node, sizeof(int));
	int gl_node = (pow(bx,2)-1)*4/3+1;
	cudaMemcpyToSymbol( global_node, &gl_node, sizeof(int));
	printf("Allocate shared of %d GNODEs\n",sh_node);
	
	// Define global & shared memory
	GNODE *root;
	root = (GNODE *)malloc(gl_node*sizeof(GNODE));
	GNODE *d_root;
	cudaMalloc((void**)&d_root, gl_node*sizeof(GNODE));
	int sm = sh_node*sizeof(GNODE); 

	// test parameters
	double *flag;
	flag = (double *)malloc(gl_node*sizeof(double));
	cudaMalloc((void**)&d_flag,gl_node*sizeof(double));

	// Call kernel function:
	// Input:center of mass and mass for each subregion, region index
	// Output:information in the shared & global memory
	cudaEventRecord(start,0);
	merge_top<<<threads,blocks,sm>>>(d_x,d_y,d_mass,d_rx,d_ry,d_rmass,d_rn,d_region_index,d_thread_load,d_regnum,d_particle_index,d_fx,d_fy,d_V,d_root,d_flag);
	//cudaMemcpy(flag,d_flag,gl_node*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(root,d_root,gl_node*sizeof(GNODE), cudaMemcpyDeviceToHost);
	cudaMemcpy(fx,d_fx,n*sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(fy,d_fy,n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("Second step of merge and copy out the result takes %.3f(ms)\n",testtime);
	double ftest,r;

	for( int i=0;i<20;i++ ){
		ftest = 0;
		for( int j=gl_node-1;j>gl_node-bx*bx;j-- ){
			r = sqrt(pow(root[j].centerofmass[0]-x[i],2)+pow(root[j].centerofmass[1]-y[i],2));
			ftest += root[j].mass*mass[i]/pow(r,3)*(root[j].centerofmass[0]-x[i]);
		}
		printf("particle id = %d,fx=%.3f,fcpu=%.3f\n",i,fx[i],ftest);
	}
	//========================End of 'Merge' particle with shared and global memory==================

	//=====================Calculate force by n-body============================
	cudaEventRecord(start,0);
	n_body<<<threads,blocks>>>(d_x,d_y,d_mass,d_fx,d_fy,d_V,d_particle_index,d_regnum,d_n,d_region_index,d_thread_load);
	cudaMemcpy(fx,d_fx,n*sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(fy,d_fy,n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("Evaluate force for every subregion takes %.3f(ms)\n",testtime);
	for( int i=0;i<20;i++ ){
		printf("particle id = %d,fx=%.3f\n",i,fx[i]);
	}
	//=====================Calculate force by n-body=============================

	//=====================Update velocity,position==============================
	cudaEventRecord(start,0);
	int blocksize = 64;
	int gridsize = 64;
	sm = blocksize*sizeof(double);
	double *E,*d_E;
	E = (double *)malloc(blocksize*sizeof(double));
	cudaMalloc((void**)&d_E, blocksize*sizeof(double));
	update_gpu<<<threads,blocks>>>(d_x,d_y,d_mass,d_vx,d_vy,d_fx,d_fy,d_Ek,d_n);
	energy_gpu<<<blocksize,gridsize,sm>>>(d_Ek,d_V,d_n,d_E);
	cudaMemcpy(E,d_E,blocksize*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(x,d_x,n*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(y,d_y,n*sizeof(double),cudaMemcpyDeviceToHost);
	for( int i=1;i<blocksize;i++ ){
		E[0] += E[i];
	}
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("update position & velocity of particles takes %.3f(ms)\n",testtime);
	printf("Total energy = %.3e\n",E[0]);







	
	//=================Evolution===============================
	/*double t=0.0;
	int step=0;
	int file=0;
	
	char preffix[15] = "./output/snap_";
	char number[5];
	char suffix[5] = ".dat";
	int length;

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
	*/

	return 0;
}
