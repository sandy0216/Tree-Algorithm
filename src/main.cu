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
#include "../inc/force_gpu.h"
#include "../inc/tree_gpu.h"

using namespace std;

__device__ double d_boxsize;
__device__ double d_theta;
__device__ double d_eplison;
__device__ double d_dt;
__device__ int d_n_work;
__device__ int d_bx;
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
	unsigned long	 gpu_memory = 0;

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
	V  = (double *)malloc(sizeof(double));

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
	cudaMemcpyToSymbol( d_bx, &bx, sizeof(int));
	cudaMemcpyToSymbol( d_theta, &theta, sizeof(double));
	cudaMemcpyToSymbol( d_eplison, &eplison, sizeof(double));
	cudaMemcpyToSymbol( d_dt,&dt, sizeof(double));
	gpu_memory += 4*sizeof(int)+3*sizeof(double);
	
	// Deliver the information of each particle to GPU
	int   *d_n;
	cudaMalloc((void**)&d_n, sizeof(int));
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	double *d_x,*d_y,*d_mass;
	cudaMalloc((void**)&d_x, n*sizeof(double));
	cudaMalloc((void**)&d_y, n*sizeof(double));
	cudaMalloc((void**)&d_mass, n*sizeof(double));
	cudaMemcpy(d_x, x, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, n*sizeof(double), cudaMemcpyHostToDevice);
	gpu_memory += sizeof(int)+3*n*sizeof(double);
	// Allocate memory for force
	double *d_fx,*d_fy,*d_V,*d_Ek;
	cudaMalloc((void**)&d_fx,n*sizeof(double));
	cudaMalloc((void**)&d_fy,n*sizeof(double));
	cudaMalloc((void**)&d_V,sizeof(double));
	cudaMalloc((void**)&d_Ek,sizeof(double));
	gpu_memory += sizeof(double)*(2*n+2);
	//====================End of setting basic parameters=====================

	//====================Split the particle into different subregion=====================
	// Record the region index of each particle
	int *index, *d_index;   
	index = (int *)malloc(n*sizeof(int));
	cudaMalloc((void**)&d_index, n*sizeof(int));
	gpu_memory += sizeof(int)*n;	
	
	// Record number of particle in each region
	/*unsigned int *regnum, *d_regnum;      
	regnum   = (unsigned int *)malloc(n_work*sizeof(unsigned int));
	cudaMalloc((void**)&d_regnum, n_work*sizeof(unsigned int));
	for( int i=0;i<n_work;i++ ){ regnum[i]=0; }
	cudaMemcpy(d_regnum, regnum, n_work*sizeof(unsigned int), cudaMemcpyHostToDevice);*/
	
	// Call kernel function :
	// Input  : parameters, postion fo the particles
	// Output : region index of each particles, number of particles in each region
	cudaEventRecord(start,0);
	split<<<blocks,threads>>>(d_x,d_y,d_index,d_n);
	//cudaMemcpy(regnum,d_regnum,n_work*sizeof(unsigned int),cudaMemcpyDeviceToHost);
	cudaMemcpy(index,d_index,n*sizeof(int),cudaMemcpyDeviceToHost);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("Split&copy out the result takes %.3e(ms)\n",testtime);
	gpu_memory -= n*sizeof(int);
	//===================End of splitting particles into different subregion============
	

	//==================='Merge' particles in different subregion into a particle================
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
	gpu_memory+=(n_work*(3*sizeof(double)+sizeof(int)));
	for( int i=0;i<n_work;i++ ){
		rx[i] = 0;
		ry[i] = 0;
		rmass[i] = 0;
		rn[i] = 0;
	}
	cudaMemcpy(d_rx,rx,n_work*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_ry,ry,n_work*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_rmass,rmass,n_work*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_rn,rn,n_work*sizeof(int),cudaMemcpyHostToDevice);

	// Call kernel function:
	// Input:origin information of each particles, particle index, load of each region, total number of particle, 
	//	region index, load of each thread
	// Output:information of output 'particles', and how many number they contain.
	cudaEventRecord(start,0);
	merge_bottom<<<blocks,threads>>>(d_x,d_y,d_mass,d_index,d_rx,d_ry,d_rmass,d_rn,d_n);
	merge_bottom2<<<blocks,threads>>>(d_rx,d_ry,d_rmass);
	//merge_bottom<<<blocks,threads>>>(d_px,d_py,d_pmass,d_regnum,d_n,d_region_index,d_thread_num,d_rx,d_ry,d_rmass,d_rn);
	//merge_bottom<<<blocks,threads>>>(d_px,d_pmass,d_regnum,d_n,d_region_index,d_thread_num,d_rx,d_ry,d_rmass,d_rn);
	cudaFree(d_index);
	gpu_memory -= n*sizeof(int);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
//#ifdef DEBUG_MERGE
	cudaMemcpy(rn,d_rn,n_work*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(rx,d_rx,n_work*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(ry,d_ry,n_work*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(rmass,d_rmass,n_work*sizeof(double),cudaMemcpyDeviceToHost);
#ifdef DEBUG_MERGE
	int st=0;
	for( int i=0;i<n_work;i++ ){
		if(rn[i]==0){
		printf("region:%d, %d particles, xcm=%.3f, ycm=%.3f, mass=%.3f\n",st+i,rn[st+i],rx[st+i],ry[st+i],rmass[st+i]);
		}
	}
#endif
	printf("First step of merge and copy out the result takes %.3f(ms)\n",testtime);
	//===================End of 'Merge' particles in different region=======================


		
	//==================='Merge' particles with shared and global memory====================
	// Calculate memory for shared and global memory
	//int sh_node = (pow(nx/bx,2)-1)*4/3+1;
	//cudaMemcpyToSymbol( share_node, &sh_node, sizeof(int));
	int gl_node = (pow(nx,2)-1)*4/3+1;
	cudaMemcpyToSymbol( global_node, &gl_node, sizeof(int));
	printf("Allocate %d GNODEs\n",gl_node);
	
	// Define global & shared memory
	GNODE *root;
	root = (GNODE *)malloc(gl_node*sizeof(GNODE));
	GNODE *d_root;
	cudaMalloc((void**)&d_root, gl_node*sizeof(GNODE));
	gpu_memory += gl_node*sizeof(GNODE);
	
	// Calculate the Morton ordering of the sub-regions
	// Align the sub-regions with Morton ordering
	int *morton_index,*d_morton_index,*t_morton_index;
	morton_index = (int *)malloc(n_work*sizeof(int));
	cudaMalloc((void**)&d_morton_index,n_work*sizeof(int));
	gpu_memory += n_work*sizeof(int);
	int which_region = 0;
	block(nx,nx,0,0,morton_index,&which_region);
	cudaMemcpy(d_morton_index,morton_index,n_work*sizeof(int),cudaMemcpyHostToDevice);
	printf("Finish calculate morton ordering\n");

	// Call kernel function:
	// Input:center of mass and mass for each subregion, region index
	// Output:information in the shared & global memory
	// Calculate the Morton ordering of the sub-regions
	merge_top1<<<blocks,threads>>>(d_rx,d_ry,d_rmass,d_rn,d_morton_index,d_root);
	dim3 block(1,1);
        dim3 thread(32,32);	
	merge_top2<<<block,thread>>>(d_root);
	cudaMemcpy(root,d_root,gl_node*sizeof(GNODE), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("Second step of merge and copy out the result takes %.3f(ms)\n",testtime);
	cudaFree(d_rx);
	cudaFree(d_ry);
	cudaFree(d_rmass);
	cudaFree(d_morton_index);
	gpu_memory -= n_work*(3*sizeof(double)+sizeof(int));
	
	int test=0;
	for( int i=gl_node-nx*nx;i<gl_node;i++ ){
		test += root[i].num;
		//printf("global=%d,num=%d,xm=%.3f\n",i,root[i].num,root[i].centerofmass[0]);
	}
	printf("Total particle=%d\n",test);
	

#ifdef DEBUG_GLOBAL
	for( int i=0;i<5;i++ ){
		printf("global=%d,num=%d,xm=%.3f\n",i,root[i].num,root[i].centerofmass[0]);
	}
#endif
	cudaEventRecord(start,0);
	force_gpu<<<blocks,threads>>>(d_root,d_x,d_y,d_mass,d_fx,d_fy,d_V,d_n);
	cudaMemcpy(fx,d_fx,n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(fy,d_fy,n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_root);
	gpu_memory -= gl_node*sizeof(GNODE);

	double ftest,r;

	/*for( int i=0;i<20;i++ ){
		ftest = 0;
		for( int j=0;j<n_work;j++ ){
			r = sqrt(pow(rx[j]-x[i],2)+pow(ry[j]-y[i],2));
			ftest += rmass[j]*mass[i]/pow(r,3)*(rx[j]-x[i]);
		}
		printf("particle id = %d,fx=%.3f,fcpu=%.3f\n",i,fx[i],ftest);
	}
	int st=gl_node-n_work;
	for( int i=0;i<20;i++ ){
		ftest = 0;
		for( int j=0;j<n_work;j++ ){
			r = sqrt(pow(root[st+j].centerofmass[0]-x[i],2)+pow(root[st+j].centerofmass[1]-y[i],2));
			ftest += root[st+j].mass*mass[i]/pow(r,3)*(root[st+j].centerofmass[0]-x[i]);
		}
		printf("particle id = %d,x=%.3f,y=%.3f,fx=%.3f\n",i,x[i],y[i],fx[i]);
	}*/
	for( int i=0;i<n;i++ ){
		if( abs(fx[i]-100)<1e-7 ){
			printf("Buffer is not enough!!!\n");
			printf("%d,%.3f\n",i,fx[i]);
			exit(1);
		}else if( abs(fx[i]-200)<1e-7 ){
			printf("Error exist!!!\n");
			printf("%d,%,3f\n",i,fx[i]);
			exit(1);
		}
	}
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("Calculating global force takes %.3f(ms)\n",testtime);
	//========================End of 'Merge' particle with shared and global memory==================
	


	//=============================Do Load Balance=================================
	cudaEventRecord(start,0);
	// Record thread index for each region
	int *reg_thread_index,*reg_index;
	reg_index = (int *)malloc(n_work*sizeof(int));
	reg_thread_index = (int *)malloc(n_work*sizeof(int));
	for( int i=0;i<n_work;i++ ){ reg_index[i]=i; }

	// Record number of regions in each thread
	int *thread_num;
	thread_num = (int *)malloc(n_thread*sizeof(int));
	for( int i=0;i<n_thread;i++ ){ thread_num[i]=0; }

	// Do the load balance
	printf("Use %d regions, each thread takes %d \n",n_work,n_work/n_thread);
	if( n_work<n_thread ){
		for( int i=0;i<n_work;i++ ){
			reg_index[i] = i;
			thread_num[i] = 1;
		}
	}else{
		balance(rn,reg_thread_index,thread_num);
		HeapSort(reg_thread_index,reg_index,n_work);
	}
	free(reg_thread_index);
	
	// Check to region is left and cumsum the load of each thread
	int check = thread_num[0];
	for( int i=1;i<n_thread;i++ ){
		check += thread_num[i];
		thread_num[i] += thread_num[i-1];
	}
	if( check != n_work ){
		printf("Error no load balance!!!\n");
		exit(1);
	}
	
	int *d_region_index,*d_thread_num;
	cudaMalloc((void**)&d_region_index,n_work*sizeof(int));
	cudaMalloc((void**)&d_thread_num,n_thread*sizeof(int));
	cudaMemcpy(d_region_index,reg_index,n_work*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_thread_num,thread_num,n_thread*sizeof(int),cudaMemcpyHostToDevice);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("Load balancing takes %.3f(ms)\n",testtime);

	//=====================End fo load balance=======================

	//=====================Allocate each particles===================	
	cudaEventRecord(start,0);
	// Cumsum of the # of particle in each region
	for( int i=1;i<n_work;i++ ){
		rn[i] += rn[i-1];
	}
	cudaMemcpy(d_rn,rn,n_work*sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	// Sort the particle by the region index
	int *particle_index,*d_particle_index;
	particle_index = (int *)malloc(n*sizeof(int));
	cudaMalloc((void**)&d_particle_index,n*sizeof(int));
	for( int i=0;i<n;i++ ){
		particle_index[i] = i;
	}
	
	double *d_p;
	cudaMalloc((void**)&d_p,n*sizeof(double));
	//cudaMalloc((void**)&d_py,n*sizeof(double));
	//cudaMalloc((void**)&d_pmass,n*sizeof(double));

#ifdef DEBUG_SORTING
	printf("Before sorting:\n");
	for( int i=n-1;i>n-15;i-- ){
		printf("%.3f %.3f\n",x[i],y[i]);
	}
#endif
	/*if( n>6e7 ){
		HeapSort4(index,x,y,mass,particle_index,n);
		cudaMemcpy(d_px,x,n*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(d_py,y,n*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(d_pmass,mass,n*sizeof(double),cudaMemcpyHostToDevice);
	}else{*/
	HeapSort(index,particle_index,n);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("Sorting particles takes %.3f(ms)\n",testtime);

	cudaEventRecord(start,0);
	cudaMemcpy(d_particle_index,particle_index,n*sizeof(int),cudaMemcpyHostToDevice);
	spread_par<<<blocks,threads>>>(d_x,d_p,d_particle_index,d_n);
	cudaMemcpy(x,d_p,n*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(d_x,d_p,n*sizeof(double),cudaMemcpyDeviceToDevice);
	spread_par<<<blocks,threads>>>(d_y,d_p,d_particle_index,d_n);
	cudaMemcpy(y,d_p,n*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(d_y,d_p,n*sizeof(double),cudaMemcpyDeviceToDevice);
	spread_par<<<blocks,threads>>>(d_mass,d_p,d_particle_index,d_n);
	cudaMemcpy(mass,d_p,n*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(d_mass,d_p,n*sizeof(double),cudaMemcpyDeviceToDevice);
	cudaFree(d_p);

	//cudaMemcpy(y,d_py,n*sizeof(double),cudaMemcpyDeviceToHost);
	//}
	/*cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_mass);*/

#ifdef DEBUG_SORTING
	printf("After sorting:\n");
	for( int i=n-1;i>n-15;i-- ){
		printf("%.3f %.3f\n",x[i],y[i]);
	}
#endif
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("Locating particles takes %.3f(ms)\n",testtime);
	//====================End of allocate each particles=====================*/


	
	//=====================Calculate force by n-body============================
	/*for( int i=0;i<n;i++ ){ fx[i] = 100; }
	cudaMemcpy(d_fx,fx,n*sizeof(double),cudaMemcpyHostToDevice);*/
	for( int i=n-1;i>n-20;i-- ){
		printf("particle id = %d,fx=%.3f\n",i,fx[i]);
	}
	cudaEventRecord(start,0);
	treeforce<<<blocks,threads>>>(d_x,d_y,d_mass,d_fx,d_fy,d_V,d_rn,d_region_index,d_thread_num,d_n);
	cudaMemcpy(fy,d_fx,n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(V,d_V,sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("Evaluate force for every subregion takes %.3f(ms)\n",testtime);
	for( int i=n-1;i>n-20;i-- ){
		printf("particle id = %d,fx=%.3f(d=%.3e)\n",i,fy[i],fx[i]-fy[i]);
	}
	cudaFree(d_rn);
	cudaFree(d_region_index);
	cudaFree(d_thread_num);
	//=====================Calculate force by n-body=============================

	
	//=====================Update velocity,position==============================
	cudaEventRecord(start,0);
	double *d_v;
	cudaMalloc((void**)&d_v,n*sizeof(double));
	cudaMalloc((void**)&d_p,n*sizeof(double));
	cudaMemcpy(d_v,vx,n*sizeof(double),cudaMemcpyHostToDevice);
	spread_par<<<blocks,threads>>>(d_v,d_p,d_particle_index,d_n);
	cudaMemcpy(vx,d_p,n*sizeof(double),cudaMemcpyDeviceToHost);
	
	cudaMemcpy(d_v,vy,n*sizeof(double),cudaMemcpyHostToDevice);
	spread_par<<<blocks,threads>>>(d_v,d_p,d_particle_index,d_n);
	cudaMemcpy(vy,d_p,n*sizeof(double),cudaMemcpyDeviceToHost);
	cudaFree(d_p);
	cudaFree(d_v);
	cudaFree(d_particle_index);

	double *d_vx,*d_vy;
	cudaMalloc((void**)&d_vx,n*sizeof(double));
	cudaMalloc((void**)&d_vy,n*sizeof(double));
	cudaMemcpy(d_vx,vx,n*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_vy,vy,n*sizeof(double),cudaMemcpyHostToDevice);

	double *E;
	E = (double *)malloc(sizeof(double));
	*E = 0.0;
	cudaMemcpy(d_Ek,E,sizeof(double),cudaMemcpyHostToDevice);
	update_gpu<<<blocks,threads>>>(d_x,d_y,d_mass,d_vx,d_vy,d_fx,d_fy,d_Ek,d_n);
	cudaMemcpy(x,d_x,n*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(y,d_y,n*sizeof(double),cudaMemcpyDeviceToHost);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_fx);
	cudaFree(d_fy);
	cudaFree(d_vx);
	cudaFree(d_vy);
	cudaFree(d_mass);
	cudaMemcpy(E,d_Ek,sizeof(double),cudaMemcpyDeviceToHost);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("update position & velocity of particles takes %.3f(ms)\n",testtime);
	printf("Total energy = %.3e\n",*E+*V);
	printf("Ek = %.3e V = %.3e\n",*E,*V);






	
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
