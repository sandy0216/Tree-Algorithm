#include <cstdio>
#include <cstring>
#include <vector>
#include <omp.h>
#include "../inc/init.h"
#include "../inc/def_node.h"
#include "../inc/create_tree.h"
#include "../inc/create_tree_gpu.h"
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
__device__ int dn_gnode;



int main( int argc, char* argv[] )
{
	double *x, *y, *mass;
	double *vx, *vy;
	double *fx,*fy;
	double *V;
	double E,Ek;
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

	//==================GPU settings==============================
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

	// Set basic parameters of GPU
	cudaMemcpyToSymbol( d_boxsize, &boxsize, sizeof(double));
	cudaMemcpyToSymbol( d_side, &nx, sizeof(int));
	cudaMemcpyToSymbol( d_n_work, &n_work, sizeof(int));
	cudaMemcpyToSymbol( d_n_thread, &n_thread, sizeof(int));
	

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

	// Split the particle into different subregion
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

	// =============================Do Load Balance=================================
	cudaEventRecord(start,0);
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
	block(nx,0,0,region_index,&which_region);
	//printf("expect load=%d\n",n/n_thread);
	for( int i=0;i<n_thread;i++ ){ thread_load[i]=0; }
	for( int i=0;i<n_work;i++ ){
		reg_id = region_index[i];
		current_thread_load += regnum[reg_id];
		thread_load[which_thread] += 1;
		if( current_thread_load>n/n_thread-regnum[reg_id]/4 && which_thread<n_thread-1 ){
			//printf("thread %d, load %d, take %d region\n",which_thread,current_thread_load,thread_load[which_thread]);
			which_thread+=1;
			current_thread_load = 0;
		}
		if( current_thread_load>n/n_thread*2 ){
			printf("Load no balance!!!\n");
			exit(1);
		}
	}

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
	// =====================End of Load Balance===========================
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("Load balance&copy the result takes %.3e(ms)\n",time);	
		
	// Cumsum of the # of particle in each region
	for( int i=1;i<n_work;i++ ){
		//printf("Region %d, %d particles\n",i,regnum[i]);
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

	// Define GPU parameters
	NODE *p_local_node;
	p_local_node = (NODE *)malloc(n_work*sizeof(NODE));
	NODE *d_local_node;
	cudaMalloc((void**)&d_local_node, n_work*sizeof(NODE));
	NODE *node_ly1;
	node_ly1 = (NODE *)malloc(n_work/4*sizeof(NODE));
	NODE *d_node_ly1;
	cudaMalloc((void**)&d_node_ly1, n_work/4*sizeof(NODE));

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

	int n_gnode = (pow(nx/bx,2)-1)*4/3+1;
	cudaMemcpyToSymbol( dn_gnode, &n_gnode, sizeof(int));
	
	int *d_flag,*flag;
	cudaMalloc((void**)&d_flag, n_gnode*sizeof(int));
	flag = (int *)malloc(n_gnode*sizeof(int));
	for( int i=0;i<n_gnode;i++ ){ flag[i]=0; }
	cudaMemcpy(d_flag,flag,n_gnode*sizeof(int),cudaMemcpyHostToDevice);
	
	printf("n_gnode=%d\n",n_gnode);
	cudaEventRecord(start,0);
	merge_gpu<<<threads,blocks>>>(d_x,d_y,d_mass,d_particle_index,d_regnum,d_n,d_region_index,d_thread_load,d_rx,d_ry,d_rmass,d_rn,d_flag);
	//merge_tree<<<threads,blocks>>>(d_rx,d_ry,d_rmass,d_flag);
	cudaMemcpy(rn,d_rn, n_work*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(rx,d_rx, n_work*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(ry,d_ry, n_work*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(rmass,d_rmass, n_work*sizeof(double), cudaMemcpyDeviceToHost);

	int sm = n_gnode*sizeof(GNODE); // for x,y,mass,num
	merge_top<<<threads,blocks,sm>>>(d_rx,d_ry,d_rmass,d_rn,d_region_index,d_flag);
	cudaMemcpy(flag,d_flag,n_gnode*sizeof(int), cudaMemcpyDeviceToHost);

	for( int i=0;i<n_gnode;i++ ){
		printf("region %d, topn=%d\n",i,flag[i]);
	}
	/*NODE *root;
	root = new NODE();
	create_tree(root,rx,ry,rmass,n_work);*/
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&testtime, start, stop);
	printf("First step of merge and copy out the result takes %.3f(ms)\n",testtime);


	cudaMemcpy(flag,d_flag,n_thread*sizeof(int),cudaMemcpyDeviceToHost);

	//bool check;
	//check = CUDA_CHECK_ERROR(CUDA_FUNCTION(tree));
	//printf("%d\n",check);
	/*NODE *head;
	for( int i=1;i<n_work;i++ ){
		head = &p_local_node[i];
		printf("%d\t%d\t%d\n",i, regnum[i]-regnum[i-1],head->num);
	}*/
	//int total=flag[0];	
	/*for( int i=1;i<n_work;i++ ){
		printf("region id=%d, GPU=%d, CPU=%d, xcm=%.3f\n",i,rn[i],regnum[i]-regnum[i-1],rx[i]);
	}*/
	/*for( int i=1;i<n_thread;i++ ){
		printf("thread id=%d, load = %d\n",i,thread_load[i]-thread_load[i-1]);
	}
	printf("%d\n",thread_load[n_thread-1]);
	printf("well done\n");*/
	printf("size of NODE = %d\n",sizeof(NODE));
	printf("size of GNODE = %d\n",sizeof(GNODE));







	
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
	}*/
	

	return 0;
}
