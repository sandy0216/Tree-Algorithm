#include <cstdio>
#include <cstring>
#include "../inc/def_node.h"
#include "../inc/print_tree.h"
#include "../inc/tool_tree.h"
#include "../inc/param.h"

//#define DEBUG_force

void node_force(NODE *head, double x, double y, double mass, double *dfx, double *dfy)
{
	double cmx, cmy, r, side, m;
	cmx  = head->centerofmass[0];
	cmy  = head->centerofmass[1];
	side = head->side;
	m    = head->mass;
	r   = sqrt(pow(cmx-x,2)+pow(cmy-y,2));
#ifdef DEBUG_force
	printf("Calculating node center at %.3f,%.3f\n",head->center[0],head->center[1]);
	printf("Center of mass = %.3f, %.3f\n",head->centerofmass[0],head->centerofmass[1]);
	printf("Position of particle = %.3f, %.3f\n",x,y);
#endif
	if( r<1e-7 ){
#ifdef DEBUG_force
		printf("Too close, skip this node\n");
		printf("particle position:%.3f,%.3f\n",x,y);
		printf("Center of mass = %.3f, %.3f\n",cmx,cmy);
#endif
	}else if( side/r < theta || head->num==1 ){
		if( r>eplison ){
			*dfx = *dfx+mass*m/pow(r,2)*(cmx-x)/r;
			*dfy = *dfy+mass*m/pow(r,2)*(cmy-y)/r;
		}else{
			*dfx = *dfx+mass*m/sqrt(pow(pow(r,2)+pow(eplison,2),3))*(cmx-x);
			*dfy = *dfy+mass*m/sqrt(pow(pow(r,2)+pow(eplison,2),3))*(cmy-y);
		}
#ifdef DEBUG_force
		printf("r=%.3f, D=%.3f, D/r=%.3f, take\n",r,side,side/r );
		printf("[after]fx, fy=%.3f,%.3f\n",*dfx,*dfy);
#endif
	}else{
#ifdef DEBUG_force
		printf("D/r=%.3f, explore\n", side/r);
#endif
		for( int i=0;i<4;i++ ){
		if( head->next[i] != NULL ){
#ifdef DEBUG_force
			printf("Exploring node %d......\n",i);	
#endif
			node_force(head->next[i],x,y,mass,dfx,dfy);
		}
		}
	}
}// End of the function

void node_potential(NODE *head, double x, double y, double mass, double *dv)
{
	double cmx, cmy, r, side, m;
	cmx  = head->centerofmass[0];
	cmy  = head->centerofmass[1];
	side = head->side;
	m    = head->mass;
	r   = sqrt(pow(cmx-x,2)+pow(cmy-y,2));
	if( r<1e-3 ){
	}else if( side/r < theta || head->num==1 ){
		*dv = *dv-mass*m/r;
	}else{
		for( int i=0;i<4;i++ ){
		if( head->next[i] != NULL ){
			node_potential(head->next[i],x,y,mass,dv);
		}
		}
	}
}



void cal_ana(double px, double py, double pmass, double *x, double *y, double *mass, int n, double tx, double ty)
{
	double afx=0;
	double afy=0;
	double r;
	for( int i=0;i<n;i++ ){
		r = sqrt(pow(x[i]-px,2)+pow(y[i]-py,2));
		if( r>1e-3 ){
			afx += pmass*mass[i]/pow(r,2)*(x[i]-px)/r;
			afy += pmass*mass[i]/pow(r,2)*(y[i]-py)/r;
		}
	}
	if( afx/tx>10 || afy/ty>10 || afx/tx<0.1 || afy/ty<0.1 ){
	printf("Position of the particle:%.3f,%.3f\n",px,py);
	printf("Tree solution (fx,fy):\t\t%.3e,%.3e\n",tx,ty);
	printf("Analytic solution (fx,fy):\t%.3e,%.3e\n",afx,afy);
	}
}

void direct_nbody( double px, double py, double pmass, double *x, double *y, double *mass, int n, double *fx, double *fy)
{
	*fx = 0;
	*fy = 0;
	double r;
	for( int i=0;i<n;i++ ){
		r = sqrt(pow(x[i]-px,2)+pow(y[i]-py,2));
		if( r>1e-6 ){
			*fx += pmass*mass[i]/pow(r,2)*(x[i]-px)/r;
			*fy += pmass*mass[i]/pow(r,2)*(y[i]-py)/r;
		}
	}
}

void force(NODE *head, double *x, double *y, double *mass, double *fx, double *fy, int n)
{
	double *pfx, *pfy;
	for( int i=0;i<n;i++ ){
		fx[i]=fy[i]=0;
		pfx = &fx[i];
		pfy = &fy[i];
		//direct_nbody(x[i],y[i],mass[i],x,y,mass,n,pfx,pfy);
		node_force(head,x[i],y[i],mass[i],pfx,pfy);
#ifdef DEBUG_force
		cal_ana(x[i],y[i],mass[i],x,y,mass,n,*pfx,*pfy);
#endif
	}
}// End of the function

void potential( NODE *head, double *x, double *y, double *mass, double *V, int n)
{
	for( int i=0;i<n;i++ ){
		V[i]=0;
		node_potential(head,x[i],y[i],mass[i],&V[i]);
	}
}










