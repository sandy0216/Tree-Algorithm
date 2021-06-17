#include <vector>

using namespace std;

struct NODE{
	double 	center[2];
	double 	centerofmass[2];
	double 	mass;
	double 	side;
	int 	num;
	int 	leaf;
	NODE *next[4];
};

struct LOCAL{
	vector<double> x;
	vector<double> y;
	vector<double> mass;
	int num;
};

struct REG{
	double *x;
	double *y;
	double *mass;
	int num;
};
