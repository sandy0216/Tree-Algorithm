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

struct GNODE{
	float centerofmass[2];
	float  mass;
	float side;
	int num;
	int leaf;
	GNODE *next[4];
};
