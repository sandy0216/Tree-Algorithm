
struct NODE{
	double 	center[2];
	double 	centerofmass[2];
	double 	mass;
	double 	side;
	int 	num;
	int 	leaf;
	NODE *next[4];//={next1,next2,next3,next4};
	//NODE *next2;
	//NODE *next3;
	//NODE *next4;
};
