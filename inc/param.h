const double		boxsize	= 100.0;
const double 		theta	= 0.8;
const unsigned long	initial_n = 100;
const double 		eplison = boxsize*3/initial_n;

const double 		dt	= 1e-9;

const int tx	= 2;
const int ty	= 2;
const int bx	= 2;
const int by	= 2;
const int nx	= 4*tx*bx;
const int ny	= 4*ty*by;
const int n_work = nx*ny;


//#define OPENMP




