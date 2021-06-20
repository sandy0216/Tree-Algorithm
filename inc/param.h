const double		boxsize	= 100.0;
const double 		theta	= 0.8;
const unsigned long	initial_n = 1e7;
const double 		eplison = boxsize*3/initial_n;

const double 		dt	= 1e-9;


const int nx	= 200; // Must be k*4
const int ny	= 200;
const int tx	= 4;
const int ty	= 4;
const int bx	= 4;
const int by	= 4;
const int n_work = nx*ny;
const int n_thread = (tx*ty)*(bx*by);


//#define OPENMP




