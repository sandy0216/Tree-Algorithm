
/*const double		boxsize= 100.0;
const double 		theta	= 0.8;
const unsigned long	initial_n = 1e7;
const double 		eplison = boxsize*3/initial_n;

const double 		dt	= 1e-7;


const int nx	= 1024; // Must be k*4
const int ny	= 1024;
const int tx	= 16;
const int ty	= 16;
const int bx	= 32;
const int by	= 32;
const int n_work = nx*nx;
const int n_thread = tx*ty*bx*by;*/

extern double boxsize;
extern unsigned long initial_n;
extern double theta;
extern double eplison;
extern double dt;
extern int nx;
extern int ny;
extern int tx;
extern int ty;
extern int bx;
extern int by;
extern int n_work;
extern int n_thread;

extern __device__ double d_boxsize;
extern __device__ double d_theta;
extern __device__ double d_eplison;
extern __device__ double d_dt;
extern __device__ int d_n_work;
extern __device__ int d_n_thread;
extern __device__ int d_side; // =nx=ny
extern __device__ int d_bx;
extern __device__ int share_node;
extern __device__ int global_node;



//#define OPENMP
//#define DEBUG_MERGE
//#define DEBUG_SHARE
//#define DEBUG_CACHE
//#define DEBUG_FORCE

//#define RECORD_INI
#define OUTPUT_DETAIL

