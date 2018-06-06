#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <mpi.h>

#include "senseiConfig.h"
#ifdef ENABLE_SENSEI
#include "Bridge.h"
#else
#include "histogram.h"
#endif

static int parse_args(int argc, char **argv);
static void usage(void);

static void gen_data_global_index (double* volume,
  long long my_off_z, long long my_off_y, long long my_off_x);

static void gen_data (double* volume,
  long long my_off_z, long long my_off_y, long long my_off_x);

static void gen_data_sparse (double* volume,
  long long my_off_z, long long my_off_y, long long my_off_x);

// global dimensions of volume
static int g_nx = 0;
static int g_ny = 0;
static int g_nz = 0;

// per-process dimensions of each sub-block
static int l_nx = 0;
static int l_ny = 0;
static int l_nz = 0;

#ifdef ENABLE_SENSEI
static char *config_file = 0;
#else
static int bins = 10;
#endif

int main(int argc, char **argv)
{
  uint64_t slice;
  int tot_blocks_x, tot_blocks_y, tot_blocks_z;
  uint64_t offs_z, offs_y, offs_x;

  // The buffers/ variables
  double *pressure = NULL;
  double *temperature = NULL;
  double *density = NULL;

  // Initialize MPI
  MPI_Init(&argc, &argv);

  int nprocs = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // parse args on rank 0
  int ret = parse_args(argc, argv);
  if ((rank == 0) && (ret < 0))
  {
    usage();
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  // check if the num procs is appropriate
  int tot_blocks = (g_nz/l_nz) * (g_ny/l_ny) * (g_nx/l_nx);
  if ((rank == 0) && (tot_blocks != nprocs))
  {
    printf("Error: number of blocks (%d) doesn't match "
      "number of procs (%d)\n", tot_blocks, nprocs);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  // check block size. histogram will segv if it is given
  // a constant array
  int local_block_size = l_nx*l_ny*l_nz;
  if ((rank == 0) && (local_block_size < 2))
  {
    printf("Error: local block must have more than 1 cell\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  // figure out some kind of block distribution
  tot_blocks_z = (g_nz/l_nz);
  tot_blocks_y = (g_ny/l_ny);
  tot_blocks_x = (g_nx/l_nx);

  // start extents in Z, Y, X for my block
  if (nprocs == 1)
  {
    offs_z = 0; offs_y = 0; offs_x = 0;
  }
  else
  {
    offs_z = (rank / (tot_blocks_y * tot_blocks_x)) * l_nz;
    slice = rank % (tot_blocks_y * tot_blocks_x);
    offs_y = (slice / tot_blocks_x) * l_ny;
    offs_x = (slice % tot_blocks_x) * l_nx;
  }

  // Print Info
  if (0 == rank)
  {
    printf("Global Dimensions %dX%dX%d: Local Dimensions %dX%dX%d \n",
      g_nz, g_ny, g_nx, l_nz, l_ny, l_nx);

    printf("Total Blocks are %dX%dX%d \n",
      tot_blocks_z, tot_blocks_y, tot_blocks_x);
  }

  // Allocate the variables
  long nbytes = sizeof(double)*l_nz*l_ny*l_nx;

  pressure = (double*)malloc(nbytes);
  temperature = (double*)malloc(nbytes);
  density = (double*)malloc(nbytes);

#ifdef ENABLE_SENSEI
  // Initialize sensei adaptors
  if (bridge_initialize(config_file, g_nx, g_ny, g_nz,
    offs_x, offs_y, offs_z, l_nx, l_ny, l_nz, pressure,
    temperature, density))
  {
    printf("Error: failed to initialize sensei\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
#endif

  // Intialize the variables
  gen_data_global_index(pressure, offs_z, offs_y, offs_x);
  gen_data(temperature, offs_z, offs_y, offs_x);
  gen_data_sparse(density, offs_z, offs_y, offs_x);

  // Iterate over multiple timesteps
  for (int i = 0; i < 5; ++i)
  {
#ifdef ENABLE_SENSEI
    // use sensei to do the analysis
    bridge_update(i, i*10.0);
#else
    // compute a histogram directly
    histogram(MPI_COMM_WORLD, pressure, l_nz*l_ny*l_nx, bins);
    histogram(MPI_COMM_WORLD, temperature, l_nz*l_ny*l_nx, bins);
    histogram(MPI_COMM_WORLD, density, l_nz*l_ny*l_nx, bins);
#endif
  }

  // Clean up heap variables
#ifdef ENABLE_SENSEI
  bridge_finalize();
  free(config_file);
#endif
  free(pressure);
  free(temperature);
  free(density);

  MPI_Finalize();

  return(0);
}


/* parse_args()
 *
 * parses command line argument
 *
 * returns 0 on success, -1 on failure
 */
static int parse_args(int argc, char **argv)
{
#ifdef ENABLE_SENSEI
    char flags[] = "g:l:f:";
#else
    char flags[] = "g:l:b:";
#endif
    int one_opt = 0;


  while((one_opt = getopt(argc, argv, flags)) != EOF){
  // postpone error checking for after while loop */
    switch(one_opt){
        case('g'):
            sscanf(optarg, "%dx%dx%d", &g_nz, &g_ny, &g_nx);
            break;
        case('l'):
            sscanf(optarg, "%dx%dx%d", &l_nz, &l_ny, &l_nx);
            break;
#ifdef ENABLE_SENSEI
        case('f'):
            config_file = (char*)malloc(strlen(optarg) + 1);
            strcpy(config_file, optarg);
            break;
#else
        case('b'):
            sscanf(optarg, "%d", &bins);
            break;
#endif
        case('?'):
            return(-1);
    }
  }

    //printf ("Global Values : %d %d %d \n", g_nz, g_ny, g_nx );
    //printf ("Local Values : %d %d %d \n", l_nz, l_ny, l_nx );


  // need positive dimensions
  if(g_nz < 1 || g_ny < 1 || g_nx < 1 ||l_nz < 1 || l_ny < 1 || l_nx < 1 ) {
      printf("Error: bad dimension specification.\n");
      return(-1);
  }

  // need everything to be divisible
  if((g_nz % l_nz) || (g_ny % l_ny) || (g_nx % l_nx)){
      printf("Error: global dimensions and local dimensions aren't evenly divisible\n");
      return(-1);
  }
#ifdef ENABLE_SENSEI
  if (config_file == NULL) {
    printf("Error: please specify a sensei config xml\n");
    return(-1);
  }
#endif

  return 0;
}

// prints usage instructions
static void usage(void)
{
#ifdef ENABLE_SENSEI
  printf("Usage: <exec> -g 4x4x4 -l 2x2x2 -f config.xml \n");
#else
  printf("Usage: <exec> -g 4x4x4 -l 2x2x2 -b 10 \n");
#endif
  printf("  -g global mesh dimensions in number of cells\n");
  printf("  -l number of cells in a local block, must divide the global space into number of MPI Comm size blocks\n");
#ifdef ENABLE_SENSEI
  printf("  -f Sensei xml configuration file for analysis\n");
#else
  printf("  -b histogram bins\n");
#endif
  printf("\n");
  printf("generates a 3D volume in parallel\n");
  return;
}

static void gen_data_global_index (double* volume,
  long long my_off_z, long long my_off_y, long long my_off_x)
{
  for(long long k = 0; k < l_nz; ++k)
  {
    for(long long j = 0; j < l_ny; ++j)
    {
      for(long long i = 0; i < l_nx; ++i)
      {
        long long index = (l_nx * l_ny * k) + (l_nx*j) + i;

        volume[index] = (g_nx * g_ny * (my_off_z + k)) +
                        (g_nx * (my_off_y + j)) + my_off_x + i;

      }
    }
  }

}

/* generate a data set */
static void gen_data(double* volume,
  long long off_z, long long off_y, long long off_x)
{
  double center[3] = {g_nx/2.0, g_ny/2.0, g_nz/2.0};

  for(long long k = 0; k < l_nz; ++k)
  {
    double zdist = sin((k + off_z - center[2])/5.0)*center[2];
    zdist *= zdist;

    for(long long j = 0; j < l_ny; ++j)
    {
      double ydist = sin((j + off_y - center[1])/3.0)*center[1];
      ydist *= ydist;

      for(long long i = 0; i < l_nx; ++i)
      {
        double xdist = sin((i + off_x - center[0])/2.0)*center[0];
        xdist *= xdist;

        long long index = (l_nx * l_ny * k) + (l_nx*j) + i;

        volume[index] = sqrt(xdist + ydist + zdist);
      }
    }
  }
}

static void gen_data_sparse(double* volume,
  long long off_z, long long off_y, long long off_x)
{
  double center[3] = {g_nx/2.0, g_ny/2.0, g_nz/2.0};

  for(long long k = 0; k < l_nz; ++k)
  {
    double zdist = k + off_z - center[2];
    zdist *= zdist;

    for(long long j = 0; j < l_ny; ++j)
    {
      double ydist = j + off_y - center[1];
      ydist *= ydist;

      for(long long i = 0; i < l_nx; ++i)
      {
        double xdist = i + off_x - center[0];
        xdist *= xdist;

        long long index = (l_nx * l_ny * k) + (l_nx*j) + i;
        volume[index] = sqrt(xdist + ydist + zdist);
      }
    }
  }
}
