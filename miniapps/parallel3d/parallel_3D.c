//
// Parallel_3D_Volume.c
//
//
//  Created by Venkatram Vishwanath on 12/17/14.
//
//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <mpi.h>
#ifdef ENABLE_SENSEI
# include "Bridge.h"
#endif

int dim = 3;

// global dimensions of 2D volume
static int g_x = 0;
static int g_y = 0;
static int g_z = 0;

// per-process dimensions of each sub-block
static int l_x = 0;
static int l_y = 0;
static int l_z = 0;

static int bins = 10;

static int parse_args(int argc, char **argv);
static void usage(void);

// Assigns value based on global index
static void gen_data_global_index (double* volume,
                     long long my_off_z, long long my_off_y, long long my_off_x);

static void gen_data (double* volume,
                     long long my_off_z, long long my_off_y, long long my_off_x);

static void gen_data_sparse (double* volume,
                     long long my_off_z, long long my_off_y, long long my_off_x);

#ifdef ENABLE_SENSEI
char* config_file = 0;
#endif

int main(int argc, char **argv)
{
  int nprocs, rank;
  int ret;
  uint64_t  index, slice;
  int tot_blocks_x, tot_blocks_y, tot_blocks_z, tot_blocks;
  uint64_t start_extents_z, start_extents_y, start_extents_x;
  int block_id_x, block_id_y, block_id_z;
  int i,j,k;


  // The buffers/ variables
  // Let's have Pressure, Temperature, Density
  // TODO: Make the num of variables a command line argument
  double* pressure = 0;
  double* temperature = 0;
  double* density = 0;


  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ////////////////////////////
  // parse args on rank 0
  ////////////////////////////
  if(rank == 0){

      ret = parse_args(argc, argv);
      if(ret < 0){
          usage();
          MPI_Abort(MPI_COMM_WORLD, -1);
      }

      // check if the num procs is appropriate
      tot_blocks = (g_z/l_z) * (g_y/l_y) * (g_x/l_x);

      if(tot_blocks != nprocs){
          printf("Error: number of blocks (%d) doesn't match   \
                 number of procs (%d)\n", tot_blocks, nprocs);
          MPI_Abort(MPI_COMM_WORLD, -1);
      }
  }

  /////////////////////////////
  // share the command line args and other params
  /////////////////////////////

  MPI_Bcast(&g_z, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&g_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&g_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&l_z, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&l_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&l_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&tot_blocks, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bins, 1, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef ENABLE_SENSEI
  {
      char buf[2048];
      memset(buf, 0, 2048 * sizeof(char));
      if(rank == 0)
          strcpy(buf, config_file);
      MPI_Bcast(buf, 2048, MPI_CHAR, 0, MPI_COMM_WORLD);
      /* Copy the string on other ranks so config_file is not empty! */
      if(rank != 0)
          config_file = strdup(buf);
  }
#endif

  // figure out some kind of block distribution
  tot_blocks_z = (g_z/l_z);
  tot_blocks_y = (g_y/l_y);
  tot_blocks_x = (g_x/l_x);


  // start extents in Z, Y, X for my block
  if (nprocs == 1)
  {
    start_extents_z = 0; start_extents_y = 0; start_extents_x = 0;
  }
  else
  {
    start_extents_z = (rank / (tot_blocks_y * tot_blocks_x)) * l_z;
    slice = rank % (tot_blocks_y * tot_blocks_x);
    start_extents_y = (slice / tot_blocks_x) * l_y;
    start_extents_x = (slice % tot_blocks_x) * l_x;
  }


  block_id_z = start_extents_z / l_z;
  block_id_y = start_extents_y / l_y;
  block_id_x = start_extents_x / l_x;

  // Print Info
  if (0 == rank){
    printf("Global Dimensions %dX%dX%d: Local Dimensions %dX%dX%d \n", \
          g_z, g_y, g_x, l_z, l_y, l_x);
    printf("Total Blocks are %dX%dX%d \n", tot_blocks_z, tot_blocks_y, tot_blocks_x);
  }

#ifdef ENABLE_SENSEI
  bridge_initialize(MPI_COMM_WORLD,
    g_x, g_y, g_z,
    l_x, l_y, l_z,
    start_extents_x, start_extents_y, start_extents_z,
    tot_blocks_x, tot_blocks_y, tot_blocks_z,
    block_id_x, block_id_y, block_id_z,
    config_file);
#endif

  //////////////////////////////////
  // allocate the variables and
  // intialize to a pattern
  //////////////////////////////////

  // Variable allocation
  pressure = (double*) malloc (sizeof(double) * l_z * l_y *l_x);
  if(!pressure){
      perror("malloc");
      MPI_Abort(MPI_COMM_WORLD, -1);
  }

  temperature = (double*) malloc (sizeof(double) * l_z * l_y *l_x);
  if(!temperature){
    perror("malloc");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  density = (double*) malloc (sizeof(double) * l_z * l_y *l_x);
  if(!density){
    perror("malloc");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }


  // Initialize Variables to a value for testing
  // Note: Currently Set this to the Global Index Value
 /*
  for(k=0; k<l_z; k++){
    for(j=0; j<l_y; j++){
      for(i=0; i<l_x; i++){
        index = (l_x * l_y * k) + (l_x*j) + i;
        pressure[index] = (g_x * g_y * (start_extents_z + k))
                       + (g_x * (start_extents_y + j)) + start_extents_x + i;

        temperature[index] = (g_x * g_y * (start_extents_z + k))
                          + (g_x * (start_extents_y + j)) + start_extents_x + i;

        density[index] = (g_x * g_y * (start_extents_z + k))
                        + (g_x * (start_extents_y + j)) + start_extents_x + i;

      }
    }
  }
 */

  // Intialize the variables to the values of the global index
  gen_data_global_index (pressure, start_extents_z, start_extents_y, start_extents_x);
  /*gen_data_global_index (temperature, start_extents_z, start_extents_y, start_extents_x);*/
  gen_data (temperature, start_extents_z, start_extents_y, start_extents_x);
  /*gen_data_global_index (density, start_extents_z, start_extents_y, start_extents_x);*/
  gen_data_sparse (density, start_extents_z, start_extents_y, start_extents_x);

  // DEBUG: Print the values of the variables..


  MPI_Barrier(MPI_COMM_WORLD);


  /////////////////////////////
  // Iterate over multiple timesteps?
  // Compute several analyses?
  /////////////////////////////

#ifdef ENABLE_SENSEI

    {
    int cc=0;
    for (cc=0; cc < 5; cc++)
      {
      bridge_update(cc, cc*10.0, pressure, temperature, density);
      }
    }

#else
  // "Analysis" routine
  histogram(MPI_COMM_WORLD, pressure, l_z*l_y*l_x, bins);
  histogram(MPI_COMM_WORLD, temperature, l_z*l_y*l_x, bins);
  histogram(MPI_COMM_WORLD, density, l_z*l_y*l_x, bins);
#endif

  MPI_Barrier(MPI_COMM_WORLD);

  /////////////////////////////
  // Clean up heap variables
  /////////////////////////////

#ifdef ENABLE_SENSEI

  bridge_finalize();

  if (config_file) {
    free(config_file);
    config_file = 0;
  }
#endif

  if (pressure){
    free(pressure);
    pressure = 0;
  }

  if (temperature){
    free(temperature);
    temperature = 0;
  }

  if (density){
    free(density);
    density = 0;
  }

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
            sscanf(optarg, "%dx%dx%d", &g_z, &g_y, &g_x);
            break;
        case('l'):
            sscanf(optarg, "%dx%dx%d", &l_z, &l_y, &l_x);
            break;
#ifdef ENABLE_SENSEI
        case('f'):
            config_file = (char*)malloc(strlen(optarg) + 1);
            strncpy(config_file, optarg, strlen(optarg) + 1);
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

    //printf ("Global Values : %d %d %d \n", g_z, g_y, g_x );
    //printf ("Local Values : %d %d %d \n", l_z, l_y, l_x );


  // need positive dimensions
  if(g_z < 1 || g_y < 1 || g_x < 1 ||l_z < 1 || l_y < 1 || l_x < 1 ) {
      printf("Error: bad dimension specification.\n");
      return(-1);
  }

  // need everything to be divisible
  if((g_z % l_z) || (g_y % l_y) || (g_x % l_x)){
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
  printf("  -g global dimensions\n");
  printf("  -l local (per-process) dimensions\n");
#ifdef ENABLE_SENSEI
  printf("  -f Sensei xml configuration file for analysis\n");
#else
  printf("  -b histogram bins\n");
#endif
  printf("\n");
  printf("  test-one-side generates a 3D volume in parallel\n");

  return;
}



static void gen_data_global_index (double* volume,
                     long long my_off_z, long long my_off_y, long long my_off_x)
{
  unsigned long long i,j,k, index;

  for(k=0; k<l_z; k++){
    for(j=0; j<l_y; j++){
      for(i=0; i<l_x; i++){
        index = (l_x * l_y * k) + (l_x*j) + i;

        volume[index] = (g_x * g_y * (my_off_z + k)) +  \
                        (g_x * (my_off_y + j)) + my_off_x + i;

      }
    }
  }

}



/* generate a data set */
static void gen_data(double* volume,
                     long long my_off_z, long long my_off_y, long long my_off_x)
{
  unsigned long long i,j,k;
  double center[3];
  center[0] = g_x / 2.;
  center[1] = g_y / 2.;
  center[2] = g_z / 2.;

    //printf("l_x: %d, l_y: %d, l_z: %d\n", l_x, l_y, l_z);
  for(i = 0; i < l_z; i++)
  {
    double zdist = sin((i + my_off_z - center[2])/5.0)*center[2];
    //      float zdist = sinf((i + my_off_z - center[2])/g_y);
    for(j = 0; j < l_y; j++)
    {
      double ydist = sin((j + my_off_y - center[1])/3.0)*center[1];
        //      float ydist = sinf((j + my_off_y - center[1])/g_x);
      for(k = 0; k < l_x; k++)
      {
        double xdist = sin((k + my_off_x - center[0])/2.0)*center[0];
          //            float xdist = sinf((k + my_off_x - center[0])/g_z);
        volume[i * l_x * l_y + j * l_x + k] = sqrt(xdist * xdist + ydist *ydist + zdist * zdist);

      }
    }
  }

}

static void gen_data_sparse(double* volume,
                            long long my_off_z, long long my_off_y, long long my_off_x)
{
  unsigned long long i,j,k;
  double center[3];
  center[0] = g_x / 2.;
  center[1] = g_y / 2.;
  center[2] = g_z / 2.;

    //printf("l_x: %d, l_y: %d, l_z: %d\n", l_x, l_y, l_z);
  for(i = 0; i < l_z; i++)
  {
    double zdist = i + my_off_z - center[2];
    for(j = 0; j < l_y; j++)
    {
      double ydist = j + my_off_y - center[1];
      for(k = 0; k < l_x; k++)
      {
        double xdist = k + my_off_x - center[0];

        volume[i * l_x * l_y + j * l_x + k] = sqrt(xdist * xdist + ydist *ydist + zdist * zdist);

      }
    }
  }


}




