#include <conduit.hpp>
#include <cstring>
#include <conduit_blueprint.hpp>
#include <mpi.h>
#include "bridge.h"

int main(int argc, char* argv[])
{
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
    
  conduit::Node res;
  conduit::blueprint::mesh::examples::spiral( 7, res);
  initialize(MPI_COMM_WORLD, &res, argv[1]);

  analyze(&res);

  finalize();

  MPI_Finalize();
    
  return 0;
}
