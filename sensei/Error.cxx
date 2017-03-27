#include "Error.h"

#include <mpi.h>
#include <unistd.h>
#include <cstdio>
#include <ostream>
#include <sstream>

using std::ostringstream;
using std::ostream;

namespace sensei
{

// --------------------------------------------------------------------------
int haveTty()
{
  static int have = -1;
  if (have < 0)
    have = isatty(fileno(stderr));
  return have;
}

// --------------------------------------------------------------------------
ostream &operator<<(ostream &os, const parallelId &)
{
  int rank = 0;
  int isInit = 0;
  MPI_Initialized(&isInit);
  if (isInit)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ostringstream oss;
  oss << rank;
  os << oss.str();
  return os;
}

// --------------------------------------------------------------------------
int ioEnabled(int active_rank)
{
  if (active_rank < 0)
    return 1;

  int rank = 0;
  int isInit = 0;
  MPI_Initialized(&isInit);
  if (isInit)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == active_rank)
    return 1;

  return 0;
}

}
