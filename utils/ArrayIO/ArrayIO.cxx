#include "ArrayIO.h"

#include <sstream>
using std::ostringstream;

namespace arrayIO
{
// ****************************************************************************
int open(
        MPI_Comm comm,                 // MPI communicator handle
        const char *fileName,          // file name to write.
        MPI_Info hints,                // MPI file hints
        MPI_File &file)                // file handle
{
  int iErr = 0;
#ifndef NDEBUG
  int mpiOk = 0;
  MPI_Initialized(&mpiOk);
  if (!mpiOk)
  {
    std::cerr << "This class requires the MPI runtime" << std::endl;
    return -1;
  }
#endif
  const int eStrLen = 2048;
  char eStr[eStrLen] = {'\0'};
  // Open the file
  if ((iErr = MPI_File_open(comm, const_cast<char *>(fileName),
      MPI_MODE_WRONLY|MPI_MODE_CREATE, hints, &file)))
  {
    MPI_Error_string(iErr, eStr, const_cast<int *>(&eStrLen));
    std::cerr << "Error opeing file: " << fileName << std::endl
        << eStr << std::endl;
    return -1;
  }
  return 0;
}

// ****************************************************************************
MPI_Info createHints(
    int useCollectiveIO,
    int numberOfIONodes,
    int collectBufferSize,
    int useDirectIO,
    int useDeferredOpen,
    int useDataSieving,
    int sieveBufferSize,
    int stripeCount,
    int stripeSize)
{
  MPI_Info hints = MPI_INFO_NULL;

  int mpiOk;
  MPI_Initialized(&mpiOk);
  if (!mpiOk)
    {
    std::cerr << "This class requires the MPI runtime" << std::endl;
    return hints;
    }

  MPI_Info_create(&hints);

  switch (useCollectiveIO)
    {
    case HINT_AUTOMATIC:
      // do nothing, it's up to implementation.
      break;
    case HINT_DISABLED:
      MPI_Info_set(hints,"romio_cb_write","disable");
      break;
    case HINT_ENABLED:
      MPI_Info_set(hints,"romio_cb_write","enable");
      break;
    default:
      std::cerr << "Invalid value for UseCollectiveIO." << std::endl;
      break;
    }

  if (numberOfIONodes > 0)
    {
    std::ostringstream os;
    os << numberOfIONodes;
    MPI_Info_set(hints,"cb_nodes",const_cast<char *>(os.str().c_str()));
    }

  if (collectBufferSize > 0)
    {
    std::ostringstream os;
    os << collectBufferSize;
    MPI_Info_set(hints,"cb_buffer_size",const_cast<char *>(os.str().c_str()));
    //MPI_Info_set(hints,"striping_unit", const_cast<char *>(os.str().c_str()));
    }

  switch (useDirectIO)
    {
    case HINT_DEFAULT:
      // do nothing, it's up to implementation.
      break;
    case HINT_DISABLED:
      MPI_Info_set(hints,"direct_write","false");
      break;
    case HINT_ENABLED:
      MPI_Info_set(hints,"direct_write","true");
      break;
    default:
      std::cerr << "Invalid value for UseDirectIO." << std::endl;
      break;
    }

  switch (useDeferredOpen)
    {
    case HINT_DEFAULT:
      // do nothing, it's up to implementation.
      break;
    case HINT_DISABLED:
      MPI_Info_set(hints,"romio_no_indep_rw","false");
      break;
    case HINT_ENABLED:
      MPI_Info_set(hints,"romio_no_indep_rw","true");
      break;
    default:
      std::cerr << "Invalid value for UseDeferredOpen." << std::endl;
      break;
    }

  switch (useDataSieving)
    {
    case HINT_AUTOMATIC:
      // do nothing, it's up to implementation.
      break;
    case HINT_DISABLED:
      MPI_Info_set(hints,"romio_ds_write","disable");
      break;
    case HINT_ENABLED:
      MPI_Info_set(hints,"romio_ds_write","enable");
      break;
    default:
      std::cerr << "Invalid value for UseDataSieving." << std::endl;
      break;
    }

  if (sieveBufferSize > 0)
    {
    std::ostringstream os;
    os << sieveBufferSize;
    MPI_Info_set(hints,"ind_rd_buffer_size", const_cast<char *>(os.str().c_str()));
    }

  if (stripeCount > 0)
    {
    std::ostringstream os;
    os << stripeCount;
    MPI_Info_set(hints,"striping_count", const_cast<char *>(os.str().c_str()));
    }

  if (stripeSize > 0)
    {
    std::ostringstream os;
    os << stripeSize;
    MPI_Info_set(hints,"striping_unit", const_cast<char *>(os.str().c_str()));
    }

  return hints;
}
}
