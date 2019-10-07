#ifndef sensei_MPIManager_h
#define sensei_MPIManager_h

#include "senseiConfig.h"
#define SENSEI_HAS_MPI

namespace sensei
{

/// A RAII class to ease MPI initalization and finalization
// MPI_Init is handled in the constructor, MPI_Finalize is handled in the
// destructor. Given that this is an application level helper rank and size
// are reported relatoive to MPI_COMM_WORLD.
class MPIManager
{
public:
  MPIManager() = delete;
  MPIManager(const MPIManager &) = delete;
  void operator=(const MPIManager &) = delete;

  MPIManager(int &argc, char **&argv);
  ~MPIManager();

  int GetCommRank(){ return mRank; }
  int GetCommSize(){ return mSize; }

private:
  int mRank;
  int mSize;
};

}

#endif
