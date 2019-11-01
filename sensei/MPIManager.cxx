#include "MPIManager.h"
#include "Profiler.h"
#include "Error.h"

#include <cstdlib>

using seconds_t =
  std::chrono::duration<double, std::chrono::seconds::period>;

namespace sensei
{

// --------------------------------------------------------------------------
MPIManager::MPIManager(int &argc, char **&argv)
  : mRank(0),  mSize(1)
{
  Profiler::Enable(0x01);
  Profiler::StartEvent("TotalRunTime");
  Profiler::StartEvent("AppInitialize");

#if defined(SENSEI_HAS_MPI)
  int required = MPI_THREAD_SERIALIZED;
  int provided = 0;
  MPI_Init_thread(&argc, &argv, required, &provided);
  if (provided < required)
    {
    SENSEI_ERROR("This MPI does not support thread serialized");
    abort();
    }
#else
  (void)argc;
  (void)argv;
#endif

  Profiler::Disable();
  Profiler::SetCommunicator(MPI_COMM_WORLD);
  Profiler::Initialize();

#if defined(SENSEI_HAS_MPI)
  MPI_Comm_rank(MPI_COMM_WORLD, &mRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mSize);
#endif

  Profiler::EndEvent("AppInitialize");
}

// --------------------------------------------------------------------------
MPIManager::~MPIManager()
{
  Profiler::StartEvent("AppFinalize");
  Profiler::Finalize();

#if defined(SENSEI_HAS_MPI)
  int ok = 0;
  MPI_Initialized(&ok);
  if (ok)
    MPI_Finalize();
#endif

  Profiler::EndEvent("AppFinalize");
  Profiler::EndEvent("TotalRunTime");

  if (mRank == 0)
    Profiler::Flush();
}

}
