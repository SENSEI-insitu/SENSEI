#ifndef sensei_MemoryProfiler_h
#define sensei_MemoryProfiler_h

#include "senseiConfig.h"
#include <mpi.h>
#include <string>

extern "C" void *profile(void *argp);

namespace sensei
{

// MemoryProfiler - A sampling memory use profiler
/**
The class samples process memory usage at the specified interval
given in seconds. For each sample the time is aquired. Calling
Initialize starts profiling, and Finalize ends it. During
Finaliziation the buffers are written using MPI-I/O to the
file name provided
*/
class MemoryProfiler
{
public:
  MemoryProfiler();
  ~MemoryProfiler();

  MemoryProfiler(const MemoryProfiler &) = delete;
  void operator=(const MemoryProfiler &) = delete;

  // start and stop the profiler
  int Initialize();
  int Finalize();

  // Set the interval in seconds between querrying
  // the processes memory use.
  void SetInterval(double interval);
  double GetInterval() const;

  // Set the comunicator for parallel I/O
  void SetCommunicator(MPI_Comm comm);

  // Set the file name to write the data to
  void SetFilename(const std::string &filename);
  const char *GetFilename() const;

  friend void *::profile(void *argp);

private:
  struct InternalsType;
  InternalsType *Internals;
};

}
#endif
