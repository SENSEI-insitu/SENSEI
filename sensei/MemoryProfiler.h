#ifndef MemoryProfiler_h
#define MemoryProfiler_h

#include <mpi.h>
#include <string>

namespace sensei
{
//namespace timer
//{

extern "C" void *Profile(void *argp);

/// MemoryProfiler - A sampling memory use profiler
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
  MemoryProfiler(const MemoryProfiler &) = delete;
  void operator=(const MemoryProfiler &) = delete;

  MemoryProfiler();
  ~MemoryProfiler();

  int Initialize();
  int Finalize();

  /// Set the interval in seconds between querrying
  /// the processes memory use.
  void SetInterval(double interval);
  double GetInterval() const;

  /// Set the comunicator for parallel I/O
  void SetCommunicator(MPI_Comm comm);

  /// Set the file name to write the data to
  void SetFileName(const std::string &fileName);
  const char *GetFileName() const;

  friend void *Profile(void *argp);

private:
  struct InternalsType;
  InternalsType *Internals;
};

}
//}

#endif
