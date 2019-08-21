#include "MemoryProfiler.h"

#include <vtksys/SystemInformation.hxx>

#include <vector>
#include <deque>
#include <sstream>
#include <sys/time.h>
#include <cstring>
#include <errno.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>

namespace sensei
{
//namespace timer
//{
struct MemoryProfiler::InternalsType
{
  InternalsType() : Comm(MPI_COMM_WORLD), FileName("MemProf.csv"),
    Interval(60.0), DataMutex(PTHREAD_MUTEX_INITIALIZER)
  {
  }

  MPI_Comm Comm;
  std::string FileName;
  double Interval;
  std::deque<long long> MemUse;
  std::deque<double> TimePt;
  pthread_t Thread;
  pthread_mutex_t DataMutex;
  vtksys::SystemInformation SysInfo;
};

extern "C" void *Profile(void *argp)
{
  MemoryProfiler::InternalsType *internals =
    reinterpret_cast<MemoryProfiler::InternalsType*>(argp);

  while (1)
    {
    // capture the current time and memory usage.
    struct timeval tv;
    gettimeofday(&tv, nullptr);

    double curTime = tv.tv_sec + tv.tv_usec/1.0e6;

    long long curMem = internals->SysInfo.GetProcMemoryUsed();

    pthread_mutex_lock(&internals->DataMutex);

    // log time and mem use
    internals->TimePt.push_back(curTime);
    internals->MemUse.push_back(curMem);

    // get next interval
    double interval = internals->Interval;

    pthread_mutex_unlock(&internals->DataMutex);

    // check for shut down code
    if (interval < 0)
      pthread_exit(nullptr);

    // suspend the thread for the requested interval
    long long secs = floor(interval);
    long nsecs = (interval - secs)*1e9;
    struct timespec sleepTime = {secs, nsecs};

    int ierr = 0;
    int tries = 0;
    while ((ierr = nanosleep(&sleepTime, &sleepTime)) && (errno == EINTR) && (++tries < 1000));
    if (ierr)
      {
      const char *estr = strerror(errno);
      std::cerr << "Error: nanosleep had an error \"" << estr << "\"" << std::endl;
      abort();
      }
    }

  return nullptr;
}


// --------------------------------------------------------------------------
MemoryProfiler::MemoryProfiler()
{
  this->Internals = new InternalsType;
}

// --------------------------------------------------------------------------
MemoryProfiler::~MemoryProfiler()
{
  delete this->Internals;
}

// --------------------------------------------------------------------------
int MemoryProfiler::Initialize()
{
  if (pthread_create(&this->Internals->Thread,
     nullptr, Profile, this->Internals))
    {
    const char *estr = strerror(errno);
    std::cerr << "Error: Failed to create memory profiler. "
      << estr << std::endl;
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int MemoryProfiler::Finalize()
{
  int rank = 0;
  int nRanks = 1;

  MPI_Comm_rank(this->Internals->Comm, &rank);
  MPI_Comm_size(this->Internals->Comm, &nRanks);

  pthread_mutex_lock(&this->Internals->DataMutex);

  // tell the thread to quit
  this->Internals->Interval = -1;

  // create the ascii buffer
  // use ascii in the file as a convenince
  std::ostringstream oss;
  oss.precision(std::numeric_limits<double>::digits10 + 2);
  oss.setf(std::ios::scientific, std::ios::floatfield);

  if (rank == 0)
    oss << "# rank, time, memory kiB" << std::endl;

  long nElem = this->Internals->MemUse.size();
  for (long i = 0; i < nElem; ++i)
    {
    oss << rank << ", " << this->Internals->TimePt[i]
      << ", " << this->Internals->MemUse[i] << std::endl;
    }

  // free resources
  this->Internals->TimePt.clear();
  this->Internals->MemUse.clear();

  pthread_mutex_unlock(&this->Internals->DataMutex);

  // cancle the profiler thread
  pthread_cancel(this->Internals->Thread);

  // compute the file offset
  long nBytes = oss.str().size();

  std::vector<long> gsizes(nRanks);
  gsizes[rank] = nBytes;

  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
    gsizes.data(), 1, MPI_LONG, this->Internals->Comm);

  long offset = 0;
  for (int i = 0; i < rank; ++i)
    offset += gsizes[i];

  long fileSize = 0;
  for (int i = 0; i < nRanks; ++i)
    fileSize += gsizes[i];

  // write the buffer
  MPI_File fh;
  MPI_File_open(this->Internals->Comm, this->Internals->FileName.c_str(),
    MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

  MPI_File_set_view(fh, offset, MPI_BYTE, MPI_BYTE,
     "native", MPI_INFO_NULL);

  MPI_File_write(fh, oss.str().c_str(), nBytes,
    MPI_BYTE, MPI_STATUS_IGNORE);

  MPI_File_set_size(fh, fileSize);

  MPI_File_close(&fh);

  // wait for the proiler thread to finish
  pthread_join(this->Internals->Thread, nullptr);

  return 0;
}

// --------------------------------------------------------------------------
double MemoryProfiler::GetInterval() const
{
  return this->Internals->Interval;
}

// --------------------------------------------------------------------------
void MemoryProfiler::SetInterval(double interval)
{
  pthread_mutex_lock(&this->Internals->DataMutex);
  this->Internals->Interval = interval;
  pthread_mutex_unlock(&this->Internals->DataMutex);
}

// --------------------------------------------------------------------------
void MemoryProfiler::SetCommunicator(MPI_Comm comm)
{
  this->Internals->Comm = comm;
}

// --------------------------------------------------------------------------
void MemoryProfiler::SetFileName(const std::string &fileName)
{
  this->Internals->FileName = fileName;
}

// --------------------------------------------------------------------------
const char *MemoryProfiler::GetFileName() const
{
  return this->Internals->FileName.c_str();
}

}
