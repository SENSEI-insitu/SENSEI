#include "Profiler.h"
#include "MemoryProfiler.h"
#include "Error.h"

#include <sys/time.h>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdint.h>
#include <strings.h>
#include <cstdlib>
#include <cstdlib>
#include <cstdio>

#include <map>
#include <list>
#include <vector>
#include <iomanip>
#include <limits>
#include <unordered_map>
#include <mutex>

namespace impl
{
#if defined(ENABLE_PROFILER)

// container for data captured in a timing Event
struct Event
{
  Event();

  // serializes the Event in CSV format into the stream.
  void ToStream(std::ostream &str) const;

  enum { START=0, END=1, DELTA=2 }; // record fields

  // user provided identifier for the record
  std::string Name;

  // Event duration, initially start Time, end Time and duration
  // are recorded. when summarizing this contains min,max and sum
  // of the summariezed set of events
  double Time[3];

  // the number of bytes, if this is an I/O or datamovement operation
  // else -1
  long long NumBytes;

  // how deep is the Event stack
  int Depth;

  // the thread id that generated the Event
  std::thread::id Tid;
};

#if !defined(SENSEI_HAS_MPI)
using MPI_Comm = void*;
#define MPI_COMM_NULL nullptr
#endif
static MPI_Comm comm = MPI_COMM_NULL;

static int loggingEnabled = 0x00;

static std::string timerLogFile = "timer.csv";

using eventLogType = std::list<impl::Event>;
using threadMapType = std::unordered_map<std::thread::id, eventLogType>;

static eventLogType eventLog;
static threadMapType activeEvents;
static std::mutex eventLogMutex;

// memory profiler
static sensei::MemoryProfiler memProf;

// return high res system Time relative to system epoch
static double getSystemTime()
{
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec/1.0e6;
}


// --------------------------------------------------------------------------
Event::Event() : Time{0,0,0}, NumBytes(-1ll), Depth(0),
  Tid(std::this_thread::get_id())
{
}

//-----------------------------------------------------------------------------
void Event::ToStream(std::ostream &str) const
{
#if defined(ENABLE_PROFILER)
  int rank = 0;
#if defined(SENSEI_HAS_MPI)
  int ini = 0, fin = 0;
  MPI_Initialized(&ini);
  MPI_Finalized(&fin);
  if (ini && !fin)
    MPI_Comm_rank(impl::comm, &rank);
#endif
  str << rank << ", " << this->Tid << ", \"" << this->Name << "\", "
    << this->Time[START] << ", " << this->Time[END] << ", "
    << this->Time[DELTA] << ", " << this->NumBytes  << ", "
    << this->Depth << std::endl;
#else
  (void)str;
#endif
}
#endif
}



namespace sensei
{

// ----------------------------------------------------------------------------
void Profiler::SetCommunicator(MPI_Comm comm)
{
#if defined(ENABLE_PROFILER) && defined(SENSEI_HAS_MPI)
  int ok = 0;
  MPI_Initialized(&ok);
  if (ok)
    {
    if (impl::comm != MPI_COMM_NULL)
      MPI_Comm_free(&impl::comm);

    MPI_Comm_dup(comm, &impl::comm);
    }
#else
  (void)comm;
#endif
}

// ----------------------------------------------------------------------------
void Profiler::SetTimerLogFile(const std::string &file)
{
#if defined(ENABLE_PROFILER)
  impl::timerLogFile = file;
#else
  (void)file;
#endif
}

// ----------------------------------------------------------------------------
void Profiler::SetMemProfLogFile(const std::string &file)
{
#if defined(ENABLE_PROFILER)
  impl::memProf.SetFilename(file);
#else
  (void)file;
#endif
}

// ----------------------------------------------------------------------------
void Profiler::SetMemProfInterval(int interval)
{
#if defined(ENABLE_PROFILER)
  impl::memProf.SetInterval(interval);
#else
  (void)interval;
#endif
}

// ----------------------------------------------------------------------------
int Profiler::Validate()
{
  int ierr = 0;
#if defined(ENABLE_PROFILER)
  if (impl::loggingEnabled & 0x01)
    {
#if !defined(NDEBUG)
    impl::threadMapType::iterator tmit = impl::activeEvents.begin();
    impl::threadMapType::iterator tmend = impl::activeEvents.end();
    for (; tmit != tmend; ++tmit)
      {
      unsigned int nLeft = tmit->second.size();
      if (nLeft > 0)
        {
        std::ostringstream oss;
        impl::eventLogType::iterator it = tmit->second.begin();
        impl::eventLogType::iterator end = tmit->second.end();
        for (; it != end; ++it)
          it->ToStream(oss);
        SENSEI_ERROR("Thread " << tmit->first << " has " << nLeft
          << " unmatched active events. " << std::endl
          << oss.str())
        ierr += 1;
        }
      }
#endif
    }
#endif
  return ierr;
}

// ----------------------------------------------------------------------------
int Profiler::ToStream(std::ostream &os)
{
#if defined(ENABLE_PROFILER)
  if (impl::loggingEnabled & 0x01)
    {
    // serialize the logged events in CSV format
    os.precision(std::numeric_limits<double>::digits10 + 2);
    os.setf(std::ios::scientific, std::ios::floatfield);

    // not locking this as it's intended to be accessed only from the main
    // thread, and all other threads are required to be finished by now
    impl::eventLogType::iterator iter = impl::eventLog.begin();
    impl::eventLogType::iterator end = impl::eventLog.end();

    for (; iter != end; ++iter)
      iter->ToStream(os);
    }
#else
  (void)os;
#endif
  return 0;
}

// ----------------------------------------------------------------------------
int Profiler::Initialize()
{
#if defined(ENABLE_PROFILER)

  int rank = 0;
#if defined(SENSEI_HAS_MPI)
  int ok = 0;
  MPI_Initialized(&ok);
  if (ok)
    {
    // always use isolated comm space
    if (impl::comm == MPI_COMM_NULL)
      Profiler::SetCommunicator(MPI_COMM_WORLD);

    impl::memProf.SetCommunicator(impl::comm);

    MPI_Comm_rank(impl::comm, &rank);
    }
#endif

  // look for overrides in the environment
  char *tmp = nullptr;
  if ((tmp = getenv("PROFILER_ENABLE")))
    impl::loggingEnabled = atoi(tmp);

  if ((tmp = getenv("PROFILER_LOG_FILE")))
    impl::timerLogFile = tmp;

  if ((tmp = getenv("MEMPROF_LOG_FILE")))
    impl::memProf.SetFilename(tmp);

  if ((tmp = getenv("MEMPROF_INTERVAL")))
    impl::memProf.SetInterval(atof(tmp));

  if (impl::loggingEnabled & 0x02)
    impl::memProf.Initialize();

  // report what options are in use
  if ((rank == 0) && impl::loggingEnabled)
    std::cerr << "Profiler configured with Event logging "
      << (impl::loggingEnabled & 0x01 ? "enabled" : "disabled")
      << " and memory logging " << (impl::loggingEnabled & 0x02 ? "enabled" : "disabled")
      << ", timer log file \"" << impl::timerLogFile
      << "\", memory profiler log file \"" << impl::memProf.GetFilename()
      << "\", sampling interval " << impl::memProf.GetInterval()
      << " seconds" << std::endl;
#endif
  return 0;
}

// ----------------------------------------------------------------------------
int Profiler::WriteMpiIo(MPI_Comm comm, const char *fileName,
  const std::string &str)
{
#if defined(ENABLE_PROFILER) && defined(SENSEI_HAS_MPI)
  if (impl::loggingEnabled & 0x01)
    {
    // compute the file offset
    long nBytes = str.size();

    int rank = 0;
    int nRanks = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nRanks);

    std::vector<long> gsizes(nRanks);
    gsizes[rank] = nBytes;

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
      gsizes.data(), 1, MPI_LONG, impl::comm);

    long offset = 0;
    for (int i = 0; i < rank; ++i)
      offset += gsizes[i];

    long fileSize = 0;
    for (int i = 0; i < nRanks; ++i)
      fileSize += gsizes[i];

    // write the buffer
    MPI_File fh;
    MPI_File_open(comm, fileName,
      MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    MPI_File_set_view(fh, offset, MPI_BYTE, MPI_BYTE,
      "native", MPI_INFO_NULL);

    MPI_File_write(fh, str.c_str(), nBytes,
      MPI_BYTE, MPI_STATUS_IGNORE);

    MPI_File_set_size(fh, fileSize);

    MPI_File_close(&fh);
    }
  return 0;
#else
  (void)comm;
  (void)fileName;
  (void)str;
#endif
  return -1;
}

// ----------------------------------------------------------------------------
int Profiler::Flush()
{
#if defined(ENABLE_PROFILER)
  std::ostringstream oss;
  Profiler::ToStream(oss);
  Profiler::WriteCStdio(impl::timerLogFile.c_str(), "a", oss.str());
  Profiler::Validate();
  impl::eventLog.clear();
#endif
  return 0;
}

// ----------------------------------------------------------------------------
int Profiler::WriteCStdio(const char *fileName, const char *mode,
  const std::string &str)
{
#if defined(ENABLE_PROFILER)
  if (impl::loggingEnabled & 0x01)
    {
    FILE *fh = fopen(fileName, mode);
    if (!fh)
      {
      const char *estr = strerror(errno);
      SENSEI_ERROR("Failed to open \""
        << fileName << "\" " << estr)
      return -1;
      }

    long nBytes = str.size();
    long nwritten = fwrite(str.c_str(), 1, nBytes, fh);
    if (nwritten != nBytes)
      {
      const char *estr = strerror(errno);
      SENSEI_ERROR("Failed to write " << nBytes << " bytes. " << estr)
      return -1;
      }

    fclose(fh);
    }
#else
  (void)fileName;
  (void)mode;
  (void)str;
#endif
  return 0;
}

// ----------------------------------------------------------------------------
int Profiler::Finalize()
{
#if defined(ENABLE_PROFILER)
  int ok = 0;
#if defined(SENSEI_HAS_MPI)
  MPI_Initialized(&ok);
#endif

  if (impl::loggingEnabled & 0x01)
    {
    int rank = 0;
#if defined(SENSEI_HAS_MPI)
    if (ok)
      MPI_Comm_rank(impl::comm, &rank);
#endif

    // serialize the logged events in CSV format
    std::ostringstream oss;

    if (rank == 0)
      oss << "# rank, thread, Name, start Time, end Time, delta, Depth" << std::endl;

    Profiler::ToStream(oss);

    // free up resources
    impl::eventLog.clear();

    if (ok)
      Profiler::WriteMpiIo(impl::comm, impl::timerLogFile.c_str(), oss.str());
    else
      Profiler::WriteCStdio(impl::timerLogFile.c_str(), "w", oss.str());
    }

  // output the memory use profile and clean up resources
  if (impl::loggingEnabled & 0x02)
    impl::memProf.Finalize();

  // free up other resources
#if defined(SENSEI_HAS_MPI)
  if (ok)
    MPI_Comm_free(&impl::comm);
#endif
#endif
  return 0;
}

//-----------------------------------------------------------------------------
bool Profiler::Enabled()
{
#if defined(ENABLE_PROFILER)
  std::lock_guard<std::mutex> lock(impl::eventLogMutex);
  return impl::loggingEnabled & 0x01;
#else
  return false;
#endif
}

//-----------------------------------------------------------------------------
void Profiler::Enable(int arg)
{
#if defined(ENABLE_PROFILER)
  std::lock_guard<std::mutex> lock(impl::eventLogMutex);
  impl::loggingEnabled = arg;
#else
  (void)arg;
#endif
}

//-----------------------------------------------------------------------------
void Profiler::Disable()
{
#if defined(ENABLE_PROFILER)
  std::lock_guard<std::mutex> lock(impl::eventLogMutex);
  impl::loggingEnabled = 0x00;
#endif
}

//-----------------------------------------------------------------------------
int Profiler::StartEvent(const char* eventname, long long nbytes)
{
#if defined(ENABLE_PROFILER)
  if (impl::loggingEnabled & 0x01)
    {
    impl::Event evt;
    evt.Name = eventname;
    evt.Time[impl::Event::START] = impl::getSystemTime();
    evt.NumBytes = nbytes;

    std::lock_guard<std::mutex> lock(impl::eventLogMutex);
    impl::activeEvents[evt.Tid].push_back(evt);
    }
#else
  (void)eventname;
  (void)nbytes;
#endif
  return 0;
}

//-----------------------------------------------------------------------------
int Profiler::EndEvent(const char* eventname, long long nbytes)
{
#if defined(ENABLE_PROFILER)
  if (impl::loggingEnabled & 0x01)
    {
    // get end Time
    double endTime = impl::getSystemTime();

    // get this thread's Event log
    std::thread::id tid = std::this_thread::get_id();

    std::lock_guard<std::mutex> lock(impl::eventLogMutex);
    impl::threadMapType::iterator iter = impl::activeEvents.find(tid);
    if (iter == impl::activeEvents.end())
      {
      SENSEI_ERROR("failed to end Event \"" << eventname
        << "\" thread  " << tid << " has no events")
      return -1;
      }

    impl::Event evt(std::move(iter->second.back()));
    iter->second.pop_back();

#ifdef NDEBUG
    (void)eventname;
#else
    if (strcmp(eventname, evt.Name.c_str()) != 0)
      {
      SENSEI_ERROR("Mismatched startEvent/endEvent. Expecting: '"
        << evt.Name << "' Got: '" << eventname << "'")
      abort();
      }
#endif
    evt.Time[impl::Event::END] = endTime;
    evt.Time[impl::Event::DELTA] = endTime - evt.Time[impl::Event::START];
    evt.NumBytes = nbytes;
    evt.Depth = iter->second.size();

    impl::eventLog.emplace_back(std::move(evt));
    }
#else
  (void)eventname;
  (void)nbytes;
#endif
  return 0;
}

}
