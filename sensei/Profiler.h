#ifndef sensei_Profiler_h
#define sensei_Profiler_h

#include "senseiConfig.h"
#define SENSEI_HAS_MPI

#include <string>
#include <thread>
#include <ostream>
#include <mpi.h>

namespace sensei
{

// A class containing methods managing memory and time profiling
// Each timed event logs rank, event name, start and end time, and
// duration.
class Profiler
{
public:
  // Initialize logging from environment variables, and/or the timer
  // API below. This is a collective call with respect to the timer's
  // communicator.
  //
  // If found in the environment the following variable override the
  // the current settings
  //
  //   PROFILER_ENABLE     : bit mask turns on or off logging,
  //               0x01 -- event profiling enabled
  //               0x02 -- memory profiling enabled
  //   PROFILER_LOG_FILE   : path to write timer log to
  //   MEMPROF_LOG_FILE    : path to write memory profiler log to
  //   MEMPROF_INTERVAL    : number of seconds between memory recordings
  //
  static int Initialize();

  // Finalize the log. this is where logs are written and cleanup occurs.
  // All processes in the communicator must call, and it must be called
  // prior to MPI_Finalize.
  static int Finalize();

  // this can occur after MPI_Finalize. It should only be called by rank 0.
  // Any remaining events will be appeneded to the log file. This is necessary
  // to time MPI_Initialize/Finalize and log associated I/O.
  static int Flush();

  // Sets the communicator for MPI calls. This must be called prior to
  // initialization.
  // default value: MPI_COMM_NULL
  static void SetCommunicator(MPI_Comm comm);

  // Sets the path to write the timer log to
  // overriden by PROFILER_LOG_FILE environment variable
  // default value; Timer.csv
  static void SetTimerLogFile(const std::string &fileName);

  // Sets the path to write the timer log to
  // overriden by MEMPROF_LOG_FILE environment variable
  // default value: MemProfLog.csv
  static void SetMemProfLogFile(const std::string &fileName);

  // Sets the number of seconds in between memory use recordings
  // overriden by MEMPROF_INTERVAL environment variable.
  static void SetMemProfInterval(int interval);

  // Enable/Disable logging. Overriden by PROFILER_ENABLE environment
  // variable. In the default format a CSV file is generated capturing each
  // ranks timer events. default value: disabled
  static void Enable(int arg = 0x03);
  static void Disable();

  // return true if loggin is enabled.
  static bool Enabled();

  // @brief Log start of an event.
  //
  // This marks the beginning of a event that must be logged.  The @arg
  // eventname must match when calling endEvent() to mark the end of the
  // event.
  static int StartEvent(const char *eventname, long long nbytes=-1ll);

  // @brief Log end of a log-able event.
  //
  // This marks the end of a event that must be logged.  The @arg eventname
  // must match when calling endEvent() to mark the end of the event.
  static int EndEvent(const char *eventname, long long nbytes=-1ll);

  // write contents of the string to the file.
  static int WriteCStdio(const char *fileName, const char *mode,
     const std::string &str);

  // write contents of the string to the file in rank order
  // the file is truncated first or created
  static int WriteMpiIo(MPI_Comm comm, const char *fileName,
    const std::string &str);

  // checks to see if all active events have been ended.
  // will report errors if not
  static int Validate();

  // setnd the current contents of the log to the stream
  static int ToStream(std::ostream &os);
};

// TimeEvent -- A helper class that times it's life.
// A timer event is created that starts at the object's construction and ends
// at its destruction. The pointer to the event name must be valid throughout
// the objects life.
template <int bufferSize>
class TimeEvent
{
public:
  // logs an event named
  // <className>::<method> port=<p>
  TimeEvent(const char *className,
    const char *method, int port) : Eventname(Buffer)
  {
    snprintf(Buffer, bufferSize, "%s::%s port=%d",
      className, method, port);
    Profiler::StartEvent(Eventname);
  }

  // logs an event named
  // <className>::<method>
  TimeEvent(const char *className,
    int nThreads, int nReqs) : Eventname(Buffer)
  {
    snprintf(Buffer, bufferSize,
      "%s threadPool process nThreads=%d nReqs=%d",
      className, nThreads, nReqs);
    Profiler::StartEvent(Eventname);
  }


  // logs an event named:
  // <className>::<method>
  TimeEvent(const char *className,
    const char *method) : Eventname(Buffer)
  {
    Buffer[0] = '\0';
    strcat(Buffer, className);
    strcat(Buffer, method);
    Profiler::StartEvent(Eventname);
  }

  // logs an event named:
  // <name>
  TimeEvent(const char *name) : Eventname(name)
  { Profiler::StartEvent(name); }

  ~TimeEvent()
  { Profiler::EndEvent(this->Eventname); }

private:
  char Buffer[bufferSize];
  const char *Eventname;
};

#if defined(SENSEI_ENABLE_PROFILER)
#define SENSEI_PROFILE_PIPELINE(_n, _alg, _meth, _port, _code)  \
{                                 \
  TimeEvent<_n> event(_alg->GetClassName(),           \
     _meth, _port);                       \
  _code                             \
}

#define SENSEI_PROFILE_METHOD(_n, _alg, _meth, _code)       \
{                                 \
  TimeEvent<_n>                         \
    event(_alg->GetClassName(), "::" _meth);          \
  _code                             \
}

#define SENSEI_PROFILE_THREAD_POOL(_n, _alg, _nt, _nr, _code)   \
{                                 \
  TimeEvent<_n>                         \
    event(_alg->GetClassName(), _nt, _nr);          \
  _code                             \
}
#else
#define SENSEI_PROFILE_PIPELINE(_n, _alg, _meth, _port, _code)  \
{                                 \
  _code                             \
}

#define SENSEI_PROFILE_METHOD(_n, _alg, _meth, _code)       \
{                                 \
  _code                             \
}

#define SENSEI_PROFILE_THREAD_POOL(_n, _alg, _nt, _nr, _code)   \
{                                 \
  _code                             \
}
#endif

}

#endif
