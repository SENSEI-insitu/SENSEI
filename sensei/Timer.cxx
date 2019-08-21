#include "Timer.h"
#include "MemoryProfiler.h"

#include <vtksys/SystemInformation.hxx>

#include <sys/time.h>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdint.h>
#include <strings.h>
#include <cstdlib>
#include <cstdlib>

#include <map>
#include <list>
#include <vector>
#include <iomanip>
#include <limits>

// **************************************************************************
extern "C"
void MPIErrorHandler(MPI_Comm *comm, int *code, ...)
{
  int rank = 0;
  MPI_Comm_rank(*comm, &rank);

  int estrLen = 0;
  char estr[MPI_MAX_ERROR_STRING] = {'\0'};
  MPI_Error_string(*code, estr, &estrLen);

  std::ostringstream oss;
  oss
    << "+--------------------------------------------------------+" << std::endl
    << "MPI rank " << rank  << " encountered error " << *code << std::endl
    << std::endl
    << estr << std::endl
    << std::endl
    << vtksys::SystemInformation::GetProgramStack(3,0)
    << "+--------------------------------------------------------+" << std::endl;

  std::cerr << oss.str() << std::endl;

  MPI_Abort(*comm, -1);
}

namespace sensei
{
//namespace timer
//{
namespace impl
{
struct Event;

// timer controls and data
static MPI_Comm Comm = MPI_COMM_NULL;
static bool LoggingEnabled = false;
static bool Summarize = false;
static int SummaryModulus = 100000000;
static std::string TimerLogFile = "Timer.csv";

static std::list<Event> Mark;
static std::list<Event> GlobalEvents;

static int ActiveTimeStep = -1;
static double ActiveTime = 0.0;

// memory profiler
static MemoryProfiler MemProf;



// return high res system time relative to system epoch
static double getSystemTime()
{
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec/1.0e6;
}

// helper for pretty printing events
struct Indent
{
  int Count;

  explicit Indent(int indent=0): Count(indent) {}

  Indent GetNextIndent() const
  { return Indent(this->Count+1); }
};

std::ostream &operator<<(std::ostream &os, const Indent &i)
{
  for (int cc=0; cc < i.Count; ++cc)
    os << "  ";

  return os;
}

// container for data captured in a timing event
struct Event
{
  Event();

  bool Empty() const
  { return this->Count < 1; }

  // merges or computes summary of the two events
  virtual void AddToSummary(const Event& other);

  // prints the log in a human readbale format
  virtual void PrettyPrint(std::ostream& stream, Indent indent) const;

  // serializes the event in CSV format into the stream.
  virtual void ToStream(std::ostream &str) const;

  enum { START=0, END=1, DELTA=2 };     // time record fields
  enum { MIN=0, MAX=1, SUM=2, SIZE=2 }; // time/I/O summary fields

  // user provided identifier for the record
  std::string Name;

  // for nesting of events
  std::list<Event> SubEvents;

  // event duration, initially start time, end time and duration
  // are recorded. when summarizing this contains min,max and sum
  // of the summariezed set of events
  double Time[3];

  // number of bytes if data was moved, read, or written
  long long NumBytes[3];

  // number of events in the summary, or 1 for a single event
  int Count;
};

// --------------------------------------------------------------------------
Event::Event() : Count(0)
{
  this->Time[0] = this->Time[1] = this->Time[2] = 0;
  this->NumBytes[0] = this->NumBytes[1] = this->NumBytes[2] = 0ll;
}

// --------------------------------------------------------------------------
void Event::AddToSummary(const Event& other)
{
  // convert or add to summary
  // first three cases handle conversion of this or other or both
  // into the summary form. the last case merges the summaries
  if ((this->Count == 1) && (other.Count == 1))
    {
    this->Time[MIN] = std::min(this->Time[DELTA], other.Time[DELTA]);
    this->Time[MAX] = std::max(this->Time[DELTA], other.Time[DELTA]);
    this->Time[SUM] = this->Time[DELTA] + other.Time[DELTA];

    this->NumBytes[MIN] = std::min(this->NumBytes[SIZE], other.NumBytes[SIZE]);
    this->NumBytes[MAX] = std::max(this->NumBytes[SIZE], other.NumBytes[SIZE]);
    this->NumBytes[SUM] = this->NumBytes[SIZE] + other.NumBytes[SIZE];
    }
  else if (this->Count == 1)
    {
    this->Time[MIN] = std::min(this->Time[DELTA], other.Time[MIN]);
    this->Time[MAX] = std::max(this->Time[DELTA], other.Time[MAX]);
    this->Time[SUM] = this->Time[DELTA] + other.Time[SUM];

    this->NumBytes[MIN] = std::min(this->NumBytes[SIZE], other.NumBytes[MIN]);
    this->NumBytes[MAX] = std::max(this->NumBytes[SIZE], other.NumBytes[MAX]);
    this->NumBytes[SUM] = this->NumBytes[SIZE] + other.NumBytes[SUM];
    }
  else if (other.Count == 1)
    {
    this->Time[MIN] = std::min(this->Time[MIN], other.Time[DELTA]);
    this->Time[MAX] = std::max(this->Time[MAX], other.Time[DELTA]);
    this->Time[SUM] = this->Time[SUM] + other.Time[DELTA];

    this->NumBytes[MIN] = std::min(this->NumBytes[MIN], other.NumBytes[SIZE]);
    this->NumBytes[MAX] = std::max(this->NumBytes[MAX], other.NumBytes[SIZE]);
    this->NumBytes[SUM] = this->NumBytes[SUM] + other.NumBytes[SIZE];
    }
  else
    {
    this->Time[MIN] = std::min(this->Time[MIN], other.Time[MIN]);
    this->Time[MAX] = std::max(this->Time[MAX], other.Time[MAX]);
    this->Time[SUM] += other.Time[SUM];

    this->NumBytes[MIN] = std::min(this->NumBytes[MIN], other.NumBytes[MIN]);
    this->NumBytes[MAX] = std::max(this->NumBytes[MAX], other.NumBytes[MAX]);
    this->NumBytes[SUM] += other.NumBytes[SUM];
    }

  this->Count += other.Count;

  // process nested events
  if (this->SubEvents.size() == other.SubEvents.size())
    {
    auto it = this->SubEvents.begin();
    auto end = this->SubEvents.end();

    auto oit = other.SubEvents.begin();

    for (; it != end; ++it, ++oit)
      it->AddToSummary(*oit);
    }
}

//-----------------------------------------------------------------------------
void Event::ToStream(std::ostream &str) const
{
#ifndef NDEBUG
  if (this->Empty())
    {
    std::cerr << "Empty event detected" << std::endl;
    abort();
    }
#endif

  int rank = 0;
  MPI_Comm_rank(impl::Comm, &rank);

  str << rank << ", \"" << this->Name << "\", " << this->Time[START]
    << ", " << this->Time[END] << ", " << this->Time[DELTA] << ", "
    << this->NumBytes[SIZE] << std::endl;

  // handle nested events
  auto iter = this->SubEvents.begin();
  auto end = this->SubEvents.end();

  for (; iter != end; ++iter)
    iter->ToStream(str);
}


//-----------------------------------------------------------------------------
void Event::PrettyPrint(std::ostream& stream, Indent indent) const
{
#ifndef NDEBUG
  if (this->Empty())
    {
    std::cerr << "Empty event detected" << std::endl;
    abort();
    }
#endif

  if (this->Count == 1)
    {
    stream << indent << this->Name
       << " = (" << this->Time[DELTA] <<  " s, "
       << this->NumBytes[SIZE] << " b )" << std::endl;
    }
  else
    {
    stream << indent << this->Name << " = ( min: "
      << this->Time[MIN] << " s, max: " << this->Time[MAX]
      << " s, avg:" << this->Time[SUM]/this->Count << " s min: "
      << this->NumBytes[MIN] << " b, max: " << this->NumBytes[MAX]
      << " b, sum: " << this->NumBytes[SUM] << " b, avg: "
      << this->NumBytes[SUM]/this->Count << " )" << std::endl;
    }

  // handle nested events
  auto iter = this->SubEvents.begin();
  auto  end = this->SubEvents.end();

  for (; iter != end; ++iter)
    iter->PrettyPrint(stream, indent.GetNextIndent());
}

//-----------------------------------------------------------------------------
void PrintSummary(std::ostream& stream, Indent indent)
{
  auto iter = GlobalEvents.begin();
  auto end = GlobalEvents.end();

  for (; iter != end; ++iter)
    iter->PrettyPrint(stream, indent);
}

//-----------------------------------------------------------------------------
void PrintSummary(std::ostream& stream)
{
  if (!impl::LoggingEnabled)
    return;

  int nprocs = 1;
  int rank = 0;

  MPI_Comm_size(impl::Comm, &nprocs);
  MPI_Comm_rank(impl::Comm, &rank);

  std::ostringstream tmp;

  std::ostream &output = (rank == 0)? stream : tmp;
  if (rank == 0)
    output << "\n"
           << "=================================================================\n"
           << "  Time/Memory log (rank: 0) \n"
           << "  -------------------------------------------------------------\n";

  if (rank % impl::SummaryModulus == 0)
    impl::PrintSummary(output, impl::Indent());

  if (rank == 0)
    output << "=================================================================\n";


  if (nprocs == 1)
    return;

  std::string data = tmp.str();
  int mylength = static_cast<int>(data.size()) + 1;
  std::vector<int> all_lengths(nprocs);
  MPI_Gather(&mylength, 1, MPI_INT, &all_lengths[0], 1, MPI_INT, 0, impl::Comm);
  if (rank == 0)
    {
    std::vector<int> recv_offsets(nprocs);
    for (int cc=1; cc < nprocs; cc++)
      {
      recv_offsets[cc] = recv_offsets[cc-1] + all_lengths[cc-1];
      }
    char* recv_buffer = new char[recv_offsets[nprocs-1] + all_lengths[nprocs-1]];
    MPI_Gatherv(const_cast<char*>(data.c_str()), mylength, MPI_CHAR,
      recv_buffer, &all_lengths[0], &recv_offsets[0], MPI_CHAR, 0, impl::Comm);

    for (int cc=1; cc < nprocs; cc++)
      {
      if (cc % impl::SummaryModulus == 0)
        {
        output << "\n"
               << "=================================================================\n"
               << "  Time/Memory log (rank: " << cc << ") \n"
               << "  -------------------------------------------------------------\n";
        output << (recv_buffer + recv_offsets[cc]);
        output << "=================================================================\n";
        }
      }

    delete []recv_buffer;
    }
  else
    {
    MPI_Gatherv(const_cast<char*>(data.c_str()), mylength, MPI_CHAR,
      NULL, NULL, NULL, MPI_CHAR, 0, impl::Comm);
    }
}

}

// ----------------------------------------------------------------------------
void Timer::SetCommunicator(MPI_Comm comm)
{
  if (impl::Comm != MPI_COMM_NULL)
    MPI_Comm_free(&impl::Comm);

  // install an error handler
  MPI_Errhandler meh;
  MPI_Comm_create_errhandler(MPIErrorHandler, &meh);
  MPI_Comm_set_errhandler(comm, meh);

  MPI_Comm_dup(comm, &impl::Comm);
}

// ----------------------------------------------------------------------------
void Timer::SetSummaryModulus(int modulus)
{
  impl::SummaryModulus = modulus;
}

// ----------------------------------------------------------------------------
void Timer::SetTimerLogFile(const std::string &file)
{
  impl::TimerLogFile = file;
}

// ----------------------------------------------------------------------------
void Timer::SetMemProfLogFile(const std::string &file)
{
  impl::MemProf.SetFileName(file);
}

// ----------------------------------------------------------------------------
void Timer::SetMemProfInterval(int interval)
{
  impl::MemProf.SetInterval(interval);
}

// ----------------------------------------------------------------------------
void Timer::Initialize()
{
  // always use isolated comm space
  if (impl::Comm == MPI_COMM_NULL)
    Timer::SetCommunicator(MPI_COMM_WORLD);

  impl::MemProf.SetCommunicator(impl::Comm);

  // look for overrides in the environment
  char *tmp = nullptr;
  if ((tmp = getenv("TIMER_ENABLE")))
    {
    impl::LoggingEnabled = atoi(tmp);
    impl::Summarize = false;
    }

  if ((tmp = getenv("TIMER_ENABLE_SUMMARY")))
    {
    impl::LoggingEnabled = atoi(tmp);
    impl::Summarize = impl::LoggingEnabled;
    }

  if ((tmp = getenv("TIMER_SUMMARY_MODULUS")))
    {
    impl::SummaryModulus = atoi(tmp);
    }

  if ((tmp = getenv("TIMER_LOG_FILE")))
    {
    impl::TimerLogFile = tmp;
    }

  if ((tmp = getenv("MEMPROF_LOG_FILE")))
    {
    impl::MemProf.SetFileName(tmp);
    }

  if ((tmp = getenv("MEMPROF_INTERVAL")))
    {
    impl::MemProf.SetInterval(atof(tmp));
    }

  if (impl::LoggingEnabled && !impl::Summarize)
    {
    impl::MemProf.Initialize();
    }

  // enable diagnostic info about crashes
  vtksys::SystemInformation::SetStackTraceOnError(1);

  // report what options are in use
  int rank = 0;
  MPI_Comm_rank(impl::Comm, &rank);

  if ((rank == 0) && impl::LoggingEnabled)
    std::cerr << "Timer configured with logging "
      << (impl::LoggingEnabled ? "enabled" : "disabled")
      << ", summarize events " << (impl::Summarize ? "on" : "off")
      << ", summary modulus " << impl::SummaryModulus << ", " << std::endl
      << "timer log file \"" << impl::TimerLogFile
      << "\", memory profiler log file \"" << impl::MemProf.GetFileName()
      << "\", sampling interval " << impl::MemProf.GetInterval()
      << " seconds" << std::endl;
}

// ----------------------------------------------------------------------------
void Timer::Finalize()
{
  if (impl::LoggingEnabled)
    {
    // output timer log
    if (impl::Summarize)
      {
      // pretty print to the termninal
      impl::PrintSummary(std::cerr);
      }
    else
      {
      int rank = 0;
      int nRanks = 1;

      MPI_Comm_rank(impl::Comm, &rank);
      MPI_Comm_size(impl::Comm, &nRanks);

      // serialize the logged events in CSV format
      std::ostringstream oss;
      oss.precision(std::numeric_limits<double>::digits10 + 2);
      oss.setf(std::ios::scientific, std::ios::floatfield);

      if (rank == 0)
        oss << "# rank, name, start time, end time, duration, num bytes" << std::endl;

      std::list<impl::Event>::iterator iter = impl::GlobalEvents.begin();
      std::list<impl::Event>::iterator end = impl::GlobalEvents.end();

      for (; iter != end; ++iter)
        iter->ToStream(oss);

      // compute the file offset
      long nBytes = oss.str().size();

      std::vector<long> gsizes(nRanks);
      gsizes[rank] = nBytes;

      MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
        gsizes.data(), 1, MPI_LONG, impl::Comm);

      long offset = 0;
      for (int i = 0; i < rank; ++i)
        offset += gsizes[i];

      long fileSize = 0;
      for (int i = 0; i < nRanks; ++i)
        fileSize += gsizes[i];

      // write the buffer
      MPI_File fh;
      MPI_File_open(impl::Comm, impl::TimerLogFile.c_str(),
        MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

      MPI_File_set_view(fh, offset, MPI_BYTE, MPI_BYTE,
         "native", MPI_INFO_NULL);

      MPI_File_write(fh, oss.str().c_str(), nBytes,
        MPI_BYTE, MPI_STATUS_IGNORE);

      MPI_File_set_size(fh, fileSize);

      MPI_File_close(&fh);
      }

    // free up resources
    impl::GlobalEvents.clear();
    impl::Mark.clear();

    // output the memory use profile and clean up resources
    if (impl::LoggingEnabled && !impl::Summarize)
      {
      impl::MemProf.Finalize();
      }
    }

  // free up other resources
  MPI_Comm_free(&impl::Comm);
}




//-----------------------------------------------------------------------------
bool Timer::Enabled()
{
  return impl::LoggingEnabled;
}

//-----------------------------------------------------------------------------
void Timer::Enable(bool shortFormat)
{
  impl::LoggingEnabled = true;
  impl::Summarize = shortFormat;
}

//-----------------------------------------------------------------------------
void Timer::Disable()
{
  impl::LoggingEnabled = false;
  impl::Summarize = false;
}

//-----------------------------------------------------------------------------
void Timer::MarkStartEvent(const char *eventname, long long numBytes)
{
  if (impl::LoggingEnabled)
    {
    impl::Event evt;
    evt.Name = eventname;
    evt.Time[impl::Event::START] = impl::getSystemTime();
    evt.NumBytes[impl::Event::SIZE] = numBytes;
    evt.Count = 1;

    impl::Mark.push_back(evt);
    }
}

//-----------------------------------------------------------------------------
void Timer::MarkStartEvent(const char* eventname)
{
  if (impl::LoggingEnabled)
    {
    impl::Event evt;
    evt.Name = eventname;
    evt.Time[impl::Event::START] = impl::getSystemTime();
    evt.NumBytes[impl::Event::SIZE] = 0ll;
    evt.Count = 1;

    impl::Mark.push_back(evt);
    }
}

//-----------------------------------------------------------------------------
void Timer::MarkEndEvent(const char* eventname)
{
  if (impl::LoggingEnabled)
    {
    impl::Event evt = impl::Mark.back();

#ifdef NDEBUG
    (void)eventname;
#else
    if (!eventname)
      {
      std::cerr << "null eventname detected. events must be named." << std::endl;
      abort();
      }

    if (strcmp(eventname, evt.Name.c_str()) != 0)
      {
      std::cerr << "Mismatched MarkStartEvent/MarkEndEvent. Expecting: '"
        << evt.Name.c_str() << "' Got: '" << eventname << "'" << std::endl;
      abort();
      }
#endif

    evt.Time[impl::Event::END] = impl::getSystemTime();
    evt.Time[impl::Event::DELTA] = evt.Time[impl::Event::END] - evt.Time[impl::Event::START];

    impl::Mark.pop_back();

    // handle event nesting
    if (impl::Mark.empty())
      {
      impl::GlobalEvents.push_back(evt);
      }
    else
      {
      impl::Mark.back().SubEvents.push_back(evt);
      }
    }
}

//-----------------------------------------------------------------------------
void Timer::MarkEndEvent(const char *eventname, long long numBytes)
{
  if (impl::LoggingEnabled)
    {
    impl::Event evt = impl::Mark.back();

#ifdef NDEBUG
    (void)eventname;
#else
    if (!eventname)
      {
      std::cerr << "null eventname detected. events must be named." << std::endl;
      abort();
      }

    if (strcmp(eventname, evt.Name.c_str()) != 0)
      {
      std::cerr << "Mismatched MarkStartEvent/MarkEndEvent. Expecting: '"
        << evt.Name.c_str() << "' Got: '" << eventname << "'" << std::endl;
      abort();
      }
#endif

    evt.Time[impl::Event::END] = impl::getSystemTime();
    evt.Time[impl::Event::DELTA] = evt.Time[impl::Event::END] - evt.Time[impl::Event::START];
    evt.NumBytes[impl::Event::SIZE] = numBytes;

    impl::Mark.pop_back();

    // handle event nesting
    if (impl::Mark.empty())
      {
      impl::GlobalEvents.push_back(evt);
      }
    else
      {
      impl::Mark.back().SubEvents.push_back(evt);
      }
    }
}

//-----------------------------------------------------------------------------
void Timer::MarkStartTimeStep(int timestep, double time)
{
  impl::ActiveTimeStep = timestep;
  impl::ActiveTime = time;

  std::ostringstream mk;
  mk << "timestep: " << impl::ActiveTimeStep << " time: " << impl::ActiveTime;
  MarkStartEvent(mk.str().c_str());
}

//-----------------------------------------------------------------------------
void Timer::MarkEndTimeStep()
{
  std::ostringstream mk;
  mk << "timestep: " << impl::ActiveTimeStep << " time: " << impl::ActiveTime;
  MarkEndEvent(mk.str().c_str());

  std::list<impl::Event> &activeEventList =
    impl::Mark.empty() ? impl::GlobalEvents : impl::Mark.back().SubEvents;

  // merge with previous timestep.
  if (impl::Summarize && (activeEventList.size() >= 2))
    {
    std::list<impl::Event>::reverse_iterator iter = activeEventList.rbegin();
    impl::Event& cur = *iter;
    ++iter;

    impl::Event& prev = *iter;
    if (strncmp(prev.Name.c_str(), "timestep:", 9) == 0)
      {
      prev.AddToSummary(cur);

      std::ostringstream summary_label;
      summary_label << "timestep: (summary over " << prev.Count << " timesteps)";
      prev.Name = summary_label.str();
      activeEventList.pop_back();
      }
    }

  impl::ActiveTimeStep = -1;
  impl::ActiveTime = 0.0;
}

}
//}
