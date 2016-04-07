#include "Timer.h"
#include "TimerC.h"

#include <sys/time.h>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdint.h>
#include <strings.h>

#include <map>
#include <list>
#include <vector>

using std::endl;
using std::cerr;

namespace timer
{
namespace impl
{
  static double get_mark()
    {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec/1.0e6;
    }

  static uint64_t get_vmhwm()
    {
    std::ifstream in("/proc/self/status");
    if (!in)
      {
      return 0;
      }
    std::string line;
    while (std::getline(in, line))
      {
      if (strncmp(line.c_str(), "VmHWM:", 6) == 0)
        {
        std::istringstream iss(line.substr(6));
        uint64_t val;
        iss >> val;
        return val;
        }
      }
    return 0;
    }

  struct Indent
    {
    int Count;
    Indent(int indent=0) : Count(indent) {}
    Indent GetNextIndent() const
      { return Indent(this->Count+1); }
    };

  std::ostream &operator<<(std::ostream &os, const Indent &i)
    {
    for (int cc=0; cc < i.Count; ++cc) { os << "  "; }
    return os;
    }


  struct Event
    {
    enum
      {
      MIN=0,
      MAX=1,
      SUM=2
      };

    std::string Name;
    std::list<Event> SubEvents;

    double Duration[3];
    uint64_t VmHWM[3];
    int Count;

    Event() : Count(0)
    {
    bzero(this->Duration, sizeof(double)*3);
    bzero(this->VmHWM, sizeof(uint64_t)*3);
    }

    /// Add other into this.
    void Add(const Event& other)
      {
      this->Count += other.Count;
      this->Duration[MIN] = std::min(this->Duration[MIN], other.Duration[MIN]);
      this->Duration[MAX] = std::max(this->Duration[MAX], other.Duration[MAX]);
      this->Duration[SUM] += other.Duration[SUM];

      this->VmHWM[MIN] = std::min(this->VmHWM[MIN], other.VmHWM[MIN]);
      this->VmHWM[MAX] = std::max(this->VmHWM[MAX], other.VmHWM[MAX]);
      this->VmHWM[SUM] += other.VmHWM[SUM];

      if (this->SubEvents.size() == other.SubEvents.size())
        {
        std::list<Event>::iterator me_iter = this->SubEvents.begin();
        std::list<Event>::const_iterator other_iter = other.SubEvents.begin();
        for (; me_iter != this->SubEvents.end() && other_iter != other.SubEvents.end();
          ++me_iter, ++other_iter)
          {
          me_iter->Add(*other_iter);
          }
        }
      }

    bool IsValid() const
      { return this->Count > 0; }

    void PrintLog(std::ostream& stream, Indent indent) const
      {
      if (this->Count == 0) { return; }

      if (this->Count == 1)
        {
        stream << indent << this->Name.c_str() << " = ("
               << this->Duration[MIN] <<  "s), (" << this->VmHWM[MIN] << "kB)" << endl;
        }
      else
        {
        stream << indent << this->Name.c_str() << " = "
               << "( min: " << this->Duration[MIN]
               << "s, max: " << this->Duration[MAX]
               << "s, avg:" << this->Duration[SUM] / this->Count << "s ), "
               << "( min: " <<  this->VmHWM[MIN]
               << "kB, max: " << this->VmHWM[MAX]
               << "kB, avg: " << this->VmHWM[SUM] / this->Count << "kB )" << endl;
        }
      for (std::list<Event>::const_iterator iter = this->SubEvents.begin();
        iter != this->SubEvents.end(); ++iter)
        {
        iter->PrintLog(stream, indent.GetNextIndent());
        }
      }
    };

  static std::list<Event> GlobalEvents;

  static std::list<Event> Mark;
  static bool LoggingEnabled = true;
  static bool TrackSummariesOverTime = true;

  static int ActiveTimeStep = -1;
  static double ActiveTime = 0.0;

  void PrintLog(std::ostream& stream, Indent indent)
    {
    for (std::list<Event>::iterator iter = GlobalEvents.begin();
      iter != GlobalEvents.end(); ++iter)
      {
      iter->PrintLog(stream, indent);
      }
    }
}

//-----------------------------------------------------------------------------
void SetLogging(bool val)
{
  impl::LoggingEnabled = val;
}

//-----------------------------------------------------------------------------
bool GetLogging()
{
  return impl::LoggingEnabled;
}

//-----------------------------------------------------------------------------
void SetTrackSummariesOverTime(bool val)
{
  impl::TrackSummariesOverTime = val;
}

//-----------------------------------------------------------------------------
bool GetTrackSummariesOverTime()
{
  return impl::TrackSummariesOverTime;
}

//-----------------------------------------------------------------------------
void MarkStartEvent(const char* eventname)
{
  if (impl::LoggingEnabled)
    {
    impl::Event evt;
    evt.Name = eventname? eventname : "(none)";
    evt.Duration[0] = evt.Duration[1] = evt.Duration[2] = impl::get_mark();
    evt.Count = 1;

    impl::Mark.push_back(evt);
    }
}

//-----------------------------------------------------------------------------
void MarkEndEvent(const char* eventname)
{
  if (impl::LoggingEnabled)
    {
    eventname = eventname? eventname : "(none)";
    impl::Event evt = impl::Mark.back();
    if (strcmp(eventname, evt.Name.c_str()) != 0)
      {
      cerr << "Mismatched MarkStartEvent/MarkEndEvent.\n"
        << "    Expecting: '" << evt.Name.c_str() << "'\n"
        << "    Got: '" << eventname << "'\n"
        << "Aborting for debugging purposes.";
      abort();
      }

    evt.Duration[0] = evt.Duration[1] = evt.Duration[2] = impl::get_mark() - evt.Duration[0];
    evt.VmHWM[0] = evt.VmHWM[1] = evt.VmHWM[2] = impl::get_vmhwm();
    impl::Mark.pop_back();

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
void MarkStartTimeStep(int timestep, double time)
{
  impl::ActiveTimeStep = timestep;
  impl::ActiveTime = time;

  std::ostringstream mk;
  mk << "timestep: " << impl::ActiveTimeStep << " time: " << impl::ActiveTime;
  MarkStartEvent(mk.str().c_str());
}

//-----------------------------------------------------------------------------
void MarkEndTimeStep()
{
  std::ostringstream mk;
  mk << "timestep: " << impl::ActiveTimeStep << " time: " << impl::ActiveTime;
  MarkEndEvent(mk.str().c_str());

  // Try to merge with previous timestep.
  std::list<impl::Event> &activeEventList =
    impl::Mark.empty()? impl::GlobalEvents :
    impl::Mark.back().SubEvents;
  if (activeEventList.size() >= 2 && impl::TrackSummariesOverTime)
    {
    std::list<impl::Event>::reverse_iterator iter = activeEventList.rbegin();
    impl::Event& cur = *iter;
    iter++;

    impl::Event& prev = *iter;
    if (strncmp(prev.Name.c_str(), "timestep:", 9) == 0)
      {
      prev.Add(cur);

      std::ostringstream summary_label;
      summary_label << "timestep: (summary over " << prev.Count << " timesteps)";
      prev.Name = summary_label.str();
      activeEventList.pop_back();
      }
    }

  impl::ActiveTimeStep = -1;
  impl::ActiveTime = 0.0;
}

//-----------------------------------------------------------------------------
void PrintLog(std::ostream& stream, MPI_Comm world, int moduloOutput)
{
  if (!impl::LoggingEnabled)
    {
    return;
    }

  int nprocs, rank;
  MPI_Comm_size(world, &nprocs);
  MPI_Comm_rank(world, &rank);

  if(moduloOutput < 1)
    {
    moduloOutput = 1;
    }

  std::ostringstream tmp;

  std::ostream &output = (rank == 0)? stream : tmp;
  if(rank == 0)
    {
    output << "\n"
           << "=================================================================\n"
           << "  Time/Memory log (rank: 0) \n"
           << "  -------------------------------------------------------------\n";
    }
  if(rank % moduloOutput == 0)
    {
    impl::PrintLog(output, impl::Indent());
    }
  if(rank == 0)
    {
    output << "=================================================================\n";
    }

  if (nprocs == 1)
    {
    return;
    }

  std::string data = tmp.str();
  int mylength = static_cast<int>(data.size()) + 1;
  std::vector<int> all_lengths(nprocs);
  MPI_Gather(&mylength, 1, MPI_INT, &all_lengths[0], 1, MPI_INT, 0, world);
  if (rank == 0)
    {
    std::vector<int> recv_offsets(nprocs);
    for (int cc=1; cc < nprocs; cc++)
      {
      recv_offsets[cc] = recv_offsets[cc-1] + all_lengths[cc-1];
      }
    char* recv_buffer = new char[recv_offsets[nprocs-1] + all_lengths[nprocs-1]];
    MPI_Gatherv(const_cast<char*>(data.c_str()), mylength, MPI_CHAR,
      recv_buffer, &all_lengths[0], &recv_offsets[0], MPI_CHAR, 0, world);

    for (int cc=1; cc < nprocs; cc++)
      {
      if(cc % moduloOutput == 0)
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
      NULL, NULL, NULL, MPI_CHAR, 0, world);
    }

}

}

void TIMER_SetLogging(bool val)
{
  timer::SetLogging(val);
}

void TIMER_MarkStartEvent(const char* val)
{
  timer::MarkStartEvent(val);
}

void TIMER_MarkEndEvent(const char* val)
{
  timer::MarkEndEvent(val);
}

void TIMER_MarkStartTimeStep(int timestep, double time)
{
  timer::MarkStartTimeStep(timestep, time);
}

void TIMER_MarkEndTimeStep()
{
  timer::MarkEndTimeStep();
}
void TIMER_Print()
{
  // FIXME:
  timer::PrintLog(std::cout, MPI_COMM_WORLD);
}

