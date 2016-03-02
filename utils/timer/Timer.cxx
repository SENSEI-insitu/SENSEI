#include "Timer.h"
#include "TimerC.h"

#include <sys/time.h>
#include <iostream>
#include <cstdlib>
#include <cstring>

#include <map>
#include <list>

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
    std::string Name;
    double Duration;
    Event() : Duration(-1.0) {}
    bool IsValid() const
      { return this->Duration > -1.0; }

    void PrintLog(std::ostream& stream, Indent indent) const
      {
      stream << indent << this->Name.c_str() << " = " << this->Duration << endl;
      for (std::list<Event>::const_iterator iter = this->SubEvents.begin();
        iter != this->SubEvents.end(); ++iter)
        {
        iter->PrintLog(stream, indent.GetNextIndent());
        }
      }
    std::list<Event> SubEvents;
    };

  static std::list<Event> GlobalEvents;
  static std::list<Event> Mark;
  static bool LoggingEnabled = true;

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
void MarkStartEvent(const char* eventname)
{
  if (impl::LoggingEnabled)
    {
    impl::Event evt;
    evt.Name = eventname? eventname : "(none)";
    evt.Duration = impl::get_mark();

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

    evt.Duration = impl::get_mark() - evt.Duration;
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
void MarkStartTimeStep(int timestep, double Duration)
{
}

//-----------------------------------------------------------------------------
void MarkEndTimeStep()
{
}

//-----------------------------------------------------------------------------
void PrintLog(std::ostream& stream)
{
  impl::PrintLog(stream, impl::Indent());
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
  timer::PrintLog(std::cout);
}

