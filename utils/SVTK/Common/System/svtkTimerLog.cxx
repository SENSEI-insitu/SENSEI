/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTimerLog.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME svtkTimerLog - Maintains timing table for performance analysis
// .SECTION Description
// svtkTimerLog contains walltime and cputime measurements associated
// with a given event.  These results can be later analyzed when
// "dumping out" the table.
//
// In addition, svtkTimerLog allows the user to simply get the current
// time, and to start/stop a simple timer separate from the timing
// table logging.

#include "svtkTimerLog.h"

#include "svtkMath.h"
#include "svtksys/FStream.hxx"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <iomanip>
#include <iterator>
#include <string>
#include <vector>

#ifndef _WIN32
#include <climits> // for CLK_TCK
#include <sys/time.h>
#include <unistd.h>
#endif

#ifndef _WIN32_WCE
#include <ctime>
#include <sys/types.h>
#endif
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkTimerLog);

// initialize the class variables
int svtkTimerLog::Logging = 1;
int svtkTimerLog::Indent = 0;
int svtkTimerLog::MaxEntries = 100;
int svtkTimerLog::NextEntry = 0;
int svtkTimerLog::WrapFlag = 0;
std::vector<svtkTimerLogEntry> svtkTimerLog::TimerLog;

#ifdef CLK_TCK
int svtkTimerLog::TicksPerSecond = CLK_TCK;
#else
int svtkTimerLog::TicksPerSecond = 60;
#endif

#ifndef CLOCKS_PER_SEC
#define CLOCKS_PER_SEC (svtkTimerLog::TicksPerSecond)
#endif

#ifdef _WIN32
#ifndef _WIN32_WCE
timeb svtkTimerLog::FirstWallTime;
timeb svtkTimerLog::CurrentWallTime;
#else
FILETIME svtkTimerLog::FirstWallTime;
FILETIME svtkTimerLog::CurrentWallTime;
#endif
#else
timeval svtkTimerLog::FirstWallTime;
timeval svtkTimerLog::CurrentWallTime;
tms svtkTimerLog::FirstCpuTicks;
tms svtkTimerLog::CurrentCpuTicks;
#endif

//----------------------------------------------------------------------------
// Remove timer log.
void svtkTimerLog::CleanupLog()
{
  svtkTimerLog::TimerLog.clear();
}

//----------------------------------------------------------------------------
// Clear the timing table.  walltime and cputime will also be set
// to zero when the first new event is recorded.
void svtkTimerLog::ResetLog()
{
  svtkTimerLog::WrapFlag = 0;
  svtkTimerLog::NextEntry = 0;
  // may want to free TimerLog to force realloc so
  // that user can resize the table by changing MaxEntries.
}

//----------------------------------------------------------------------------
// Record a timing event.  The event is represented by a formatted
// string.
void svtkTimerLog::FormatAndMarkEvent(const char* format, ...)
{
  if (!svtkTimerLog::Logging)
  {
    return;
  }

  static char event[4096];
  va_list var_args;
  va_start(var_args, format);
  vsnprintf(event, sizeof(event), format, var_args);
  va_end(var_args);

  svtkTimerLog::MarkEventInternal(event, svtkTimerLogEntry::STANDALONE);
}

//----------------------------------------------------------------------------
// Record a timing event and capture walltime and cputicks.
void svtkTimerLog::MarkEvent(const char* event)
{
  svtkTimerLog::MarkEventInternal(event, svtkTimerLogEntry::STANDALONE);
}

//----------------------------------------------------------------------------
// Record a timing event and capture walltime and cputicks.
void svtkTimerLog::MarkEventInternal(
  const char* event, svtkTimerLogEntry::LogEntryType type, svtkTimerLogEntry* entry)
{
  if (!svtkTimerLog::Logging)
  {
    return;
  }

  double time_diff;
  int ticks_diff;

  // If this the first event we're recording, allocate the
  // internal timing table and initialize WallTime and CpuTicks
  // for this first event to zero.
  if (svtkTimerLog::NextEntry == 0 && !svtkTimerLog::WrapFlag)
  {
    if (svtkTimerLog::TimerLog.empty())
    {
      svtkTimerLog::TimerLog.resize(svtkTimerLog::MaxEntries);
    }

#ifdef _WIN32
#ifdef _WIN32_WCE
    SYSTEMTIME st;
    GetLocalTime(&st);
    SystemTimeToFileTime(&st, &(svtkTimerLog::FirstWallTime));
#else
    ::ftime(&(svtkTimerLog::FirstWallTime));
#endif
#else
    gettimeofday(&(svtkTimerLog::FirstWallTime), nullptr);
    times(&FirstCpuTicks);
#endif

    if (entry)
    {
      svtkTimerLog::TimerLog[0] = *entry;
    }
    else
    {
      svtkTimerLog::TimerLog[0].Indent = svtkTimerLog::Indent;
      svtkTimerLog::TimerLog[0].WallTime = 0.0;
      svtkTimerLog::TimerLog[0].CpuTicks = 0;
      if (event)
      {
        svtkTimerLog::TimerLog[0].Event = event;
      }
      svtkTimerLog::TimerLog[0].Type = type;
      svtkTimerLog::NextEntry = 1;
    }
    return;
  }

  if (entry)
  {
    svtkTimerLog::TimerLog[svtkTimerLog::NextEntry] = *entry;
  }
  else
  {
#ifdef _WIN32
#ifdef _WIN32_WCE
    SYSTEMTIME st;
    GetLocalTime(&st);
    SystemTimeToFileTime(&st, &(svtkTimerLog::CurrentWallTime));
    time_diff =
      (svtkTimerLog::CurrentWallTime.dwHighDateTime - svtkTimerLog::FirstWallTime.dwHighDateTime);
    time_diff = time_diff * 429.4967296;
    time_diff = time_diff +
      ((svtkTimerLog::CurrentWallTime.dwLowDateTime - svtkTimerLog::FirstWallTime.dwLowDateTime) /
        10000000.0);
#else
    static double scale = 1.0 / 1000.0;
    ::ftime(&(svtkTimerLog::CurrentWallTime));
    time_diff = svtkTimerLog::CurrentWallTime.time - svtkTimerLog::FirstWallTime.time;
    time_diff +=
      (svtkTimerLog::CurrentWallTime.millitm - svtkTimerLog::FirstWallTime.millitm) * scale;
#endif
    ticks_diff = 0;
#else
    static double scale = 1.0 / 1000000.0;
    gettimeofday(&(svtkTimerLog::CurrentWallTime), nullptr);
    time_diff = svtkTimerLog::CurrentWallTime.tv_sec - svtkTimerLog::FirstWallTime.tv_sec;
    time_diff +=
      (svtkTimerLog::CurrentWallTime.tv_usec - svtkTimerLog::FirstWallTime.tv_usec) * scale;

    times(&CurrentCpuTicks);
    ticks_diff = (CurrentCpuTicks.tms_utime + CurrentCpuTicks.tms_stime) -
      (FirstCpuTicks.tms_utime + FirstCpuTicks.tms_stime);
#endif

    svtkTimerLog::TimerLog[svtkTimerLog::NextEntry].Indent = svtkTimerLog::Indent;
    svtkTimerLog::TimerLog[svtkTimerLog::NextEntry].WallTime = static_cast<double>(time_diff);
    svtkTimerLog::TimerLog[svtkTimerLog::NextEntry].CpuTicks = ticks_diff;
    if (event)
    {
      svtkTimerLog::TimerLog[svtkTimerLog::NextEntry].Event = event;
    }
    svtkTimerLog::TimerLog[svtkTimerLog::NextEntry].Type = type;
  }

  svtkTimerLog::NextEntry++;
  if (svtkTimerLog::NextEntry == svtkTimerLog::MaxEntries)
  {
    svtkTimerLog::NextEntry = 0;
    svtkTimerLog::WrapFlag = 1;
  }
}

//----------------------------------------------------------------------------
// Record a timing event and capture walltime and cputicks.
// Increments indent after mark.
void svtkTimerLog::MarkStartEvent(const char* event)
{
  if (!svtkTimerLog::Logging)
  { // Maybe we should still change the Indent ...
    return;
  }

  svtkTimerLog::MarkEventInternal(event, svtkTimerLogEntry::START);
  ++svtkTimerLog::Indent;
}

//----------------------------------------------------------------------------
// Record a timing event and capture walltime and cputicks.
// Decrements indent after mark.
void svtkTimerLog::MarkEndEvent(const char* event)
{
  if (!svtkTimerLog::Logging)
  { // Maybe we should still change the Indent ...
    return;
  }

  svtkTimerLog::MarkEventInternal(event, svtkTimerLogEntry::END);
  --svtkTimerLog::Indent;
}

//----------------------------------------------------------------------------
// Record a timing event with known walltime and cputicks.
void svtkTimerLog::InsertTimedEvent(const char* event, double time, int cpuTicks)
{
  if (!svtkTimerLog::Logging)
  {
    return;
  }
  // manually create both the start and end event and then
  // change the start events values to appear like other events
  svtkTimerLogEntry entry;
  entry.WallTime = time;
  entry.CpuTicks = cpuTicks;
  if (event)
  {
    entry.Event = event;
  }
  entry.Type = svtkTimerLogEntry::INSERTED;
  entry.Indent = svtkTimerLog::Indent;

  svtkTimerLog::MarkEventInternal(event, svtkTimerLogEntry::INSERTED, &entry);
}

//----------------------------------------------------------------------------
// Record a timing event and capture walltime and cputicks.
int svtkTimerLog::GetNumberOfEvents()
{
  if (svtkTimerLog::WrapFlag)
  {
    return svtkTimerLog::MaxEntries;
  }
  else
  {
    return svtkTimerLog::NextEntry;
  }
}

//----------------------------------------------------------------------------
svtkTimerLogEntry* svtkTimerLog::GetEvent(int idx)
{
  int num = svtkTimerLog::GetNumberOfEvents();
  int start = 0;
  if (svtkTimerLog::WrapFlag)
  {
    start = svtkTimerLog::NextEntry;
  }

  if (idx < 0 || idx >= num)
  {
    cerr << "Bad entry index " << idx << endl;
    return nullptr;
  }
  idx = (idx + start) % svtkTimerLog::MaxEntries;

  return &(svtkTimerLog::TimerLog[idx]);
}

//----------------------------------------------------------------------------
int svtkTimerLog::GetEventIndent(int idx)
{
  if (svtkTimerLogEntry* tmp = svtkTimerLog::GetEvent(idx))
  {
    return tmp->Indent;
  }
  return 0;
}

//----------------------------------------------------------------------------
double svtkTimerLog::GetEventWallTime(int idx)
{
  if (svtkTimerLogEntry* tmp = svtkTimerLog::GetEvent(idx))
  {
    return tmp->WallTime;
  }
  return 0.0;
}

//----------------------------------------------------------------------------
const char* svtkTimerLog::GetEventString(int idx)
{
  if (svtkTimerLogEntry* tmp = svtkTimerLog::GetEvent(idx))
  {
    return tmp->Event.c_str();
  }
  return nullptr;
}

//----------------------------------------------------------------------------
svtkTimerLogEntry::LogEntryType svtkTimerLog::GetEventType(int idx)
{
  if (svtkTimerLogEntry* tmp = svtkTimerLog::GetEvent(idx))
  {
    return tmp->Type;
  }
  return svtkTimerLogEntry::INVALID;
}

//----------------------------------------------------------------------------
// Write the timing table out to a file.  Calculate some helpful
// statistics (deltas and percentages) in the process.
void svtkTimerLog::DumpLogWithIndents(ostream* os, double threshold)
{
#ifndef _WIN32_WCE
  int num = svtkTimerLog::GetNumberOfEvents();
  std::vector<bool> handledEvents(num, false);

  for (int w = 0; w < svtkTimerLog::WrapFlag + 1; w++)
  {
    int start = 0;
    int end = svtkTimerLog::NextEntry;
    if (svtkTimerLog::WrapFlag != 0 && w == 0)
    {
      start = svtkTimerLog::NextEntry;
      end = svtkTimerLog::MaxEntries;
    }
    for (int i1 = start; i1 < end; i1++)
    {
      int indent1 = svtkTimerLog::GetEventIndent(i1);
      svtkTimerLogEntry::LogEntryType eventType = svtkTimerLog::GetEventType(i1);
      int endEvent = -1; // only modified if this is a START event
      if (eventType == svtkTimerLogEntry::END && handledEvents[i1] == true)
      {
        continue; // this END event is handled by the corresponding START event
      }
      if (eventType == svtkTimerLogEntry::START)
      {
        // Search for an END event. it may be before the START event if we've wrapped.
        int counter = 1;
        while (counter < num && svtkTimerLog::GetEventIndent((i1 + counter) % num) > indent1)
        {
          counter++;
        }
        if (svtkTimerLog::GetEventIndent((i1 + counter) % num) == indent1)
        {
          counter--;
          endEvent = (i1 + counter) % num;
          handledEvents[endEvent] = true;
        }
      }
      double dtime = threshold;
      if (eventType == svtkTimerLogEntry::START)
      {
        dtime = svtkTimerLog::GetEventWallTime(endEvent) - svtkTimerLog::GetEventWallTime(i1);
      }
      if (dtime >= threshold)
      {
        int j = indent1;
        while (j-- > 0)
        {
          *os << "    ";
        }
        *os << svtkTimerLog::GetEventString(i1);
        if (endEvent != -1)
        { // Start event.
          *os << ",  " << dtime << " seconds";
        }
        else if (eventType == svtkTimerLogEntry::INSERTED)
        {
          *os << ",  " << svtkTimerLog::GetEventWallTime(i1) << " seconds (inserted time)";
        }
        else if (eventType == svtkTimerLogEntry::END)
        {
          *os << " (END event without matching START event)";
        }
        *os << endl;
      }
    }
  }
#endif
}

//----------------------------------------------------------------------------
void svtkTimerLog::DumpLogWithIndentsAndPercentages(std::ostream* os)
{
  assert(os);
  // Use indents to pair start/end events. Indents work like this:
  // Indent | Event
  // ------ | -----
  //   0    | Event 1 Start
  //   1    | SubEvent 1 Start
  //   2    | SubEvent 1 End
  //   1    | SubEvent 2 Start
  //   2    | SubSubEvent 1 Start
  //   3    | SubSubEvent 1 End
  //   2    | SubEvent 2 End
  //   1    | Event 1 End

  // If we've wrapped the entry buffer, the indent information will be
  // nonsensical. I don't trust the parsing logic below to not do bizarre,
  // possibly crashy things in this case, so let's just error out and let the
  // dev know how to fix the issue.
  if (svtkTimerLog::WrapFlag)
  {
    *os << "Error: Event log has exceeded svtkTimerLog::MaxEntries.\n"
           "Call svtkTimerLog::SetMaxEntries to increase the log buffer size.\n"
           "Current svtkTimerLog::MaxEntries: "
        << svtkTimerLog::MaxEntries << ".\n";
    return;
  }

  // Store previous 'scopes' in a LIFO buffer
  typedef std::pair<int, double> IndentTime;
  std::vector<IndentTime> parentInfo;

  // Find the longest event string:
  int numEvents = svtkTimerLog::GetNumberOfEvents();
  int longestString = 0;
  for (int i = 0; i < numEvents; ++i)
  {
    longestString =
      std::max(longestString, static_cast<int>(strlen(svtkTimerLog::GetEventString(i))));
  }

  // Loop to numEvents - 1, since the last event must be an end event.
  for (int startIdx = 0; startIdx < numEvents - 1; ++startIdx)
  {
    int curIndent = svtkTimerLog::GetEventIndent(startIdx);

    svtkTimerLogEntry::LogEntryType logEntryType = svtkTimerLog::GetEventType(startIdx);
    if (logEntryType == svtkTimerLogEntry::END)
    { // Skip this event if it is an end event:
      assert(!parentInfo.empty());
      parentInfo.pop_back();
      continue;
    }
    else if (logEntryType == svtkTimerLogEntry::STANDALONE)
    { // Skip this event if it is just to mark that an event occurred
      continue;
    }

    // Find the event that follows the end event:
    int endIdx = startIdx + 1;
    for (; endIdx < numEvents; ++endIdx)
    {
      if (svtkTimerLog::GetEventIndent(endIdx) == curIndent)
      {
        break;
      }
    }

    // Move back one event to get our end event (also works when we've reached
    // the end of the event log).
    endIdx--;

    // Get the current event time:
    double elapsedTime = logEntryType == svtkTimerLogEntry::START
      ? svtkTimerLog::GetEventWallTime(endIdx) - svtkTimerLog::GetEventWallTime(startIdx)
      : svtkTimerLog::GetEventWallTime(startIdx);

    // The total time the parent took to execute. If empty, this is the first
    // event, and just set to 100%.
    IndentTime parent = parentInfo.empty() ? IndentTime(-1, elapsedTime) : parentInfo.back();

    // Percentage of parent exec time, rounded to a single decimal:
    double percentage = std::round(elapsedTime / parent.second * 1000.) / 10.;

    *os << std::setw(12) << std::setprecision(6) << std::fixed << elapsedTime << std::setw(0) << "s"
        << std::setw(curIndent * 2) << " " << std::setprecision(1) << std::setw(5) << std::right
        << percentage << std::setw(0) << std::left << "% " << std::setw(longestString)
        << svtkTimerLog::GetEventString(startIdx);
    if (logEntryType == svtkTimerLogEntry::INSERTED)
    {
      *os << " (inserted time)";
    }
    *os << "\n";

    // Add our parent info if this was time with a START and END event:
    if (logEntryType == svtkTimerLogEntry::START)
    {
      parentInfo.push_back(IndentTime(curIndent, elapsedTime));
    }
  }
}

//----------------------------------------------------------------------------
// Write the timing table out to a file. This is meant for non-timed events,
// i.e. event type = STANDALONE. All other event types besides the first
// are ignored.
void svtkTimerLog::DumpLog(const char* filename)
{
#ifndef _WIN32_WCE
  svtksys::ofstream os_with_warning_C4701(filename);
  int i;

  if (svtkTimerLog::WrapFlag)
  {
    svtkTimerLog::DumpEntry(os_with_warning_C4701, 0,
      svtkTimerLog::TimerLog[svtkTimerLog::NextEntry].WallTime, 0,
      svtkTimerLog::TimerLog[svtkTimerLog::NextEntry].CpuTicks, 0,
      svtkTimerLog::TimerLog[svtkTimerLog::NextEntry].Event.c_str());
    int previousEvent = svtkTimerLog::NextEntry;
    for (i = svtkTimerLog::NextEntry + 1; i < svtkTimerLog::MaxEntries; i++)
    {
      if (svtkTimerLog::TimerLog[i].Type == svtkTimerLogEntry::STANDALONE)
      {
        svtkTimerLog::DumpEntry(os_with_warning_C4701, i - svtkTimerLog::NextEntry,
          svtkTimerLog::TimerLog[i].WallTime,
          svtkTimerLog::TimerLog[i].WallTime - svtkTimerLog::TimerLog[previousEvent].WallTime,
          svtkTimerLog::TimerLog[i].CpuTicks,
          svtkTimerLog::TimerLog[i].CpuTicks - svtkTimerLog::TimerLog[previousEvent].CpuTicks,
          svtkTimerLog::TimerLog[i].Event.c_str());
        previousEvent = i;
      }
    }
    for (i = 0; i < svtkTimerLog::NextEntry; i++)
    {
      if (svtkTimerLog::TimerLog[i].Type == svtkTimerLogEntry::STANDALONE)
      {
        svtkTimerLog::DumpEntry(os_with_warning_C4701,
          svtkTimerLog::MaxEntries - svtkTimerLog::NextEntry + i, svtkTimerLog::TimerLog[i].WallTime,
          svtkTimerLog::TimerLog[i].WallTime - svtkTimerLog::TimerLog[previousEvent].WallTime,
          svtkTimerLog::TimerLog[i].CpuTicks,
          svtkTimerLog::TimerLog[i].CpuTicks - svtkTimerLog::TimerLog[previousEvent].CpuTicks,
          svtkTimerLog::TimerLog[i].Event.c_str());
        previousEvent = i;
      }
    }
  }
  else
  {
    svtkTimerLog::DumpEntry(os_with_warning_C4701, 0, svtkTimerLog::TimerLog[0].WallTime, 0,
      svtkTimerLog::TimerLog[0].CpuTicks, 0, svtkTimerLog::TimerLog[0].Event.c_str());
    int previousEvent = 0;
    for (i = 1; i < svtkTimerLog::NextEntry; i++)
    {
      if (svtkTimerLog::TimerLog[i].Type == svtkTimerLogEntry::STANDALONE)
      {
        svtkTimerLog::DumpEntry(os_with_warning_C4701, i, svtkTimerLog::TimerLog[i].WallTime,
          svtkTimerLog::TimerLog[i].WallTime - svtkTimerLog::TimerLog[previousEvent].WallTime,
          svtkTimerLog::TimerLog[i].CpuTicks,
          svtkTimerLog::TimerLog[i].CpuTicks - svtkTimerLog::TimerLog[previousEvent].CpuTicks,
          svtkTimerLog::TimerLog[i].Event.c_str());
        previousEvent = i;
      }
    }
  }

  os_with_warning_C4701.close();
#endif
}

//----------------------------------------------------------------------------
// Print method for svtkTimerLog.
void svtkTimerLog::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "MaxEntries: " << svtkTimerLog::MaxEntries << "\n";
  os << indent << "NextEntry: " << svtkTimerLog::NextEntry << "\n";
  os << indent << "WrapFlag: " << svtkTimerLog::WrapFlag << "\n";
  os << indent << "TicksPerSecond: " << svtkTimerLog::TicksPerSecond << "\n";
  os << "\n";

  os << indent << "Entry \tWall Time\tCpuTicks\tEvent\n";
  os << indent << "----------------------------------------------\n";

  if (svtkTimerLog::WrapFlag)
  {
    for (int i = svtkTimerLog::NextEntry; i < svtkTimerLog::MaxEntries; i++)
    {
      os << indent << i << "\t\t" << TimerLog[i].WallTime << "\t\t" << TimerLog[i].CpuTicks
         << "\t\t" << TimerLog[i].Event << "\n";
    }
  }

  for (int i = 0; i < svtkTimerLog::NextEntry; i++)
  {
    os << indent << i << "\t\t" << TimerLog[i].WallTime << "\t\t" << TimerLog[i].CpuTicks << "\t\t"
       << TimerLog[i].Event << "\n";
  }

  os << "\n" << indent << "StartTime: " << this->StartTime << "\n";
}

// Methods to support simple timer functionality, separate from
// timer table logging.

//----------------------------------------------------------------------------
// Returns the elapsed number of seconds since 00:00:00 Coordinated Universal
// Time (UTC), Thursday, 1 January 1970. This is also called Unix Time.
double svtkTimerLog::GetUniversalTime()
{
  double currentTimeInSeconds;

#ifdef _WIN32
#ifdef _WIN32_WCE
  FILETIME CurrentTime;
  SYSTEMTIME st;
  GetLocalTime(&st);
  SystemTimeToFileTime(&st, &CurrentTime);
  currentTimeInSeconds = CurrentTime.dwHighDateTime;
  currentTimeInSeconds *= 429.4967296;
  currentTimeInSeconds = currentTimeInSeconds + CurrentTime.dwLowDateTime / 10000000.0;
#else
  timeb CurrentTime;
  static double scale = 1.0 / 1000.0;
  ::ftime(&CurrentTime);
  currentTimeInSeconds = CurrentTime.time + scale * CurrentTime.millitm;
#endif
#else
  timeval CurrentTime;
  static double scale = 1.0 / 1000000.0;
  gettimeofday(&CurrentTime, nullptr);
  currentTimeInSeconds = CurrentTime.tv_sec + scale * CurrentTime.tv_usec;
#endif

  return currentTimeInSeconds;
}

//----------------------------------------------------------------------------
double svtkTimerLog::GetCPUTime()
{
#ifndef _WIN32_WCE
  return static_cast<double>(clock()) / static_cast<double>(CLOCKS_PER_SEC);
#else
  return 1.0;
#endif
}

//----------------------------------------------------------------------------
// Set the StartTime to the current time. Used with GetElapsedTime().
void svtkTimerLog::StartTimer()
{
  this->StartTime = svtkTimerLog::GetUniversalTime();
}

//----------------------------------------------------------------------------
// Sets EndTime to the current time. Used with GetElapsedTime().
void svtkTimerLog::StopTimer()
{
  this->EndTime = svtkTimerLog::GetUniversalTime();
}

//----------------------------------------------------------------------------
// Returns the difference between StartTime and EndTime as
// a floating point value indicating the elapsed time in seconds.
double svtkTimerLog::GetElapsedTime()
{
  return (this->EndTime - this->StartTime);
}

//----------------------------------------------------------------------------
void svtkTimerLog::DumpEntry(ostream& os, int index, double ttime, double deltatime, int tick,
  int deltatick, const char* event)
{
  os << index << "   " << ttime << "  " << deltatime << "   "
     << static_cast<double>(tick) / svtkTimerLog::TicksPerSecond << "  "
     << static_cast<double>(deltatick) / svtkTimerLog::TicksPerSecond << "  ";
  if (deltatime == 0.0)
  {
    os << "0.0   ";
  }
  else
  {
    os << 100.0 * deltatick / svtkTimerLog::TicksPerSecond / deltatime << "   ";
  }
  os << event << "\n";
}

//----------------------------------------------------------------------------
void svtkTimerLog::SetMaxEntries(int a)
{
  if (a == svtkTimerLog::MaxEntries)
  {
    return;
  }
  int numEntries = svtkTimerLog::GetNumberOfEvents();
  if (svtkTimerLog::WrapFlag)
  { // if we've wrapped events, reorder them
    std::vector<svtkTimerLogEntry> tmp;
    tmp.reserve(svtkTimerLog::MaxEntries);
    std::copy(svtkTimerLog::TimerLog.begin() + svtkTimerLog::NextEntry, svtkTimerLog::TimerLog.end(),
      std::back_inserter(tmp));
    std::copy(svtkTimerLog::TimerLog.begin(), svtkTimerLog::TimerLog.begin() + svtkTimerLog::NextEntry,
      std::back_inserter(tmp));
    svtkTimerLog::TimerLog = tmp;
    svtkTimerLog::WrapFlag = 0;
  }
  if (numEntries <= a)
  {
    svtkTimerLog::TimerLog.resize(a);
    svtkTimerLog::NextEntry = numEntries;
    svtkTimerLog::WrapFlag = 0;
    svtkTimerLog::MaxEntries = a;
    return;
  }
  // Reduction so we need to get rid of the first bunch of events
  int offset = numEntries - a;
  assert(offset >= 0);
  svtkTimerLog::TimerLog.erase(
    svtkTimerLog::TimerLog.begin(), svtkTimerLog::TimerLog.begin() + offset);
  svtkTimerLog::MaxEntries = a;
  svtkTimerLog::NextEntry = 0;
  svtkTimerLog::WrapFlag = 1;
}

//----------------------------------------------------------------------------
int svtkTimerLog::GetMaxEntries()
{
  return svtkTimerLog::MaxEntries;
}
