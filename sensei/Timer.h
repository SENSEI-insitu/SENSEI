#ifndef timer_Timer_h
#define timer_Timer_h

#include <iostream>
#include <mpi.h>

namespace sensei
{
struct Timer
{
/// Initialize logging from environment variables, and/or the timer
/// API below. This is a collective call with respect to the timer's
/// communicator.
///
/// If found in the environment the following variable override the
/// the current settings
//
///     TIMER_ENABLE          : integer turns on or off logging
///     TIMER_ENABLE_SUMMARY  : integer truns on off logging in summary format
///     TIMER_SUMMARY_MODULUS : print rank data when rank % modulus == 0
///     TIMER_LOG_FILE        : path to write timer log to
///     MEMPROF_LOG_FILE      : path to write memory profiler log to
///     MEMPROF_INTERVAL      : number of seconds between memory recordings
///
static void Initialize();

/// Finalize the log. this is where logs are written and cleanup occurs.
/// All processes in the communicator must call, and it must be called
/// prior to MPI_Finalize.
static void Finalize();

/// Sets the communicator for MPI calls. This must be called prior to
/// initialization.
/// default value: MPI_COMM_NULL
static void SetCommunicator(MPI_Comm comm);

/// Sets the path to write the timer log to
/// overriden by TIMER_LOG_FILE environment variable
/// default value; TimerLog.csv
static void SetTimerLogFile(const std::string &fileName);

/// Sets the path to write the timer log to
/// overriden by MEMPROF_LOG_FILE environment variable
/// default value: MemProfLog.csv
static void SetMemProfLogFile(const std::string &fileName);

/// Sets the number of seconds in between memory use recordings
/// overriden by MEMPROF_INTERVAL environment variable.
static void SetMemProfInterval(int interval);

/// Enable/Disable logging. Overriden by TIMER_ENABLE and
/// TIMER_ENABLE_SUMMARY environment variables. In the
/// default format a CSV file is generated capturing each ranks
/// timer events. In the summary format a pretty and breif output
/// is sent to the stderr stream.
/// default value: disabled
static void Enable(bool summaryFmt = false);
static void Disable();

/// return true if loggin is enabled.
static bool Enabled();

/// Sets the timer's summary log modulus. Output incudes data from
/// MPI ranks where rank % modulus == 0 Overriden by
/// TIMER_SUMMARY_MODULUS environment variable
/// default value; 1000000000
static void SetSummaryModulus(int modulus);

/// @brief Log start of an event.
///
/// This marks the beginning of a event that must be logged.
/// The @arg eventname must match when calling MarkEndEvent() to
/// mark the end of the event. I/O events can log the number of
/// bytes read or written.
static void MarkStartEvent(const char* eventname);

/// @brief Log end of a log-able event.
///
/// This marks the end of a event that must be logged.
/// The @arg eventname must match when calling MarkEndEvent() to
/// mark the end of the event.
static void MarkEndEvent(const char *eventname);

/// Start and end an I/O event. I/O events may be started or ended
/// as normal events, but one of the following must be called to
/// log the size of the event. The last of these is the size recorded.
/// As an example if one does not know the size of the read/write at
/// the start event one may call MarkStartEvent("event name") followed
/// by MarkEndEvent("event name", nBytes). Converseley if one knows
/// the size of the event up front one may call MarkStartEvent("name", bytes)
/// followed by MarkEndEvent("name")
static void MarkStartEvent(const char *eventname, long long numBytes);

static void MarkEndEvent(const char *eventname, long long numBytes);

/// @brief Mark the beginning of a timestep.
///
/// This marks the beginning of a timestep. All MarkStartEvent and
/// MarkEndEvent after this until MarkEndTimeStep are assumed to be
/// happening for a specific timestep and will be combined with subsequent
/// timesteps.
static void MarkStartTimeStep(int timestep, double time);

/// @brief Marks the end of the current timestep.
static void MarkEndTimeStep();

/// @brief Print log to the output stream.
///
/// Note this triggers collective operations and hence must be called on all
/// ranks. The amount of processes outputting can be reduced by using
/// the moduloOuput which only outputs for (rank % moduloOutput) == 0.
/// The default value for this is 1.
static void PrintLog(std::ostream &stream);

/// MarkEvent -- A helper class that times it's life.
/// A timer event is created that starts at the object's construction
/// and ends at its destruction. The pointer to the event name must
/// be valid throughout the objects life.
class MarkEvent
{
public:
  MarkEvent() = delete;

  // start an event
  MarkEvent(const char *name) : EventName(name) { MarkStartEvent(name); }
  // destroy object emding the event if it is on going
  ~MarkEvent() { MarkEndEvent(this->EventName); }
private:
  const char *EventName;
};

};

}
#endif
