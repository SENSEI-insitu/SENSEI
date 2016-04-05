#ifndef timer_Timer_h
#define timer_Timer_h

#include <iostream>
#include <mpi.h>

namespace timer
{
  /// @brief Enable/Disable logging (default is `true`).
  void SetLogging(bool val);

  /// @brief Get whether logging is enabled.
  bool GetLogging();

  /// @brief Enable/Disable tracking temporal summaries (default is `false`).
  ///
  /// For runs where the event markers are consistent across timesteps
  /// i.e. they occur in same order and exactly as many across different
  /// timesteps, one can use this to track a summary instead of full values
  /// for duration and memory used. This not only keeps the memory overhead
  /// for long runs low, but also make it easier to process the generated
  /// output.
  void SetTrackSummariesOverTime(bool val);

  /// @brief Return whether summaries are tracked over time.
  bool GetTrackSummariesOverTime();

  /// @brief Log start of a log-able event.
  ///
  /// This marks the beginning of a event that must be logged.
  /// The @arg eventname must match when calling MarkEndEvent() to
  /// mark the end of the event.
  void MarkStartEvent(const char* eventname);

  /// @brief Log end of a log-able event.
  ///
  /// This marks the end of a event that must be logged.
  /// The @arg eventname must match when calling MarkEndEvent() to
  /// mark the end of the event.
  void MarkStartEvent(const char* eventname);
  void MarkEndEvent(const char* eventname);

  /// @brief Mark the beginning of a timestep.
  ///
  /// This marks the beginning of a timestep. All MarkStartEvent and
  /// MarkEndEvent after this until MarkEndTimeStep are assumed to be
  /// happening for a specific timestep and will be combined with subsequent
  /// timesteps.
  void MarkStartTimeStep(int timestep, double time);

  /// @brief Marks the end of the current timestep.
  void MarkEndTimeStep();

  /// @brief Print log to the output stream.
  ///
  /// Note this triggers collective operations and hence must be called on all
  /// ranks. The amount of processes outputting can be reduced by using
  /// the moduloOuput which only outputs for (rank % moduloOutput) == 0.
  /// The default value for this is 1.
  void PrintLog(std::ostream& stream, MPI_Comm world, int moduloOutput = 1);

  class MarkEvent
    {
    const char* EventName;
  public:
    MarkEvent(const char* name) : EventName(name) { MarkStartEvent(name); }
    ~MarkEvent() { MarkEndEvent(this->EventName); }
    };
}

#endif
