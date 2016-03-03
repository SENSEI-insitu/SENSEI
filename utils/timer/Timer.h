#ifndef timer_Timer_h
#define timer_Timer_h

#include <iostream>

namespace timer
{
  /// @brief Enable/Disable logging (default is enabled).
  void SetLogging(bool val);

  /// @brief Get whether logging is enabled.
  bool GetLogging();

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
  void PrintLog(std::ostream& stream);

  class MarkEvent
    {
    const char* EventName;
  public:
    MarkEvent(const char* name) : EventName(name) { MarkStartEvent(name); }
    ~MarkEvent() { MarkEndEvent(this->EventName); }
    };
}

#endif
