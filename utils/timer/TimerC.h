#ifndef timerc_h
#define timerc_h

#ifdef __cplusplus
extern "C" {
#endif

  void TIMER_SetLogging(bool val);
  void TIMER_MarkStartEvent(const char* val);
  void TIMER_MarkEndEvent(const char* val);
  void TIMER_MarkStartTimeStep(int timestep, double time);
  void TIMER_MarkEndTimeStep();
  void TIMER_Print();

#ifdef __cplusplus
}
#endif

#endif
