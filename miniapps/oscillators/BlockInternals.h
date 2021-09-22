#ifndef BlockInternals_h
#define BlockInternals_h

#include <senseiConfig.h>

#include "Oscillator.h"

namespace BlockInternals
{
/// dispatch the calculations to the requested device
int UpdateFields(
  int deviceId,
  float t,
  const Oscillator *oscillators,
  int nOscillators,
  int ni, int nj, int nk,
  int i0, int j0, int k0,
  float x0, float y0, float z0,
  float dx, float dy, float dz,
  float *pdata);
}

#endif
