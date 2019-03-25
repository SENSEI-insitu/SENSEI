#ifndef PARALLEL3D_BRIDGE_H
#define PARALLEL3D_BRIDGE_H

#include <mpi.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// This defines the analysis bridge for parallel_3d miniapp.

/// Called before simulation loop
int bridge_initialize(const char *config_file, int g_nx, int g_ny, int g_nz,
  uint64_t offs_x, uint64_t offs_y, uint64_t offs_z, int l_nx, int l_ny,
  int l_nz, double *pressure, double* temperature, double* density);

/// Called per timestep in the simulation loop
void bridge_update(int tstep, double time);

/// Called just before simulation terminates.
void bridge_finalize();

#ifdef __cplusplus
} // extern "C"
#endif

#endif
