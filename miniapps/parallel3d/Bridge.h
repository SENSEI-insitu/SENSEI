#ifndef PARALLEL3D_BRIDGE_H
#define PARALLEL3D_BRIDGE_H

#include <mpi.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
/// This defines the analysis bridge for parallel_3d miniapp.

/// Called before simulation loop
void bridge_initialize(MPI_Comm comm, int g_x, int g_y, int g_z,
  int l_x, int l_y, int l_z, uint64_t start_extents_x, uint64_t start_extents_y,
  uint64_t start_extents_z, int tot_blocks_x, int tot_blocks_y, int tot_blocks_z,
  int block_id_x, int block_id_y, int block_id_z, const char* config_file);

/// Called per timestep in the simulation loop
void bridge_update(int tstep, double time, double *pressure, double* temperature, double* density);

/// Called just before simulation terminates.
void bridge_finalize(MPI_Comm comm);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
