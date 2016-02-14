#ifndef INSITU_BRIDGE_H
#define INSITU_BRIDGE_H

#include <mpi.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

  void insitu_bridge_initialize(MPI_Comm comm,
    int g_x, int g_y, int g_z,
    int l_x, int l_y, int l_z,
    uint64_t start_extents_x, uint64_t start_extents_y, uint64_t start_extents_z,
    int tot_blocks_x, int tot_blocks_y, int tot_blocks_z,
    int block_id_x, int block_id_y, int block_id_z,
    int bins);

  void insitu_bridge_update(double *pressure, double* temperature, double* density);

  void insitu_bridge_finalize();

#ifdef __cplusplus
} // extern "C"
#endif

#endif
