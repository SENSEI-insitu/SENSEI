#ifndef OSCILLATORS_BRIDGE_H
#define OSCILLATORS_BRIDGE_H

#include <mpi.h>
#include <string>

#include "particles.h"

namespace bridge
{
  void initialize(MPI_Comm world,
                  size_t window,
                  size_t nblocks,
                  size_t n_local_blocks,
                  int domain_shape_x, int domain_shape_y, int domain_shape_z,
                  int* gid,
                  int* from_x, int* from_y, int* from_z,
                  int* to_x,   int* to_y,   int* to_z,
                  int* shape, int ghostLevels,
                  const std::string& config_file);

  void set_data(int gid, float* data);
  void set_particles(int gid, const Particles& particles);

  void analyze(float time);

  void finalize(size_t k_max, size_t nblocks);
}

#endif
