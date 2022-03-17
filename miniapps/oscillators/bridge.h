#ifndef OSCILLATORS_BRIDGE_H
#define OSCILLATORS_BRIDGE_H

#include "Particles.h"
#include "Oscillator.h"

namespace sensei { class DataAdaptor; }

#include <mpi.h>
#include <string>
#include <vector>
#include <memory>

namespace bridge
{
  /// initialize for in situ processing using SENSEI
  int initialize(size_t nblocks, size_t n_local_blocks, float *origin,
    float *spacing, int domain_shape_x, int domain_shape_y, int domain_shape_z,
    int *gid, int *from_x, int *from_y, int *from_z, int *to_x, int *to_y,
    int *to_z, int *shape, int ghostLevels, const std::string &config_file);

  /// pass the grid based array for the block identified by gid
  void set_data(int gid, float* data);

  /// pass the particle based data for for the block identified by gid
  void set_particles(int gid, const std::vector<Particle> &particles);

  /// pass the list of oscillators
  void set_oscillators(const OscillatorArray &oscilators);

  /// invoke in situ processing
  void execute(long step, float time, sensei::DataAdaptor **dataOut);

  /// finalize in situ processing
  void finalize();
}

#endif
