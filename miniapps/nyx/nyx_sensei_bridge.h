#ifndef _NYX_SENSI_BRIDGE_H
#define _NYX_SENSI_BRIDGE_H

#include <REAL.H>
#include <mpi.h>
#include <string>

class MultiFab;

namespace nyx_sensei_bridge
{
  void initialize(MPI_Comm world,
                  size_t nblocks,
                  int domain_from_x, int domain_from_y, int domain_from_z,
                  int domain_to_x, int domain_to_y, int domain_to_z,
                  Real phys_from_x, Real phys_from_y, Real phys_from_z,
                  Real phys_to_x, Real phys_to_y, Real phys_to_z,
                  const std::string& config_file);

  void analyze(const MultiFab& simulation_data, Real time, int time_step);

  void finalize();
}

#endif
