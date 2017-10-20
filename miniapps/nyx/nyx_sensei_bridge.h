#ifndef _NYX_SENSI_BRIDGE_H
#define _NYX_SENSI_BRIDGE_H

#include <AMReX_REAL.H>
#include <mpi.h>
#include <string>

namespace amrex {
  class MultiFab;
}

namespace nyx_sensei_bridge
{
  void initialize(MPI_Comm world,
                  size_t nblocks,
                  int domain_from_x, int domain_from_y, int domain_from_z,
                  int domain_to_x, int domain_to_y, int domain_to_z,
                  amrex::Real phys_from_x, amrex::Real phys_from_y, amrex::Real phys_from_z,
                  amrex::Real phys_to_x, amrex::Real phys_to_y, amrex::Real phys_to_z,
                  const std::string& config_file);

  void analyze(const amrex::MultiFab& simulation_data, amrex::Real time, int time_step);

  void finalize();
}

#endif
