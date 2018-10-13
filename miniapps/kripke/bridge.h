#ifndef _CONDUIT_SENSI_BRIDGE_H
#define _CONDUIT_SENSI_BRIDGE_H

#include <mpi.h>
#include <string>
#include <conduit.hpp>

  //called before the simulation loops
  void initialize(MPI_Comm world, 
                  conduit::Node* node, 
                  const std::string& config_file);

  //called during simulation loop to update node
  void analyze(conduit::Node* node); 

  //called for cleanup
  void finalize();

#endif
