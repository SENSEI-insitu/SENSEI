#ifndef _CONDUIT_SENSI_BRIDGE_H
#define _CONDUIT_SENSI_BRIDGE_H

#include <string>
#include <conduit.hpp>

// Called before the simulation loops.
void SenseiInitialize( const std::string& config_file );

// Called during simulation loop to update node.
void SenseiAnalyze( conduit::Node &node ); 

// Called for cleanup.
void SenseiFinalize();

#endif
