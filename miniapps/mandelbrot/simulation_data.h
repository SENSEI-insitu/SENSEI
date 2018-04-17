#ifndef SIMULATION_DATA_H
#define SIMULATION_DATA_H
#include "patch.h"

/******************************************************************************
 * Simulation data and functions
 ******************************************************************************/

class simulation_data
{
public:
    simulation_data();
    ~simulation_data();

    int     par_rank;
    int     par_size;
    int     cycle;
    double  time;
    int     max_levels;
    int     refinement_ratio;
    bool    balance;
    bool    log;

    patch_t patch;

    int     *npatches_per_rank;  // [par_size]
    int     *npatches_per_level; // [max_levels]
};

#endif
