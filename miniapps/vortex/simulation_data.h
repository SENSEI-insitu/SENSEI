#ifndef SIMULATION_DATA_H
#define SIMULATION_DATA_H
#include "patch.h"

/******************************************************************************
 * Simulation data and functions
 ******************************************************************************/

#define MAX_VORTICES 10

class vortex
{
public:
    vortex();
    vortex(float x, float y, float z, 
           float vx, float vy, float vz,
           float g, float r);

    float location[3];
    float velocity[3];
    float gamma;
    float radius;
};


class simulation_data
{
public:
    simulation_data();
    ~simulation_data();

    int     par_rank;
    int     par_size;
    int     cycle;
    double  time;
    double  dt;
    int     max_levels;
    int     refinement_ratio;
    bool    balance;
    bool    log;

    float   dims[3];
    float   window[6];
    float   data_refinement_threshold;

    int     nVortex;
    vortex  vortices[MAX_VORTICES];

    patch_t patch;

    int     *npatches_per_rank;  // [par_size]
    int     *npatches_per_level; // [max_levels]
};

#endif
