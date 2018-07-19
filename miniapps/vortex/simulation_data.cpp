#include "simulation_data.h"
#include <stdlib.h>
#include <string.h>

vortex::vortex()
{
    location[0] = location[1] = location[2] = 0.f;
    velocity[0] = velocity[1] = velocity[2] = 0.f;
    gamma = 1.f;
}

vortex::vortex(float x, float y, float z,
    float vx, float vy, float vz, 
    float g, float r)
{
    location[0] = x;
    location[1] = y;
    location[2] = z;

    velocity[0] = vx;
    velocity[1] = vy;
    velocity[2] = vz;

    gamma = g;
    radius = r;
}

simulation_data::simulation_data()
{
    cycle = 0;
    time = 0.;
    dt = 0.029;
    max_levels = 2;
    refinement_ratio = 4;
    balance = false;
    log = false;

    dims[0] = 256;
    dims[1] = 32;
    dims[2] = 32;

    // X
    window[0] = -1.f;
    window[1] = 10.f;
    // Y
    window[2] = -1.f;
    window[3] = 1.f;
    // Z
    window[4] = -1.f;
    window[5] = 1.f;

    data_refinement_threshold = 0.04f;

    // Set up the initial vortex.
    nVortex = 1;
    vortices[0] = vortex(0., 0.01, 0.,
                         0.02, 0., 0., // its velocity
                         1., // gamma
                         0.01 // radius
                        );
#if 1
    // Add another. What the heck.
    nVortex++;
    vortices[1] = vortex(5., 0.05, 0.25,
                         0.02, 0., 0., // its velocity
                         1., // gamma
                         0.02 // radius
                        );
#endif
    // TODO: put the vortices some place. Maybe increase nVortex.

    patch_ctor(&patch);
    npatches_per_rank = NULL;
    npatches_per_level = NULL;
}

simulation_data::~simulation_data()
{
    patch_dtor(&patch);

    if(npatches_per_rank != NULL)
        FREE(npatches_per_rank);

    if(npatches_per_level != NULL)
        FREE(npatches_per_level);
}

