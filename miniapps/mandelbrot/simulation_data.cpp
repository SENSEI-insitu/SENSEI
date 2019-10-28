#include "simulation_data.h"
#include <stdlib.h>
#include <string.h>

simulation_data::simulation_data()
{
    cycle = 0;
    time = 0.;
    max_levels = 2;
    refinement_ratio = 2;
    balance = false;
    log = false;
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

