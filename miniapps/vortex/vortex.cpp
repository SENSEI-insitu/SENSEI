// Vortex example adapted from Mandelbrot
 
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <sstream>
#include <string>
#include <iostream>

#include <mpi.h>

#include "patch.h"
#include "simulation_data.h"
#include "senseiConfig.h"
#ifdef ENABLE_SENSEI
#include <vtkNew.h>
#include <vtkSmartPointer.h>

#include <timer/Timer.h>
#include <ConfigurableAnalysis.h>
#include "VortexDataAdaptor.h"
#endif

//*****************************************************************************
// Code for calculating data values
//*****************************************************************************

//
// EPD vortex calculator
//

/* Note to Earl:

   I don't know the 3D derivation for the vortex stuff you are doing so
   I guessed. I made it possible to make multiple vortices, each with a 
   location, gamma, radius, velocity. See simulation_data.cpp for setting
   these parameters. They would come from an input deck if we had time.

   The data are still cell-centered, which might not be ideal for this use case.

   If we need to add more fields besides the vortex field, we can modify
   patch.h and patch.cpp. See comments there. Of course, adding new fields
   would mean also modifying VortexDataAdaptor.cpp to expose more fields.
 */

float vortex(float x, float y, float z, simulation_data *sim)
{
    float umax = 1.0;
    float umin = 0.0;
    // 10m/s => M=0.029 @ 80m hub height

    // first set the rectilinear flow
    float u = 0.0;
    float v = 0.0;
    float w = 0.0;

    //
    //now add in the vortices
    //
    for(int i = 0; i < sim->nVortex; ++i)
    {
       float dx = x - sim->vortices[i].location[0];
       float dy = y - sim->vortices[i].location[1];
       float dz = z - sim->vortices[i].location[2];

       // TODO: reformulate the vortex for 3D

       float rlocal=sqrt(dx*dx + dy*dy + dz*dz);
       float theta=atan2(dy,dx);
       float rterm = rlocal/(sim->vortices[i].radius*sim->vortices[i].radius + rlocal*rlocal);
       float utheta = sim->vortices[i].gamma/(2*M_PI)*rterm;

       v += utheta*cos(theta);
       u += -utheta*sin(theta);
       // w += ...
    }

    float umag = sqrt(u*u+v*v+w*w);
    float value = (umag-umin)/(umax-umin);
    return value;
}

void 
calculate_data(patch_t *patch, simulation_data *sim)
{
    float *data = patch->data;

    // Compute x0,x1, y0,y1, z0,z1 which help us locate cell centers. 
    float cellWidth = (patch->window[1] - patch->window[0]) / ((float)patch->nx);
    float x0 = patch->window[0] + cellWidth / 2.f;
    float x1 = patch->window[1] - cellWidth / 2.f;
    float cellHeight = (patch->window[3] - patch->window[2]) / ((float)patch->ny);
    float y0 = patch->window[2] + cellHeight / 2.f;
    float y1 = patch->window[3] - cellHeight / 2.f;
    float cellDepth = (patch->window[5] - patch->window[4]) / ((float)patch->nz);
    float z0 = patch->window[4] + cellDepth / 2.f;
    float z1 = patch->window[5] - cellDepth / 2.f;

    for(int k = 0; k < patch->nz; ++k)
    {
        float tz = (float)k / (float)(patch->nz - 1);
        float z = z0 + tz * (z1 - z0);
        for(int j = 0; j < patch->ny; ++j)
        {
            float ty = (float)j / (float)(patch->ny - 1);
            float y = y0 + ty * (y1 - y0);
            for(int i = 0; i < patch->nx; ++i)
            {
                float tx = (float)i / (float)(patch->nx - 1);
                float x = x0 + tx * (x1 - x0);

                *data++ = vortex(x, y, z, sim);
            }
        }
    }
}

//*****************************************************************************
// Code for helping calculate AMR refinement
//*****************************************************************************

// Blends the neighbors of the i,j,k element.
float
neighbors(patch_t *patch, int i, int j, int k)
{
    const float kernel2d[3][3] = {
    {0.08f, 0.17f, 0.08f},
    {0.17f, 0.f,   0.17f},
    {0.08f, 0.17f, 0.08f}
    };

    // NOTE: We could play with the numerators for these weights.
    const float face   = 0.6f  / 6.f;
    const float corner = 0.15f / 8.f;
    const float edge   = 0.25f / 12.f;
    const float kernel3d[3][3][3] = {
      {
        {corner, edge, corner},
        {edge,   face, edge},
        {corner, edge, corner},
      },
      {
        {edge, face, edge},
        {face, 0.f, face},
        {edge, face, edge},
      },
      {
        {corner, edge, corner},
        {edge,   face, edge},
        {corner, edge, corner},
      }
    };

    float sum = 0.;
    if(patch->nz == 1)
    {
        for(int jj = 0; jj < 3; ++jj)
        {
            int J = j + jj - 1;
            for(int ii = 0; ii < 3; ++ii)
            {
                int I = i + ii - 1;
                float value = (float)patch->data[J*patch->nx+I];
                sum += value * kernel2d[jj][ii];
            }
        }
    }
    else if(patch->nz >= 3) // need layers
    {
        for(int kk = 0; kk < 3; ++kk)
        {
            int K = k + kk - 1;
            int koffset = K * patch->nx*patch->ny;
            for(int jj = 0; jj < 3; ++jj)
            {
                int J = j + jj - 1;
                int offset = koffset + J*patch->nx;
                for(int ii = 0; ii < 3; ++ii)
                {
                    int I = i + ii - 1;
                    int idx = offset + I;
                    sum += patch->data[idx] * kernel3d[kk][jj][ii];
                }
            }
        }
    }
    return sum;
}

// Iterate over cells in the patch and make a mask for those that have
// neighbor value deltas above a threshold.
void
detect_refinement(patch_t *patch, image_t *mask, void *maskcbdata)
{
    simulation_data *sim = (simulation_data *)maskcbdata;

    // Let's look for large differences within a kernel. This lets us
    // figure out areas that we need to refine because they contain
    // features. We set a 1 into the mask for cells that need refinement
    if(patch->nz == 1)
    {
        for(int j = 1; j < patch->ny-1; ++j)
        for(int i = 1; i < patch->nx-1; ++i)
        {
            int index = (j*patch->nx) + i;
            float delta = patch->data[index] - neighbors(patch, i, j, 0);
            if(delta < 0) delta = -delta;
            if(delta > sim->data_refinement_threshold) 
                mask->data[index] = 1;
            else
                mask->data[index] = 0;
        }
    }
    else if(patch->nz >= 3) // Need layers
    {
        for(int k = 1; k < patch->nz-1; ++k)
        {
            int koffset = (k*patch->nx*patch->ny);
            for(int j = 1; j < patch->ny-1; ++j)
            {
                int offset = koffset + (j*patch->nx);
                for(int i = 1; i < patch->nx-1; ++i)
                {
                    int index = offset + i;
                    float delta = patch->data[index] - neighbors(patch, i, j, k);
                    if(delta < 0) delta = -delta;
                    if(delta > sim->data_refinement_threshold) 
                        mask->data[index] = 1;
                    else
                        mask->data[index] = 0;
                }
            }
        }
    }
}

//*****************************************************************************
// Calculate using AMR
//*****************************************************************************

//#define DO_LOG
#ifdef DO_LOG
FILE *debuglog = NULL;

void
log_patches(patch_t *patch, const char *msg)
{
    if(debuglog != NULL)
    {
        fprintf(debuglog, "==========================================================\n");
        fprintf(debuglog, "%s\n", msg);
        fprintf(debuglog, "==========================================================\n");

        int np = 0;
        patch_t **patches_this_rank = patch_flat_array(patch, &np);
        for(int i = 0; i < np; ++i)
            patch_print(debuglog, patches_this_rank[i]);
        FREE(patches_this_rank);  
    }
}
#endif

#if 1
int
count_local_patches(simulation_data *sim)
{
    int np = 0;
    patch_t **patches_this_rank = patch_flat_array(&sim->patch, &np);
    int count = 0;
    for(int i = 0; i < np; ++i)
    {
        if(patches_this_rank[i]->nowners > 1)
        {
            if(patches_this_rank[i]->owners[0] == sim->par_rank)
                count++;
        }
        else
            count++;
    }
    FREE(patches_this_rank);
    return count;
}

int compare_workload(const void *a, const void *b)
{
    const int *A = (const int *)a;
    const int *B = (const int *)b;
    if(A[1] < B[1])
        return -1;
    else if(A[1] == B[1])
        return 0;
    else
        return 1;
}

// -----------------------------------------------------------------------------
// @brief This routine is called among the owners of a patch to decide who has
//        the most work. The owner list is sorted so ranks with less work appear
//        first in the list so they can get a little more work when we assign
//        patches to ranks.
//
void
sort_owners_by_workload(MPI_Comm comm, simulation_data *sim, patch_t *patch)
{
    // How much work do we have?
    int nlocal_patches = count_local_patches(sim);

    // We need to among just the processors in the owner list for this patch.
    // Let's do point 2 point.
    int tag = 1000, tag2 = 1001;
    int *counts = new int[patch->nowners];
    if(sim->par_rank == patch->owners[0])
    {
        counts[0] = nlocal_patches;
        // Gather to the first
        MPI_Status status;
        for(int i = 1; i < patch->nowners; ++i)
            MPI_Recv(counts + i, 1, MPI_INT, patch->owners[i], tag, comm, &status);
        // Send to the rest
        for(int i = 1; i < patch->nowners; ++i)
            MPI_Send(counts, patch->nowners, MPI_INT, patch->owners[i], tag2, comm);
    }
    else
    {
        MPI_Send(&nlocal_patches, 1, MPI_INT, patch->owners[0], tag, comm);
        MPI_Status status;
        MPI_Recv(counts, patch->nowners, MPI_INT, patch->owners[0], tag2, comm, &status);
    }

    // Now we have counts on all ranks. Sort according to workload so the low
    // work ranks are first where they are more likely to get assigned work
    // due to the mod assignment.
    int *s = new int[patch->nowners * 2];
    for(int i = 0; i < patch->nowners; ++i)
    {
        s[2*i]   = patch->owners[i]; // owner
        s[2*i+1] = counts[i]; // workload
    }
    qsort(s, patch->nowners, 2*sizeof(int), compare_workload);
    delete [] counts;

    // We've sorted based on the workload.
    for(int i = 0; i < patch->nowners; ++i)
        patch->owners[i] = s[2*i];
    delete [] s;
}
#endif

// -----------------------------------------------------------------------------
// @brief Takes the input patch and doles out the subpatches it contains to the
//        ranks that own the input patch.
//
void
assign_patches(MPI_Comm comm, simulation_data *sim, patch_t *patch)
{
    // Decide how patches are assigned to processors. 
    if(patch->nowners > 1)
    {
#ifdef DO_LOG
        fprintf(debuglog, "assign_patches: Current patch owned by %d ranks\n", patch->nowners);
        fprintf(debuglog, "assign_patches: Current patch refined into %d subpatches\n", patch->nsubpatches);
#endif

#if 1
        // Sort the owner list by the total amount of work so the least loaded
        // ranks are first in the list.
        if(sim->balance)
            sort_owners_by_workload(comm, sim, patch);
#endif
        // The current patch exists on more than one rank. Divide its
        // subpatches (if any) among those ranks.
        if(patch->nsubpatches > 0)
        {
            std::vector<int> patches_owned_by_this_rank;
            for(int i = 0; i < patch->nsubpatches; ++i)
            {
                int owner = patch->owners[i % patch->nowners];
                int subpatchIndex = i % patch->nsubpatches;
                patch_add_owner(&patch->subpatches[subpatchIndex], owner);

                if(owner == sim->par_rank)
                {
                    patches_owned_by_this_rank.push_back(subpatchIndex);
#ifdef DO_LOG
                    fprintf(debuglog, "assign_patches: patches owned by this rank: %d\n", subpatchIndex);
#endif
                }
            }

            // Keep just the ones we want on this rank.
            int *keep = ALLOC(patch->nsubpatches, int);
            memset(keep, 0, patch->nsubpatches * sizeof(int));
            for(size_t i = 0; i < patches_owned_by_this_rank.size(); ++i)
                keep[patches_owned_by_this_rank[i]] = 1;

            patch_t *subpatches = ALLOC(patches_owned_by_this_rank.size(), patch_t);
            for(int i = 0, idx = 0; i < patch->nsubpatches; ++i)
            {
                if(keep[i])
                    patch_shallow_copy(&subpatches[idx++], &patch->subpatches[i]);
                else
                    patch_dtor(&patch->subpatches[i]);
            }
            FREE(keep);
            FREE(patch->subpatches);
            patch->subpatches = subpatches;
            patch->nsubpatches = patches_owned_by_this_rank.size();
        }
#ifdef DO_LOG
        else
        {
            fprintf(debuglog, "assign_patches: Current patch has no subpatches\n");
        }
#endif
    }
    else
    {
        // The current patch is not shared among MPI ranks. Therefore, any
        // further subdivision we do is local to this MPI rank. Let's just
        // indicate that each patch has 1 owner.
        for(int i = 0; i < patch->nsubpatches; ++i)
            patch_add_owner(&patch->subpatches[i], sim->par_rank);
    }
}

// -----------------------------------------------------------------------------
// @brief Compute data for a patch. Refine the patch if we're not beyond max levels.
//        The subpatches are divided among ranks that own patch. Then we recurse to
//        compute data for the subpatches.
//
void
calculate_amr_helper(MPI_Comm comm, simulation_data *sim, patch_t *patch, int level)
{
    // Save the level 
    patch->level = level;

    // Calculate the data on this patch 
    patch_alloc_data(patch, patch->nx, patch->ny, patch->nz);
    calculate_data(patch,sim);

    if(level+1 > sim->max_levels)
        return;

    // Examine this patch's data and refine it to populate the
    // patch's subpatches with refined patches. Note that they will not
    // have any data allocated to them yet.
    patch_refine(patch, sim->refinement_ratio, detect_refinement, sim);
#ifdef DO_LOG
    log_patches(patch, "AFTER patch_refine");
#endif

    // Assign the subpatches to MPI ranks.
    assign_patches(comm, sim, patch);
#ifdef DO_LOG
    log_patches(patch, "AFTER assign_patches");
#endif

    // Recurse and compute data on subpatches.
    for(int i = 0; i < patch->nsubpatches; ++i)
    {
        patch_t *p = &patch->subpatches[i];
        //patch_alloc_data(p, p->nx, p->ny);
        calculate_amr_helper(comm, sim, p, level+1);
    }
}

// -----------------------------------------------------------------------------
// @brief Assigns unique ids to the patches across all processors. Only the patches
//        that a processor owns will get patch ids. The remaining patches that are
//        duplicated on a rank remain at id = 0.
//
void
assign_unique_patch_ids(MPI_Comm comm, simulation_data *sim)
{
    // We created the patches such that we have different trees on all ranks.
    // Some of the patches will be duplicated across ranks up the tree to the
    // root patch. We need to assign unique ids to all of these patches
    // across all ranks.

    // Figure a unique list of patches for this rank. None of these should be
    // duplicated among other ranks since we're taking into account the owner
    // list, which should be the same on all ranks. We let the first owner own
    // the domain from a vis perspective. Let's do it by level so we can have
    // domains from levels together.
    if(sim->npatches_per_rank == NULL)
        sim->npatches_per_rank = ALLOC(sim->par_size, int);
    memset(sim->npatches_per_rank, 0, sizeof(int)*sim->par_size);
    FREE(sim->npatches_per_level)
    sim->npatches_per_level = ALLOC(sim->max_levels+1, int);

    int *npatches_per_level = ALLOC(sim->par_size, int);
#ifdef DO_LOG
    if(debuglog != NULL)
    {
        fprintf(debuglog, "==========================================================\n");
        fprintf(debuglog, "assign_unique_patch_ids\n");
        fprintf(debuglog, "==========================================================\n");
    }
#endif
    int np = 0;
    patch_t **patches_this_rank = patch_flat_array(&sim->patch, &np);
    int patch_start = 0;
    for(int level = 0; level < sim->max_levels+1; ++level)
    {
        int count = 0;
        for(int i = 0; i < np; ++i)
        {
            if(patches_this_rank[i]->level == level)
            {
                if(patches_this_rank[i]->nowners > 1)
                {
                   if(patches_this_rank[i]->owners[0] == sim->par_rank)
                       count++;
                }
                else
                    count++;
            }
        }

        // Gather the number of patches per rank this level, let all know.
        MPI_Allgather(&count, 1, MPI_INT, npatches_per_level, 1, MPI_INT, comm);

        // Add to the overall patch counts.
        int patch_count_this_level = 0;
        for(int i = 0; i < sim->par_size; ++i)
        {
            sim->npatches_per_rank[i] += npatches_per_level[i];
            patch_count_this_level += npatches_per_level[i];
        }
        sim->npatches_per_level[level] = patch_count_this_level;

        // Compute a starting patchid for this rank.
        int patchid = patch_start;
        for(int i = 0; i < sim->par_rank; ++i)
            patchid += npatches_per_level[i];
#ifdef DO_LOG
        if(debuglog != NULL)
        {
            fprintf(debuglog, "level %d: count = %d\n", level, count);
            fprintf(debuglog, "level %d: patch_count_this_level = %d\n", level, patch_count_this_level);
            fprintf(debuglog, "level %d: patch_start = %d\n", level, patch_start);
            fprintf(debuglog, "level %d: patchid = %d\n", level, patchid);
        }
#endif
        patch_start += patch_count_this_level;

        // Now, number the local patches at level with the global numbering.
        for(int i = 0; i < np; ++i)
        {
            if(patches_this_rank[i]->level == level)
            {
                if(patches_this_rank[i]->nowners > 1)
                {
                    if(patches_this_rank[i]->owners[0] == sim->par_rank)
                        patches_this_rank[i]->id = patchid++;
                }
                else
                    patches_this_rank[i]->id = patchid++;
            }
        }
    }
    FREE(npatches_per_level);
    FREE(patches_this_rank);
}

// -----------------------------------------------------------------------------
// @brief Compute the patches and the data on them.
//
void
calculate_amr(MPI_Comm comm, simulation_data *sim)
{
#ifdef DO_LOG
    char filename[100];
    sprintf(filename, "patches.%04d.txt", sim->par_rank);
    debuglog = fopen(filename, "wt");
#endif

    // Compute the AMR patches. 
    calculate_amr_helper(comm, sim, &sim->patch, 0);

    // Assign ids to all of the AMR patches. 
    assign_unique_patch_ids(comm, sim);

#ifdef DO_LOG
    if(debuglog != NULL)
    {
        log_patches(&sim->patch, "AFTER assign_unique_patch_ids");
        fclose(debuglog);
    }
#endif
}

// -----------------------------------------------------------------------------
// @brief Handle command line arguments.
//
void
handle_command_line(int argc, char **argv, simulation_data *sim, 
    int &max_iter, std::string &config_file)
{
    for(int i = 1; i < argc; ++i)
    {
        if(strcmp(argv[i], "-f") == 0 && (i+1)<argc)
        {
            config_file = argv[i+1];
            i++;
        }
        else if((strcmp(argv[i], "-i") == 0 ||
                 strcmp(argv[i], "-maxiter") == 0) && (i+1)<argc)
        {
            max_iter = atoi(argv[i+1]);
            i++;
        }
        else if((strcmp(argv[i], "-d") == 0 ||
                 strcmp(argv[i], "-dims") == 0) && (i+1)<argc)
        {
            int ivals[3];
            if(sscanf(argv[i+1], "%d,%d,%d", ivals, ivals+1, ivals+2) == 3)
            {
                sim->dims[0] = ivals[0];
                sim->dims[1] = ivals[1];
                sim->dims[2] = ivals[2];
            }
            i++;
        }
        else if((strcmp(argv[i], "-l") == 0 ||
                 strcmp(argv[i], "-maxlevels") == 0) && (i+1)<argc)
        {
            sim->max_levels = atoi(argv[i+1]);
            i++;
        }
        else if((strcmp(argv[i], "-r") == 0 ||
                 strcmp(argv[i], "-refinement") == 0)&& (i+1)<argc)
        {
            sim->refinement_ratio = atoi(argv[i+1]);
            i++;
        }
        else if((strcmp(argv[i], "-drt") == 0 ||
                 strcmp(argv[i], "-data_refinement_threshold") == 0)&& (i+1)<argc)
        {
            sim->data_refinement_threshold = atof(argv[i+1]);
            i++;
        }
        else if((strcmp(argv[i], "-b") == 0 ||
                 strcmp(argv[i], "-balance") == 0))
        {
            sim->balance = true;
        }
        else if(strcmp(argv[i], "-log") == 0)
        {
            sim->log = true;
        }
    }
}

// TODO -- remove or fix??
// gcc defines a function of the same name, this breaks the build.
// need to rename ours, but it's not currently in use, so we may
// alternatively remove it.
/*void pause()
{
    FILE *f = NULL;
    while((f = fopen("pause.txt", "rt")) == NULL);
    fclose(f);
}*/

//*****************************************************************************
//
// Purpose: This is the main function for the program.
//
// Programmer: Brad Whitlock
// Date:       Thu Mar 19 11:54:04 PDT 2009
//
// Input Arguments:
//   argc : The number of command line arguments.
//   argv : The command line arguments.
//
// Modifications:
//   Brad Whitlock, Thu Apr 19 18:09:16 PDT 2018
//   Ported to SENSEI, parallelized better.
//
// ****************************************************************************

int main(int argc, char **argv)
{
 //   int max_iter = 100;
    int max_iter = 1000;
    std::string config_file("vortex.xml");
    simulation_data sim;

    // Initialize MPI 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &sim.par_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &sim.par_size);

    // Handle any command line args. 
    handle_command_line(argc, argv, &sim, max_iter, config_file);

#ifdef ENABLE_SENSEI
    timer::Initialize();

    timer::MarkStartEvent("vortex::initialize");
    // Initialize in situ
    vtkSmartPointer<VortexDataAdaptor> dataAdaptor;
    dataAdaptor = vtkSmartPointer<VortexDataAdaptor>::New();
    dataAdaptor->Initialize(&sim);
    dataAdaptor->SetDataTimeStep(-1);

    vtkSmartPointer<sensei::ConfigurableAnalysis> analysisAdaptor;
    analysisAdaptor = vtkSmartPointer<sensei::ConfigurableAnalysis>::New();
    analysisAdaptor->SetCommunicator(MPI_COMM_WORLD);
    analysisAdaptor->Initialize(config_file);
    
    timer::MarkEndEvent("vortex::initialize");
#endif
    //pause();

    ofstream log;
    if(sim.log && sim.par_rank == 0)
    {
        log.open("vortex.log");
        for(int i = 0; i < argc; ++i)
           log << argv[i] << " ";
        log << endl;
    }

    // Patch 0 is owned by all ranks. This will be freed by the simulation_data's
    // patch at the end of the run.
    int *patch0_owners = ALLOC(sim.par_size, int);
    for(int i = 0; i < sim.par_size; ++i)
        patch0_owners[i] = i;

    // Iterate.
    for(sim.cycle = 0; sim.cycle < max_iter; ++sim.cycle)
    {
        timer::MarkStartTimeStep(sim.cycle, sim.time);
        if(sim.par_rank == 0)
        {
            std::cout << "Simulating time step: cycle=" << sim.cycle
                      << ", time=" << sim.time << std::endl;
        }

        // Blow away the previous patch data and calculate.
        sim.patch.owners = NULL;
        patch_dtor(&sim.patch);
        patch_ctor(&sim.patch);
        sim.patch.owners = patch0_owners;
        sim.patch.nowners = sim.par_size;
        sim.patch.window[0] = sim.window[0];
        sim.patch.window[1] = sim.window[1];
        sim.patch.window[2] = sim.window[2];
        sim.patch.window[3] = sim.window[3];
        sim.patch.window[4] = sim.window[4];
        sim.patch.window[5] = sim.window[5];
        sim.patch.logical_extents[0] = 0;
        sim.patch.logical_extents[1] = sim.dims[0]-1;
        sim.patch.logical_extents[2] = 0;
        sim.patch.logical_extents[3] = sim.dims[1]-1;
        sim.patch.logical_extents[4] = 0;
        sim.patch.logical_extents[5] = sim.dims[2]-1;
        sim.patch.nx = sim.dims[0];
        sim.patch.ny = sim.dims[1];
        sim.patch.nz = sim.dims[2];
#ifdef ENABLE_SENSEI
        timer::MarkStartEvent("vortex::compute");
#endif
        calculate_amr(MPI_COMM_WORLD, &sim);

        if(sim.log && sim.par_rank == 0)
        {
            log << "# patches_per_rank" << sim.cycle << endl;
            for(int i = 0; i < sim.par_size; ++i)
                log << i << " " << sim.npatches_per_rank[i] << std::endl;
        }

#ifdef ENABLE_SENSEI
        timer::MarkEndEvent("vortex::compute");

        // Do in situ 
        dataAdaptor->SetDataTime(sim.time);
        dataAdaptor->SetDataTimeStep(sim.cycle);
        timer::MarkStartEvent("vortex::analyze");
        analysisAdaptor->Execute(dataAdaptor.GetPointer());
        timer::MarkEndEvent("vortex::analyze");

        timer::MarkStartEvent("vortex::analyze::release-data");
        dataAdaptor->ReleaseData();
        timer::MarkEndEvent("vortex::analyze::release-data");
#endif

        // Update
        sim.time += sim.dt;

        // Update vortex locations
        for(int i = 0; i < sim.nVortex; ++i)
        {
            // TODO: maybe they interact and take curved paths...
            sim.vortices[i].location[0] += sim.vortices[i].velocity[0];
            sim.vortices[i].location[1] += sim.vortices[i].velocity[1];
            sim.vortices[i].location[2] += sim.vortices[i].velocity[2];
        }

        timer::MarkEndTimeStep();
    }

    // Cleanup
    if(sim.log && sim.par_rank == 0)
        log.close();
#ifdef ENABLE_SENSEI
    timer::MarkStartEvent("vortex::finalize");
    analysisAdaptor->Finalize();
    analysisAdaptor = NULL;
    dataAdaptor = NULL;
    timer::MarkEndEvent("vortex::finalize");

    timer::Finalize();
#endif
    MPI_Finalize();

    return 0;
}
