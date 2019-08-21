// mandelbrot example adapted from VisIt Libsim examples.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
#include <ConfigurableAnalysis.h>
#include <Timer.h>
#include "MandelbrotDataAdaptor.h"
#endif

//*****************************************************************************
// Code for calculating data values
//*****************************************************************************

class complex
{
public:
    complex() : a(0.), b(0.) { }
    complex(float A, float B) : a(A), b(B) { }
    complex(const complex &obj) : a(obj.a), b(obj.b) { }
    complex operator = (const complex &obj) { a = obj.a; b = obj.b; return *this;}
    complex operator + (const complex &obj) const
    {
        return complex(a + obj.a,  b + obj.b);
    }
    complex operator * (const complex &obj) const
    {
        return complex(a * obj.a - b * obj.b, a * obj.b + b * obj.a);
    }
    float mag2() const
    {
        return a*a + b*b;
    }
    float mag() const
    {
        return sqrtf(a*a + b*b);
    }
private:
    float a,b;
};

#define MAXIT 30

inline unsigned char
mandelbrot(const complex &C)
{
    complex Z;
    for(unsigned char zit = 0; zit < MAXIT; ++zit)
    {
        Z = (Z * Z) + C;
        if(Z.mag2() > 4.f)
            return zit+1;
    }
    return 0;
}

void
calculate_data(patch_t *patch)
{
    unsigned char *data = patch->data;

    // Compute x0, x1 and y0,y1 which help us locate cell centers. 
    float cellWidth = (patch->window[1] - patch->window[0]) / ((float)patch->nx);
    float x0 = patch->window[0] + cellWidth / 2.f;
    float x1 = patch->window[1] - cellWidth / 2.f;
    float cellHeight = (patch->window[3] - patch->window[2]) / ((float)patch->ny);
    float y0 = patch->window[2] + cellHeight / 2.f;
    float y1 = patch->window[3] - cellHeight / 2.f;
    for(int j = 0; j < patch->ny; ++j)
    {
        float ty = (float)j / (float)(patch->ny - 1);
        float y = y0 + ty * (y1 - y0);
        for(int i = 0; i < patch->nx; ++i)
        {
            float tx = (float)i / (float)(patch->nx - 1);
            float x = x0 + tx * (x1 - x0);

            *data++ = mandelbrot(complex(x, y));
        }
    }
}

//*****************************************************************************
// Code for helping calculate AMR refinement
//*****************************************************************************

int
neighbors(patch_t *patch, int i, int j)
{
    const float kernel[3][3] = {
    {0.08f, 0.17f, 0.08f},
    {0.17f, 0.f,   0.17f},
    {0.08f, 0.17f, 0.08f}
    };

    float sum = 0;
    for(int jj = 0; jj < 3; ++jj)
    {
        int J = j + jj - 1;
        for(int ii = 0; ii < 3; ++ii)
        {
            int I = i + ii - 1;
            float value = (float)patch->data[J*patch->nx+I];
            sum += value * kernel[jj][ii];
        }
    }
    return (int)sum;
}

void
detect_refinement(patch_t *patch, image_t *mask)
{
    // Let's look for large differences within a kernel. This lets us
    // figure out areas that we need to refine because they contain
    // features. We set a 1 into the mask for cells that need refinement
    for(int j = 1; j < patch->ny-1; ++j)
        for(int i = 1; i < patch->nx-1; ++i)
        {
            int index = j*patch->nx+i;
            int dval = (int)patch->data[index] - neighbors(patch, i, j);
            if(dval < 0) dval = -dval;
            if(dval > 2) 
                mask->data[index] = 1;
            else
                mask->data[index] = 0;
        }
}

//*****************************************************************************
// Calculate the Mandelbrot set using AMR
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
        // The current patch exists on more than one rank. Divide the
        // refined patch list among those ranks.
        int n = std::max(patch->nowners, patch->nsubpatches);
        std::vector<int> patches_owned_by_this_rank;
        for(int i = 0; i < n; ++i)
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
    patch_alloc_data(patch, patch->nx, patch->ny);
    calculate_data(patch);

    if(level+1 > sim->max_levels)
        return;

    // Examine this patch's data and refine it to populate the
    // patch's subpatches with refined patches. Note that they will not
    // have any data allocated to them yet.
    patch_refine(patch, sim->refinement_ratio, detect_refinement);
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
// this function should be renamed, as gcc has one with same name
// and it now breaks the build. It is not currently used, so I
// commented it out for now.
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
    int max_iter = 100;
    std::string config_file("mandelbrot.xml");
    simulation_data sim;

    // Initialize MPI 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &sim.par_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &sim.par_size);

    // Handle any command line args. 
    handle_command_line(argc, argv, &sim, max_iter, config_file);

#ifdef ENABLE_SENSEI
    sensei::Timer::Initialize();

    sensei::Timer::MarkStartEvent("mandelbrot::initialize");
    // Initialize in situ
    vtkSmartPointer<MandelbrotDataAdaptor> dataAdaptor;
    dataAdaptor = vtkSmartPointer<MandelbrotDataAdaptor>::New();
    dataAdaptor->Initialize(&sim);
    dataAdaptor->SetDataTimeStep(-1);

    vtkSmartPointer<sensei::ConfigurableAnalysis> analysisAdaptor;
    analysisAdaptor = vtkSmartPointer<sensei::ConfigurableAnalysis>::New();
    analysisAdaptor->SetCommunicator(MPI_COMM_WORLD);
    analysisAdaptor->Initialize(config_file);
    sensei::Timer::MarkEndEvent("mandelbrot::initialize");
#endif
    //pause();

    ofstream log;
    if(sim.log && sim.par_rank == 0)
    {
        log.open("mandelbrot.log");
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
        sensei::Timer::MarkStartTimeStep(sim.cycle, sim.time);
        if(sim.par_rank == 0)
        {
            std::cout << "Simulating time step: cycle=" << sim.cycle
                      << ", time=" << sim.time << std::endl;
        }

        const float window0[] = {-1.6f, 0.6f, -1.1f, 1.1f};
#define ORIGINX -1.5f
#define ORIGINY -0.5f
#define WSIZE 0.5f
        const float window1[] = {ORIGINX, ORIGINX + WSIZE, ORIGINY, ORIGINY + WSIZE};
#define NX 256
#define NY 256

        // oscillate between 2 windows 
        float window[4];
        float t = 0.5f * sin(sim.time) + 0.5f;
        window[0] = (1.f - t)*window0[0] + t*window1[0];
        window[1] = (1.f - t)*window0[1] + t*window1[1];
        window[2] = (1.f - t)*window0[2] + t*window1[2];
        window[3] = (1.f - t)*window0[3] + t*window1[3];

        // Blow away the previous patch data and calculate.
        sim.patch.owners = NULL;
        patch_dtor(&sim.patch);
        patch_ctor(&sim.patch);
        sim.patch.owners = patch0_owners;
        sim.patch.nowners = sim.par_size;
        sim.patch.window[0] = window[0];
        sim.patch.window[1] = window[1];
        sim.patch.window[2] = window[2];
        sim.patch.window[3] = window[3];
        sim.patch.logical_extents[0] = 0;
        sim.patch.logical_extents[1] = NX-1;
        sim.patch.logical_extents[2] = 0;
        sim.patch.logical_extents[3] = NY-1;
        sim.patch.nx = NX;
        sim.patch.ny = NY;
#ifdef ENABLE_SENSEI
        sensei::Timer::MarkStartEvent("mandelbrot::compute");
#endif
        calculate_amr(MPI_COMM_WORLD, &sim);

        if(sim.log && sim.par_rank == 0)
        {
            log << "# patches_per_rank" << sim.cycle << endl;
            for(int i = 0; i < sim.par_size; ++i)
                log << i << " " << sim.npatches_per_rank[i] << std::endl;
        }

#ifdef ENABLE_SENSEI
        sensei::Timer::MarkEndEvent("mandelbrot::compute");

        // Do in situ 
        dataAdaptor->SetDataTime(sim.time);
        dataAdaptor->SetDataTimeStep(sim.cycle);
        sensei::Timer::MarkStartEvent("mandelbrot::analyze");
        analysisAdaptor->Execute(dataAdaptor.GetPointer());
        sensei::Timer::MarkEndEvent("mandelbrot::analyze");

        sensei::Timer::MarkStartEvent("mandelbrot::analyze::release-data");
        dataAdaptor->ReleaseData();
        sensei::Timer::MarkEndEvent("mandelbrot::analyze::release-data");
#endif

        sim.time += 0.1;
        sensei::Timer::MarkEndTimeStep();
    }

    // Cleanup
    if(sim.log && sim.par_rank == 0)
        log.close();
#ifdef ENABLE_SENSEI
    sensei::Timer::MarkStartEvent("mandelbrot::finalize");
    analysisAdaptor->Finalize();
    analysisAdaptor = NULL;
    dataAdaptor = NULL;
    sensei::Timer::MarkEndEvent("mandelbrot::finalize");

    sensei::Timer::Finalize();
#endif
    MPI_Finalize();

    return 0;
}
