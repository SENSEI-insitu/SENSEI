#include "VortexDataAdaptor.h"
#include "MeshMetadata.h"
#include "Error.h"

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkRectilinearGrid.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>

#define REPRESENT_VTK_AMR
#ifdef REPRESENT_VTK_AMR
#include <vtkAMRBox.h>
#include <vtkAMRInformation.h>
#include <vtkOverlappingAMR.h>
#include <vtkUniformGrid.h>
#endif

#include "simulation_data.h"
#include "patch.h"

static const char *arrname = "vortex";

struct VortexDataAdaptor::DInternals
{
#ifdef REPRESENT_VTK_AMR
  vtkSmartPointer<vtkOverlappingAMR> Mesh;
#else
  vtkSmartPointer<vtkMultiBlockDataSet> Mesh;
#endif
  simulation_data *sim;
};

//-----------------------------------------------------------------------------
senseiNewMacro(VortexDataAdaptor);

//-----------------------------------------------------------------------------
VortexDataAdaptor::VortexDataAdaptor() :
  Internals(new VortexDataAdaptor::DInternals())
{
}

//-----------------------------------------------------------------------------
VortexDataAdaptor::~VortexDataAdaptor()
{
  delete this->Internals;
}

//-----------------------------------------------------------------------------
void VortexDataAdaptor::Initialize(simulation_data *sim)
{
  DInternals& internals = (*this->Internals);
  internals.sim = sim;
 
  this->ReleaseData();
}

//-----------------------------------------------------------------------------
int VortexDataAdaptor::GetMesh(const std::string &meshName, bool /*structureOnly*/,
    vtkDataObject *&mesh)
{
  if (meshName != "AMR_mesh")
    {
    SENSEI_ERROR("the miniapp provides meshes named \"AMR_mesh\"" 
       " you requested \"" << meshName << "\"")
    return -1;
    }

  DInternals& internals = (*this->Internals);
  if (!internals.Mesh)
    {
//#define DEBUG_GET_MESH
#ifdef DEBUG_GET_MESH
    char filename[100];
    sprintf(filename, "getmesh.%04d.txt", internals.sim->par_rank);
    FILE *f = fopen(filename, "wt");
    if(f != NULL)
    {
        for(int i =0; i <internals.sim->max_levels+1; ++i)
            fprintf(f, "npatches_per_level[%d] = %d\n", i, internals.sim->npatches_per_level[i]);
    }
#endif

    // Make the dataset, set the blocks per level. The blocks per level is the
    // global number of patches per level.
    internals.Mesh = vtkSmartPointer<vtkOverlappingAMR>::New();
    internals.Mesh->Initialize(internals.sim->max_levels+1,
                               internals.sim->npatches_per_level);

    // Set the origin. Use the origin of patch 0, the root patch. All ranks
    // compute the root patch in this simulation.
    double origin0[3];
    origin0[0] = internals.sim->patch.window[0];
    origin0[1] = internals.sim->patch.window[2];
    origin0[2] = internals.sim->patch.window[4];
    internals.Mesh->SetOrigin(origin0);
#ifdef DEBUG_GET_MESH
    if(f != NULL)
    {
        fprintf(f, "origin = {%lg, %lg, %lg}\n", origin0[0],origin0[1],origin0[2]);
    }
#endif

    // Indicate we have not set the spacing for each level.
    bool *spacingSet = new bool[internals.sim->max_levels];
    for(int i = 0; i < internals.sim->max_levels; ++i)
        spacingSet[i] = false;

    // Now, let's insert local patches into the AMR dataset.
    int np = 0;
    patch_t **patches_this_rank = patch_flat_array(&internals.sim->patch, &np);
#ifdef DEBUG_GET_MESH
    if(f != NULL)
      {
      for(int i = 0; i < np; ++i)
          patch_print(f, patches_this_rank[i]);
      }
#endif
    for(int i = 0; i < np; ++i)
      {
      // Compute just a little information for this patch.
      int low[3], high[3];
      low[0] = patches_this_rank[i]->logical_extents[0];
      low[1] = patches_this_rank[i]->logical_extents[2];
      low[2] = patches_this_rank[i]->logical_extents[4];
      high[0] = patches_this_rank[i]->logical_extents[1];
      high[1] = patches_this_rank[i]->logical_extents[3];
      high[2] = patches_this_rank[i]->logical_extents[5];
      int cx = (high[0] - low[0] + 1);
      int cy = (high[1] - low[1] + 1);
      int cz = (high[2] - low[2] + 1);
      double origin[3] = {0., 0., 0.};
      origin[0] = patches_this_rank[i]->window[0];
      origin[1] = patches_this_rank[i]->window[2];
      origin[2] = patches_this_rank[i]->window[4];
      double spacing[3] = {0., 0., 0.};
      spacing[0] = (patches_this_rank[i]->window[1] - 
                    patches_this_rank[i]->window[0]) / double(cx);
      spacing[1] = (patches_this_rank[i]->window[3] - 
                    patches_this_rank[i]->window[2]) / double(cy);
      if(patches_this_rank[i]->nz == 1)
          spacing[2] = 0.001; // Just make it thin for 2D.
      else
      {
          spacing[2] = (patches_this_rank[i]->window[5] - 
                        patches_this_rank[i]->window[4]) / double(cz);
      }

      // Set the spacing for the level if we have not done it yet.
      if(!spacingSet[patches_this_rank[i]->level])
        {
        internals.Mesh->SetSpacing(patches_this_rank[i]->level, spacing);
        spacingSet[patches_this_rank[i]->level] = true;
#ifdef DEBUG_GET_MESH
        if(f != NULL)
          {
          fprintf(f, "Set level %d spacing = {%lg, %lg, %lg}\n",
                  patches_this_rank[i]->level,
                  spacing[0],spacing[1],spacing[2]);
          }
#endif
        }

      // Global patch number
      int domain = patches_this_rank[i]->id;
      // Convert to level, patch id within level.
      unsigned int mylevel, mypatch;
      internals.Mesh->GetLevelAndIndex(domain, mylevel, mypatch);
#ifdef DEBUG_GET_MESH
      if(f != NULL)
        {
        fprintf(f, "Get domain %d: mylevel=%d, mypatch=%d\n",
                domain, (int)mylevel, (int)mypatch);
        }
#endif

      // Skip making actual datasets for duplicate patches not owned by this rank.
      if(patches_this_rank[i]->nowners > 1)
        {
        if(patches_this_rank[i]->owners[0] != internals.sim->par_rank)
          continue;
        }

      // Make a vtkUniformGrid for the current patch.
      vtkUniformGrid *p = vtkUniformGrid::New();
      p->SetOrigin(origin);
      p->SetSpacing(spacing);
      int dims[3] = {0,0,1};
      dims[0] = cx+1;
      dims[1] = cy+1;
      dims[2] = (cz < 2) ? 1 : (cz+1); // Using 2 points here seems to work for 2D.
      p->SetDimensions(dims);
#ifdef DEBUG_GET_MESH
      if(f != NULL)
        {
          fprintf(f, "Set domain %d: mylevel=%d, mypatch=%d, "
                     "origin={%lg, %lg, %lg}, spacing={%lg, %lg, %lg}, "
                     "dims={%d,%d,%d}, patch->nx=%d, patch->ny=%d"
                      "\n",
                  domain, (int)mylevel, (int)mypatch
                  ,origin[0],origin[1],origin[2]
                  ,spacing[0],spacing[1],spacing[2]
                  ,dims[0], dims[1], dims[2],
                  patches_this_rank[i]->nx, patches_this_rank[i]->ny
                 );
        }
#endif
      // If the patch has children, and blank data then expose that data as
      // vtkGhostType.
      vtkUnsignedCharArray *arr = vtkUnsignedCharArray::New();
      arr->SetName("vtkGhostType");
      int sz = patches_this_rank[i]->nx*patches_this_rank[i]->ny*patches_this_rank[i]->nz;
      if(patches_this_rank[i]->blank != nullptr)
        {
        arr->SetArray(patches_this_rank[i]->blank, sz, 1);
        }
      else
        {
        // leaf patches won't have a blank array.
        arr->SetNumberOfTuples(sz);
        memset(arr->GetVoidPointer(0), 0, sz * sizeof(unsigned char));
        }
      p->GetCellData()->AddArray(arr);
      arr->FastDelete();

      // Set the vtkUniformGrid into the AMR dataset.
      vtkAMRBox box(low, high);
      internals.Mesh->SetAMRBox(mylevel, mypatch, box);
      internals.Mesh->SetDataSet(mylevel, mypatch, p);
      p->Delete();
      }

    // Set the refinement ratio for each level.
    for(int i = 0; i < internals.sim->max_levels+1; ++i)
      {
#ifdef DEBUG_GET_MESH
      if(f != NULL)
        {
          fprintf(f, "Set refinement for level %d = %d\n",
                  i, internals.sim->refinement_ratio);
        }
#endif
      internals.Mesh->SetRefinementRatio(i, internals.sim->refinement_ratio);
      }
#ifdef DEBUG_GET_MESH
      if(f != NULL)
        fclose(f);
#endif
    delete [] spacingSet;
    FREE(patches_this_rank);
    }

  mesh = internals.Mesh;
  return 0;
}

//-----------------------------------------------------------------------------
int VortexDataAdaptor::AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName)
{
#ifndef NDEBUG
  if ((association != vtkDataObject::FIELD_ASSOCIATION_CELLS) ||
    (arrayName != arrname) || (meshName != "AMR_mesh"))
    {
    SENSEI_ERROR("the miniapp provides a cell centered array named \"vortex\" "
      " on a mesh named \"AMR_mesh\"")
    return 1;
    }
#else
  (void)meshName;
  (void)association;
  (void)arrayName;
#endif
  int retVal = 1;
  DInternals& internals = (*this->Internals);
  vtkOverlappingAMR *ds = vtkOverlappingAMR::SafeDownCast(mesh);
  // Set the arrays for the local domains.
  int np = 0;
  patch_t **patches_this_rank = patch_flat_array(&internals.sim->patch, &np);
  for(int i = 0; i < np; ++i)
    {
    // Skip any duplicate patches not owned by this rank.
    if(patches_this_rank[i]->nowners > 1)
      {
      if(patches_this_rank[i]->owners[0] != internals.sim->par_rank)
        continue;
      }

    int domain = patches_this_rank[i]->id;
    // Convert to level, patch id within level.
    unsigned int mylevel, mypatch;
    ds->GetLevelAndIndex(domain, mylevel, mypatch);

    vtkDataSet *block = vtkDataSet::SafeDownCast(ds->GetDataSet(mylevel, mypatch));
    if(block)
      {
      vtkDataArray *m = block->GetCellData()->GetArray(arrname);
      if(m == nullptr)
        {
        vtkFloatArray *arr = vtkFloatArray::New();
        arr->SetName(arrname);
        arr->SetArray(patches_this_rank[i]->data, 
                      patches_this_rank[i]->nx*patches_this_rank[i]->ny*patches_this_rank[i]->nz,
                      1);
        block->GetCellData()->SetScalars(arr);
        block->GetCellData()->SetActiveScalars(arrname);
        arr->FastDelete();
        retVal = 0;
        }
      }
    }
  FREE(patches_this_rank);
  return retVal;
}

//-----------------------------------------------------------------------------
int VortexDataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = 1;
  return 0;
}

//-----------------------------------------------------------------------------
int VortexDataAdaptor::GetMeshMetadata(unsigned int id,
  sensei::MeshMetadataPtr &metadata)
{
  if (id == 0)
    {
    metadata->MeshName = "AMR_mesh";
    metadata->GlobalView = true;
    metadata->MeshType = VTK_OVERLAPPING_AMR;
    metadata->BlockType = VTK_IMAGE_DATA;

    // TODO -- fill these in
    /*
    metadata->NumBlocks = ; // total number of patches
    metadata->NumBlocksLocal = ; // num pacthes on this rank

    metadata->NumLevels = ;
    metadata->RefRatio = ; // refinement on each level (use amrex convention)

    if (metadata->Flags.BlockExtentsSet())
      {
      metadata->Extent = ; // level zero index space bounding box
      metadata->BlockExtents = ; // index space bounds of all blocks on all ranks
      }

    if (metadata->Flags.BlockBoundsSet())
      {
      metadata->Bounds = ; // level zero bounding box
      metadata->BlockBounds = ; // index space bounds of all blocks on all ranks
      }

    if (metadata->Flags.BlockSizesSet())
      {
      metadata->NumPoints = ; // total num points all blocks
      metadata->NumCells = ; // total num cells all bocks
      metadata->BlockNumPoints = ; // num points each block
      metadata->BlockNumCells = ; // num cells each blocks
      }

    if (metadata->Flags.BlockOwnerSet())
      {
      metadata->BlockOwner = ; // rank for each block
      }
    */

    metadata->NumArrays = 1;
    metadata->ArrayName = {"mandelbrot"};
    metadata->ArrayCentering = {vtkDataObject::CELL};
    metadata->ArrayType = {VTK_UNSIGNED_CHAR};
    metadata->ArrayComponents = {1};
    return 0;
    }

  SENSEI_ERROR("Failed to get mesh name")
  return -1;
}

//-----------------------------------------------------------------------------
int VortexDataAdaptor::ReleaseData()
{
  DInternals& internals = (*this->Internals);
  internals.Mesh = NULL;
  return 0;
}
