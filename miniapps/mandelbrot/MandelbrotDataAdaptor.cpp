#include "MandelbrotDataAdaptor.h"
#include "MeshMetadata.h"
#include "Profiler.h"
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
#include <vtkUniformGridAMRDataIterator.h>
#include <vtkCompositeDataIterator.h>

#include <vtkAMRBox.h>
#include <vtkAMRInformation.h>
#include <vtkOverlappingAMR.h>
#include <vtkUniformGrid.h>

#include "simulation_data.h"
#include "patch.h"

struct MandelbrotDataAdaptor::DInternals
{
  simulation_data *sim;
  sensei::MeshMetadataPtr metadata;
};

//-----------------------------------------------------------------------------
senseiNewMacro(MandelbrotDataAdaptor);

//-----------------------------------------------------------------------------
MandelbrotDataAdaptor::MandelbrotDataAdaptor() :
  Internals(new MandelbrotDataAdaptor::DInternals())
{
}

//-----------------------------------------------------------------------------
MandelbrotDataAdaptor::~MandelbrotDataAdaptor()
{
  delete this->Internals;
}

//-----------------------------------------------------------------------------
void MandelbrotDataAdaptor::Initialize(simulation_data *sim)
{
  sensei::TimeEvent<64> event("MandelbrotDataAdaptor::Initialize");

  DInternals& internals = (*this->Internals);
  internals.sim = sim;

  this->ReleaseData();
}

//-----------------------------------------------------------------------------
int MandelbrotDataAdaptor::GetMesh(const std::string &meshName,
   bool structureOnly, vtkDataObject *&mesh)
{
  sensei::TimeEvent<64> event("MandelbrotDataAdaptor::GetMesh");

  // structure only flag means provide no geometry. geometry is implicit
  // with block structured amr, hence we can safely ignore this flag
  (void)structureOnly;

  if (meshName != "mesh")
    {
    SENSEI_ERROR("the miniapp provides meshes named \"mesh\""
       " you requested \"" << meshName << "\"")
    return -1;
    }

  DInternals& internals = (*this->Internals);

  // VTK's data model requires a global view of blocks, but this simualtion
  // doesn't provide it. construct it here.
  sensei::MeshMetadataFlags flags;
  flags.SetBlockDecomp();
  flags.SetBlockBounds();
  flags.SetBlockExtents();

  sensei::MeshMetadataPtr mmd = sensei::MeshMetadata::New(flags);
  this->GetMeshMetadata(0, mmd);

  // problem domain information
  double x0[3] = {mmd->Bounds[0], mmd->Bounds[2], 0.0};
  double x1[3] = {mmd->Bounds[1], mmd->Bounds[3], 0.0};

  int nx0[3] = {mmd->Extent[1] - mmd->Extent[0] + 1,
    mmd->Extent[3] - mmd->Extent[2] + 1, 1};

  int rr = mmd->RefRatio[0][0];

  // create the VTK dataset
  vtkSmartPointer<vtkOverlappingAMR> amrMesh =
    vtkSmartPointer<vtkOverlappingAMR>::New();

  amrMesh->Initialize(mmd->NumLevels,
    mmd->BlocksPerLevel.data());

  amrMesh->SetOrigin(x0);

  for (int j = 0; j < mmd->NumLevels; ++j)
    {
    double rfacx = j ? j*mmd->RefRatio[j][0] : 1;
    double rfacy = j ? j*mmd->RefRatio[j][1] : 1;

    double dx[3] = {(x1[0] - x0[0]) / (nx0[0]*rfacx),
      (x1[1] - x0[1]) / (nx0[1]*rfacy), 0.001};

    amrMesh->SetSpacing(j, dx);
    amrMesh->SetRefinementRatio(j, rr);

    int lbid = 0; // level block id
    for(int i = 0; i < mmd->NumBlocks; ++i)
      {
      // go level by level
      if (mmd->BlockLevel[i] != j)
        continue;

      // get patch info for VTK
      int cellExt[6];
      memcpy(cellExt, mmd->BlockExtents[i].data(), 6*sizeof(int));
      int cellExtLow[3] = {cellExt[0], cellExt[2], 0};
      int cellExtHigh[3] = {cellExt[1], cellExt[3], 0};

      // save the global patch number, so we can later identify this
      // patch when we need to add arrays
      int gid = mmd->BlockIds[i];

      // pass metadata describing all boxes, including off rank, into VTK
      vtkAMRBox box(cellExtLow, cellExtHigh);
      amrMesh->SetAMRBox(j, lbid, box);
      amrMesh->SetAMRBlockSourceIndex(j, lbid, gid);

      // skip non local patches
      if (mmd->BlockOwner[i] == internals.sim->par_rank)
        {
        // construct the patches
        int ptExt[6]= {0};
        memcpy(ptExt, cellExt, 6*sizeof(int));
        ptExt[1] += 1;
        ptExt[3] += 1;
        ptExt[5] += 1;

        vtkUniformGrid *p = vtkUniformGrid::New();
        p->SetOrigin(x0);
        p->SetSpacing(dx);
        p->SetExtent(ptExt);

        // Set the vtkUniformGrid into the AMR dataset.
        amrMesh->SetDataSet(j, lbid, p);
        p->Delete();
        }

      lbid += 1;
      }
    }

  mesh = amrMesh;
  mesh->Register(0);

  return 0;
}

//-----------------------------------------------------------------------------
int MandelbrotDataAdaptor::AddGhostCellsArray(vtkDataObject* mesh,
    const std::string &meshName)
{
  sensei::TimeEvent<64> event("MandelbrotDataAdaptor::AddGhostCellsArray");

  if (meshName != "mesh")
    {
    SENSEI_ERROR("the miniapp provides meshes named \"mesh\""
       " you requested \"" << meshName << "\"")
    return -1;
    }

  // get the simulation data
  int np = 0;
  patch_t **patches = patch_flat_array(&this->Internals->sim->patch, &np);

  // walk over all local blocks and zero-copy simulation data into the blocks
  vtkOverlappingAMR *amrMesh = dynamic_cast<vtkOverlappingAMR*>(mesh);

  vtkUniformGridAMRDataIterator *it =
    dynamic_cast<vtkUniformGridAMRDataIterator*>(amrMesh->NewIterator());

  for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextItem())
    {
    int level = it->GetCurrentLevel();
    int index = it->GetCurrentIndex();

    int gid = amrMesh->GetAMRBlockSourceIndex(level, index);

    // get the simulation data
    patch_t *patch = nullptr;
    if (patch_find_patch(patches, np, gid, patch))
      {
      it->Delete();
      patch_free_flat_array(patches);
      SENSEI_ERROR("at level " << level << " index " << index << " no patch " << gid);
      return -1;
      }

    int nxy = patch->nx*patch->ny;
    vtkUnsignedCharArray *arr = vtkUnsignedCharArray::New();
    arr->SetName("vtkGhostType");
    if (patch->blank)
      {
      arr->SetArray(patch->blank, nxy, 1);
      }
    else
      {
      // leaf patches won't have a blank array.
      arr->SetNumberOfTuples(nxy);
      memset(arr->GetVoidPointer(0), 0, nxy*sizeof(unsigned char));
      }

    vtkUniformGrid *block =
      dynamic_cast<vtkUniformGrid*>(it->GetCurrentDataObject());
    block->GetCellData()->AddArray(arr);
    arr->Delete();
    }

  it->Delete();
  patch_free_flat_array(patches);

  return 0;
}

//-----------------------------------------------------------------------------
int MandelbrotDataAdaptor::AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName)
{
  sensei::TimeEvent<64> event("MandelbrotDataAdaptor::AddArray");

  if ((association != vtkDataObject::FIELD_ASSOCIATION_CELLS) ||
    (arrayName != "mandelbrot") || (meshName != "mesh"))
    {
    SENSEI_ERROR("the miniapp provides a cell centered array named \"mandelbrot\" "
      " on a mesh named \"mesh\"")
    return 1;
    }

  // get the simulation data
  int np = 0;
  patch_t **patches = patch_flat_array(&this->Internals->sim->patch, &np);

  // walk over all local blocks and zero-copy simulation data into the blocks
  vtkOverlappingAMR *amrMesh = dynamic_cast<vtkOverlappingAMR*>(mesh);

  vtkUniformGridAMRDataIterator *it =
    dynamic_cast<vtkUniformGridAMRDataIterator*>(amrMesh->NewIterator());

  for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextItem())
    {
    int level = it->GetCurrentLevel();
    int index = it->GetCurrentIndex();

    int gid = amrMesh->GetAMRBlockSourceIndex(level, index);

    // get the simulation data
    patch_t *patch = nullptr;
    if (patch_find_patch(patches, np, gid, patch))
      {
      it->Delete();
      patch_free_flat_array(patches);
      SENSEI_ERROR("at level " << level << " index " << index << " no patch " << gid);
      return -1;
      }

    // pass it into VTK
    vtkUniformGrid *block = dynamic_cast<vtkUniformGrid*>(it->GetCurrentDataObject());
    vtkUnsignedCharArray *arr = vtkUnsignedCharArray::New();
    arr->SetName("mandelbrot");
    arr->SetArray(patch->data, patch->nx*patch->ny, 1);
    block->GetCellData()->SetScalars(arr);
    block->GetCellData()->SetActiveScalars("mandelbrot");
    arr->Delete();
    }

  it->Delete();
  patch_free_flat_array(patches);

  return 0;
}

//-----------------------------------------------------------------------------
int MandelbrotDataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  sensei::TimeEvent<64> event("MandelbrotDataAdaptor::GetNumberOfMeshes");
  numMeshes = 1;
  return 0;
}

//-----------------------------------------------------------------------------
int MandelbrotDataAdaptor::GetMeshMetadata(unsigned int id,
  sensei::MeshMetadataPtr &metadata)
{
  sensei::TimeEvent<64> event("MandelbrotDataAdaptor::GetMeshMetadata");

  if (id != 0)
    {
    SENSEI_ERROR("invalid mesh id " << id)
    return -1;
    }

  DInternals &internals = (*this->Internals);

  metadata->MeshName = "mesh";
  metadata->MeshType = VTK_OVERLAPPING_AMR;
  metadata->BlockType = VTK_UNIFORM_GRID;

  metadata->NumGhostCells = 0;
  metadata->NumGhostNodes = 0;

  metadata->NumArrays = 1;
  metadata->ArrayName = {"mandelbrot"};
  metadata->ArrayCentering = {vtkDataObject::CELL};
  metadata->ArrayType = {VTK_UNSIGNED_CHAR};
  metadata->ArrayComponents = {1};

  metadata->NumLevels = internals.sim->max_levels + 1;

  metadata->RefRatio.resize(metadata->NumLevels,
    std::array<int,3>({2,2,1}));

  metadata->NumBlocks = 0;
  metadata->NumBlocksLocal.resize(1);

  if (metadata->Flags.BlockExtentsSet())
    metadata->Extent = {internals.sim->patch.logical_extents[0],
      internals.sim->patch.logical_extents[1], internals.sim->patch.logical_extents[2],
      internals.sim->patch.logical_extents[3], 0, 0};

  if (metadata->Flags.BlockBoundsSet())
    metadata->Bounds = {internals.sim->patch.window[0],
      internals.sim->patch.window[1], internals.sim->patch.window[2],
      internals.sim->patch.window[3], 0, 0};

  if (metadata->Flags.BlockSizeSet())
    {
    metadata->NumPoints = 0;
    metadata->NumCells = 0;
    }

  int np = 0;
  patch_t **local_patches = patch_flat_array(&internals.sim->patch, &np);

  metadata->BlocksPerLevel.resize(metadata->NumLevels);

  for (int j = 0; j < metadata->NumLevels; ++j)
    {
    for(int i = 0; i < np; ++i)
      {
      // skip non local patches.
      if (local_patches[i]->owners[0] != internals.sim->par_rank)
        continue;

      // work level by level.
      if (local_patches[i]->level != j)
        continue;

      metadata->NumBlocks += 1;
      metadata->NumBlocksLocal[0] += 1;

      metadata->BlocksPerLevel[j] += 1;
      metadata->BlockLevel.push_back(j);

      if (metadata->Flags.BlockDecompSet())
        {
        metadata->BlockOwner.push_back(internals.sim->par_rank);
        metadata->BlockIds.push_back(local_patches[i]->id);
        }

      if (metadata->Flags.BlockSizeSet())
        {
        long long np = patch_num_points(local_patches[i]);
        metadata->BlockNumPoints.push_back(np);
        metadata->NumPoints += np;

        long long nc = patch_num_cells(local_patches[i]);
        metadata->BlockNumCells.push_back(nc);
        metadata->NumCells += nc;
        }

      if (metadata->Flags.BlockExtentsSet())
        metadata->BlockExtents.push_back(std::array<int,6>{{
          local_patches[i]->logical_extents[0], local_patches[i]->logical_extents[1],
          local_patches[i]->logical_extents[2], local_patches[i]->logical_extents[3],
          0, 0}});

      if (metadata->Flags.BlockBoundsSet())
        metadata->BlockBounds.push_back(std::array<double,6>{{
          local_patches[i]->window[0], local_patches[i]->window[1],
          local_patches[i]->window[2], local_patches[i]->window[3],
          0.0, 0.0}});
      }
    }

  patch_free_flat_array(local_patches);

  // AMR data is always to be a global view.
  metadata->GlobalizeView(this->GetCommunicator());

  return 0;
}

//-----------------------------------------------------------------------------
int MandelbrotDataAdaptor::ReleaseData()
{
  sensei::TimeEvent<64> event("MandelbrotDataAdaptor::ReleaseData");
  return 0;
}
