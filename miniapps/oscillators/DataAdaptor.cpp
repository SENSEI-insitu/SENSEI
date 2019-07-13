#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "Error.h"

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>
#include <vtkIntArray.h>
#include <vtkImageData.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPolyData.h>
#include <vtkNew.h>

#include <diy/master.hpp>


static
long getBlockNumCells(const diy::DiscreteBounds &ext)
{
  return (ext.max[0] - ext.min[0] + 1)*
   (ext.max[1] - ext.min[1] + 1)*(ext.max[2] - ext.min[2] + 1);
}

static
long getBlockNumPoints(const diy::DiscreteBounds &ext)
{
  return (ext.max[0] - ext.min[0] + 2)*
   (ext.max[1] - ext.min[1] + 2)*(ext.max[2] - ext.min[2] + 2);
}

static
void getBlockBounds(const diy::DiscreteBounds &ext,
  const double x0[3], const double dx[3], double *bounds)
{
  bounds[0] = x0[0] + dx[0]*ext.min[0];
  bounds[1] = x0[0] + dx[0]*(ext.max[0] + 1);
  bounds[2] = x0[1] + dx[1]*ext.min[1];
  bounds[3] = x0[1] + dx[1]*(ext.max[1] + 1);
  bounds[4] = x0[2] + dx[2]*ext.min[2];
  bounds[5] = x0[2] + dx[2]*(ext.max[2] + 1);
}

static
void getBlockExtent(const diy::DiscreteBounds &db, int *ext)
{
  // converts from DIY layout to VTK
  ext[0] = db.min[0];
  ext[1] = db.max[0];
  ext[2] = db.min[1];
  ext[3] = db.max[1];
  ext[4] = db.min[2];
  ext[5] = db.max[2];
}

static
vtkImageData *newCartesianBlock(double *origin,
  double *spacing, const diy::DiscreteBounds &cellExts,
  bool structureOnly)
{
  vtkImageData *id = vtkImageData::New();

  if (!structureOnly)
    {
    id->SetOrigin(origin);
    id->SetSpacing(spacing);
    id->SetExtent(cellExts.min[0], cellExts.max[0]+1,
      cellExts.min[1], cellExts.max[1]+1, cellExts.min[2],
      cellExts.max[2]+1);
    }

  return id;
}

static
vtkUnstructuredGrid *newUnstructuredBlock(const double *origin,
  const double *spacing, const diy::DiscreteBounds &cellExts,
  bool structureOnly)
{
  vtkUnstructuredGrid *ug = vtkUnstructuredGrid::New();

  if (!structureOnly)
    {
    // Add points.
    int nx = cellExts.max[0] - cellExts.min[0] + 1 + 1;
    int ny = cellExts.max[1] - cellExts.min[1] + 1 + 1;
    int nz = cellExts.max[2] - cellExts.min[2] + 1 + 1;

    vtkPoints *pts = vtkPoints::New();
    pts->SetNumberOfPoints(nx*ny*nz);

    vtkIdType idx = 0;

    for(int k = cellExts.min[2]; k <= cellExts.max[2]+1; ++k)
      {
      double z = origin[2] + spacing[2]*k;
      for(int j = cellExts.min[1]; j <= cellExts.max[1]+1; ++j)
        {
        double y = origin[1] + spacing[1]*j;
        for(int i = cellExts.min[0]; i <= cellExts.max[0]+1; ++i)
          {
          double x = origin[0] + spacing[0]*i;
          pts->SetPoint(idx++, x,y,z);
          }
        }
      }

    ug->SetPoints(pts);
    pts->Delete();

    // Add cells
    int ncx = nx - 1;
    int ncy = ny - 1;
    int ncz = nz - 1;
    vtkIdType ncells = ncx*ncy*ncz;
    vtkIdTypeArray *nlist = vtkIdTypeArray::New();
    nlist->SetNumberOfValues(ncells * 9);
    vtkUnsignedCharArray *cellTypes = vtkUnsignedCharArray::New();
    cellTypes->SetNumberOfValues(ncells);
    vtkIdTypeArray *cellLocations = vtkIdTypeArray::New();
    cellLocations->SetNumberOfValues(ncells);

    vtkIdType *nl = nlist->GetPointer(0);
    unsigned char *ct = cellTypes->GetPointer(0);
    vtkIdType *cl = cellLocations->GetPointer(0);
    int nxny = nx*ny;
    int offset = 0;
    for(int k = 0; k < ncz; ++k)
    for(int j = 0; j < ncy; ++j)
    for(int i = 0; i < ncx; ++i)
      {
      *ct++ = VTK_HEXAHEDRON;
      *cl++ = offset;
      offset += 9;

      nl[0] = 8;
      nl[1] = (k) * nxny + j*nx + i;
      nl[2] = (k+1) * nxny + j*nx + i;
      nl[3] = (k+1) * nxny + j*nx + i + 1;
      nl[4] = (k) * nxny + j*nx + i + 1;
      nl[5] = (k) * nxny + (j+1)*nx + i;
      nl[6] = (k+1) * nxny + (j+1)*nx + i;
      nl[7] = (k+1) * nxny + (j+1)*nx + i + 1;
      nl[8] = (k) * nxny + (j+1)*nx + i + 1;
      nl += 9;
      }

    vtkCellArray *cells = vtkCellArray::New();
    cells->SetCells(ncells, nlist);
    nlist->Delete();
    ug->SetCells(cellTypes, cellLocations, cells);
    cellTypes->Delete();
    cellLocations->Delete();
    cells->Delete();
  }

  return ug;
}

static
vtkPolyData *newParticleBlock(const std::vector<Particle> *particles,
  bool structureOnly)
{
  vtkPolyData *block = vtkPolyData::New();

  if (structureOnly)
    return block;

  vtkNew<vtkPoints> points;
  vtkNew<vtkCellArray> cells;
  points->Allocate(particles->size());
  cells->Allocate(particles->size());

  vtkIdType pointId = 0;
  for (const auto &p : *particles)
    {
    points->InsertNextPoint(p.position[0], p.position[1], p.position[2]);
    cells->InsertNextCell(1, &pointId);
    ++pointId;
    }
  block->SetPoints(points.Get());
  block->SetVerts(cells.Get());

  return block;
}

static
int newParticleArray(const std::vector<Particle> &particles,
  const std::string &arrayName, vtkFloatArray *&fa)
{
  enum {PID, VEL, VELMAG};

  fa = vtkFloatArray::New();

  int aid = PID;
  if (arrayName == "pid")
    {
    aid = PID;
    }
  else if (arrayName == "velocity")
    {
    aid = VEL;
    fa->SetNumberOfComponents(3);
    }
  else if (arrayName == "velocityMagnitude")
    {
    aid = VELMAG;
    }
  else
    {
    SENSEI_ERROR("Invalid array name \"" << arrayName << "\"")
    return -1;
    }

  unsigned int np = particles.size();

  fa->SetName(arrayName.c_str());
  fa->SetNumberOfTuples(np);

  float *pfa = fa->GetPointer(0);

  for (unsigned int i = 0; i < np; ++i)
    {
    switch (aid)
      {
      case PID:
        pfa[i] = particles[i].id;
        break;
      case VEL:
        pfa[3*i] = particles[i].velocity[0];
        pfa[3*i+1] = particles[i].velocity[1];
        pfa[3*i+2] = particles[i].velocity[2];
        break;
      case VELMAG:
        {
        float vx = particles[i].velocity[0];
        float vy = particles[i].velocity[1];
        float vz = particles[i].velocity[2];
        pfa[i] = sqrt(vx*vx + vy*vy + vz*vz);
        }
        break;
      }
    }

  return 0;
}

static
vtkUnsignedCharArray *newGhostCellsArray(int *shape,
  diy::DiscreteBounds &cellExt, int ng)
{
    // This sim is a:lways 3D.
    int imin = cellExt.min[0];
    int jmin = cellExt.min[1];
    int kmin = cellExt.min[2];
    int imax = cellExt.max[0];
    int jmax = cellExt.max[1];
    int kmax = cellExt.max[2];
    int nx = imax - imin + 1;
    int ny = jmax - jmin + 1;
    int nz = kmax - kmin + 1;
    int nxny = nx*ny;
    int ncells = nx*ny*nz;

    vtkUnsignedCharArray *g = vtkUnsignedCharArray::New();
    g->SetNumberOfTuples(ncells);
    memset(g->GetVoidPointer(0), 0, sizeof(unsigned char) * ncells);
    g->SetName("vtkGhostType");
    unsigned char *gptr = (unsigned char *)g->GetVoidPointer(0);
    unsigned char ghost = 1;

    if(imin > 0)
    {
        // Set the low I faces to ghosts.
        for(int k = 0; k < nz; ++k)
        for(int j = 0; j < ny; ++j)
        for(int i = 0; i < ng; ++i)
            gptr[k * nxny + j*nx + i] = ghost;
    }
    if(imax < shape[0]-1)
    {
        // Set the high I faces to ghosts.
        for(int k = 0; k < nz; ++k)
        for(int j = 0; j < ny; ++j)
        for(int i = nx-ng; i < nx; ++i)
            gptr[k * nxny + j*nx + i] = ghost;
    }
    if(jmin > 0)
    {
        // Set the low J faces to ghosts.
        for(int k = 0; k < nz; ++k)
        for(int j = 0; j < ng; ++j)
        for(int i = 0; i < nx; ++i)
            gptr[k * nxny + j*nx + i] = ghost;
    }
    if(jmax < shape[1]-1)
    {
        // Set the high J faces to ghosts.
        for(int k = 0; k < nz; ++k)
        for(int j = ny-ng; j < ny; ++j)
        for(int i = 0; i < nx; ++i)
            gptr[k * nxny + j*nx + i] = ghost;
    }
    if(kmin > 0)
    {
        // Set the low K faces to ghosts.
        for(int k = 0; k < ng; ++k)
        for(int j = 0; j < ny; ++j)
        for(int i = 0; i < nx; ++i)
            gptr[k * nxny + j*nx + i] = ghost;
    }
    if(kmax < shape[2]-1)
    {
        // Set the high K faces to ghosts.
        for(int k = nz-ng; k < nz; ++k)
        for(int j = 0; j < ny; ++j)
        for(int i = 0; i < nx; ++i)
            gptr[k * nxny + j*nx + i] = ghost;
    }

    return g;
}

namespace oscillators
{

struct DataAdaptor::InternalsType
{
  InternalsType() : NumBlocks(0), Origin{},
    Spacing{1,1,1}, Shape{}, NumGhostCells(0) {}

  long NumBlocks;                                   // total number of blocks on all ranks
  diy::DiscreteBounds DomainExtent;                 // global index space
  std::map<long, diy::DiscreteBounds> BlockExtents; // local block extents, indexed by global block id
  std::map<long, float*> BlockData;                 // local data array, indexed by block id
  std::map<long, const std::vector<Particle>*> ParticleData;

  double Origin[3];                                 // lower left corner of simulation domain
  double Spacing[3];                                // mesh spacing

  int Shape[3];
  int NumGhostCells;                                // number of ghost cells
};

//-----------------------------------------------------------------------------
senseiNewMacro(DataAdaptor);

//-----------------------------------------------------------------------------
DataAdaptor::DataAdaptor() :
  Internals(new DataAdaptor::InternalsType())
{
}

//-----------------------------------------------------------------------------
DataAdaptor::~DataAdaptor()
{
  delete this->Internals;
}

//-----------------------------------------------------------------------------
void DataAdaptor::Initialize(size_t nblocks, size_t n_local_blocks,
  float *origin, float *spacing, int domain_shape_x, int domain_shape_y,
  int domain_shape_z, int *gid, int *from_x, int *from_y, int *from_z,
  int *to_x, int *to_y, int *to_z, int *shape, int ghostLevels)
{
  this->Internals->NumBlocks = nblocks;

  for (int i = 0; i < 3; ++i)
    this->Internals->Origin[i] = origin[i];

  for (int i = 0; i < 3; ++i)
    this->Internals->Spacing[i] = spacing[i];

  for (int i = 0; i < 3; ++i)
    this->Internals->Shape[i] = shape[i];

  this->Internals->NumGhostCells = ghostLevels;

  this->SetDomainExtent(0, domain_shape_x-1, 0,
    domain_shape_y-1, 0, domain_shape_z-1);

  for (size_t cc=0; cc < n_local_blocks; ++cc)
    {
    this->SetBlockExtent(gid[cc],
      from_x[cc], to_x[cc], from_y[cc], to_y[cc],
      from_z[cc], to_z[cc]);
    }
}

//-----------------------------------------------------------------------------
void DataAdaptor::SetBlockExtent(int gid, int xmin, int xmax, int ymin,
   int ymax, int zmin, int zmax)
{
  this->Internals->BlockExtents[gid].min[0] = xmin;
  this->Internals->BlockExtents[gid].min[1] = ymin;
  this->Internals->BlockExtents[gid].min[2] = zmin;

  this->Internals->BlockExtents[gid].max[0] = xmax;
  this->Internals->BlockExtents[gid].max[1] = ymax;
  this->Internals->BlockExtents[gid].max[2] = zmax;
}

//-----------------------------------------------------------------------------
void DataAdaptor::SetDomainExtent(int xmin, int xmax, int ymin,
   int ymax, int zmin, int zmax)
{
  this->Internals->DomainExtent.min[0] = xmin;
  this->Internals->DomainExtent.min[1] = ymin;
  this->Internals->DomainExtent.min[2] = zmin;

  this->Internals->DomainExtent.max[0] = xmax;
  this->Internals->DomainExtent.max[1] = ymax;
  this->Internals->DomainExtent.max[2] = zmax;
}

//-----------------------------------------------------------------------------
void DataAdaptor::SetBlockData(int gid, float* data)
{
  this->Internals->BlockData[gid] = data;
}

//-----------------------------------------------------------------------------
void DataAdaptor::SetParticleData(int gid, const std::vector<Particle> &particles)
{
  this->Internals->ParticleData[gid] = &particles;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetMesh(const std::string &meshName, bool structureOnly,
    vtkDataObject *&mesh)
{
  mesh = nullptr;

  if ((meshName != "mesh") && (meshName != "ucdmesh") && (meshName != "particles"))
    {
    SENSEI_ERROR("the miniapp provides meshes named \"mesh\", \"ucdmesh\","
      " and \"particles\". you requested \"" << meshName << "\"")
    return -1;
    }

  int particleBlocks = meshName == "particles";
  int unstructuredBlocks = particleBlocks ? 0 : meshName == "ucdmesh";

  vtkMultiBlockDataSet *mb = vtkMultiBlockDataSet::New();
  mb->SetNumberOfBlocks(this->Internals->NumBlocks);

  auto it = this->Internals->BlockExtents.begin();
  auto end = this->Internals->BlockExtents.end();
  for (; it != end; ++it)
    {
    if (particleBlocks)
      {
      vtkPolyData *pd =
        newParticleBlock(this->Internals->ParticleData[it->first],
        structureOnly);

      mb->SetBlock(it->first, pd);
      pd->Delete();
      }
    else if (unstructuredBlocks)
      {
      vtkUnstructuredGrid *ug = newUnstructuredBlock(this->Internals->Origin,
        this->Internals->Spacing, it->second, structureOnly);

      mb->SetBlock(it->first, ug);
      ug->Delete();
      }
    else
      {
      vtkImageData *id = newCartesianBlock(this->Internals->Origin,
        this->Internals->Spacing, it->second, structureOnly);

      mb->SetBlock(it->first, id);
      id->Delete();
      }
    }

  mesh = mb;

  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName)
{
  vtkMultiBlockDataSet *mb = dynamic_cast<vtkMultiBlockDataSet*>(mesh);
  if (!mb)
    {
    SENSEI_ERROR("unexpected mesh type "
      << (mesh ? mesh->GetClassName() : "nullptr"))
    return -1;
    }

  enum {BLOCK, PARTICLE};
  int meshId = BLOCK;
  if ((meshName == "mesh") || (meshName == "ucdmesh"))
    {
    meshId = BLOCK;
    if ((arrayName != "data") || (association != vtkDataObject::CELL))
      {
      SENSEI_ERROR("mesh \"" << meshName
        << "\" only has cell data array named \"data\"")
      return -1;
      }
    }
  else if (meshName == "particles")
    {
    meshId = PARTICLE;
    if (association != vtkDataObject::POINT)
      {
      SENSEI_ERROR("mesh \"particles\" only has point data")
      return -1;
      }
    if ((arrayName != "velocity") && (arrayName != "velocityMagnitude") &&
      (arrayName != "id"))
      {
      SENSEI_ERROR("Invalid particle mesh array \"" << arrayName << "\"")
      return -1;
      }
    }
  else
    {
    SENSEI_ERROR("Invalid mesh name \"" << meshName << "\"")
    return -1;
    }

  auto it = this->Internals->BlockData.begin();
  auto end = this->Internals->BlockData.end();
  for (; it != end; ++it)
    {
    // this code is the same for the Cartesian and unstructured blocks
    // because they both have the same number of cells and are in the
    // same order
    vtkDataObject *blk = mb->GetBlock(it->first);
    if (!blk)
      {
      SENSEI_ERROR("encountered empty block at index " << it->first)
      return -1;
      }

    vtkFloatArray *fa = nullptr;
    vtkDataSetAttributes *dsa = nullptr;

    if (meshId == BLOCK)
      {
      dsa = blk->GetAttributes(vtkDataObject::CELL);
      vtkIdType nCells = getBlockNumCells(this->Internals->BlockExtents[it->first]);

      // zero coopy the array
      fa = vtkFloatArray::New();
      fa->SetName("data");
      fa->SetArray(it->second, nCells, 1);
      }
    else
      {
      dsa = blk->GetAttributes(vtkDataObject::POINT);
      newParticleArray(*this->Internals->ParticleData[it->first], arrayName, fa);
      }

    dsa->AddArray(fa);
    fa->Delete();
    }

  return 0;
}


//----------------------------------------------------------------------------
int DataAdaptor::AddGhostCellsArray(vtkDataObject *mesh, const std::string &meshName)
{
  if ((meshName != "mesh") && (meshName != "ucdmesh"))
    {
    SENSEI_ERROR("the miniapp provides meshes \"mesh\" and \"ucdmesh\".")
    return -1;
    }

  vtkMultiBlockDataSet *mb = dynamic_cast<vtkMultiBlockDataSet*>(mesh);
  if (!mb)
    {
    SENSEI_ERROR("unexpected mesh type "
      << (mesh ? mesh->GetClassName() : "nullptr"))
    return -1;
    }

  auto it = this->Internals->BlockExtents.begin();
  auto end = this->Internals->BlockExtents.end();
  for (; it != end; ++it)
    {
    // this code is the same for the Cartesian and unstructured blocks
    // because they both have the same number of cells and are in the
    // same order
    vtkDataObject *blk = mb->GetBlock(it->first);
    if (!blk)
      {
      SENSEI_ERROR("encountered empty block at index " << it->first)
      return -1;
      }

    vtkDataSetAttributes *dsa = blk->GetAttributes(vtkDataObject::CELL);

    vtkUnsignedCharArray *ga = newGhostCellsArray(this->Internals->Shape,
      it->second, this->Internals->NumGhostCells);

    dsa->AddArray(ga);
    ga->Delete();
    }

  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = 2;
  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &metadata)
{
  if (id > 1)
    {
    SENSEI_ERROR("invalid mesh id " << id)
    return -1;
    }

  int rank = 0;
  int nRanks = 1;

  MPI_Comm_rank(this->GetCommunicator(), &rank);
  MPI_Comm_size(this->GetCommunicator(), &nRanks);

  // this exercises the multimesh api
  // mesh 0 is a multiblock with uniform Cartesian blocks
  // mesh 1 is a multiblock with unstructured blocks
  // otherwise the meshes are identical
  int nBlocks = this->Internals->BlockData.size();

  metadata->MeshName = (id == 0 ? "mesh" : "ucdmesh");

  metadata->MeshType = VTK_MULTIBLOCK_DATA_SET;
  metadata->BlockType = (id == 0 ? VTK_IMAGE_DATA : VTK_UNSTRUCTURED_GRID);
  metadata->NumBlocks = this->Internals->NumBlocks;
  metadata->NumBlocksLocal = {nBlocks};
  metadata->NumGhostCells = this->Internals->NumGhostCells;
  metadata->NumArrays = 1;
  metadata->ArrayName = {"data"};
  metadata->ArrayCentering = {vtkDataObject::CELL};
  metadata->ArrayComponents = {1};
  metadata->ArrayType = {VTK_FLOAT};
  metadata->StaticMesh = 1;

  if ((id == 0) && metadata->Flags.BlockExtentsSet())
    {
    std::array<int,6> ext;
    getBlockExtent(this->Internals->DomainExtent, ext.data());
    metadata->Extent = std::move(ext);

    metadata->BlockExtents.reserve(nBlocks);

    auto it = this->Internals->BlockExtents.begin();
    auto end = this->Internals->BlockExtents.end();
    for (; it != end; ++it)
      {
      getBlockExtent(it->second, ext.data());
      metadata->BlockExtents.emplace_back(std::move(ext));
      }
    }

  if (metadata->Flags.BlockBoundsSet())
    {
    std::array<double,6> bounds;
    getBlockBounds(this->Internals->DomainExtent,
      this->Internals->Origin, this->Internals->Spacing,
      bounds.data());
    metadata->Bounds = std::move(bounds);

    metadata->BlockBounds.reserve(nBlocks);

    auto it = this->Internals->BlockExtents.begin();
    auto end = this->Internals->BlockExtents.end();
    for (; it != end; ++it)
      {
      getBlockBounds(it->second, this->Internals->Origin,
        this->Internals->Spacing, bounds.data());
      metadata->BlockBounds.emplace_back(std::move(bounds));
      }
    }

  if (metadata->Flags.BlockSizeSet())
    {
    auto it = this->Internals->BlockExtents.begin();
    auto end = this->Internals->BlockExtents.end();
    for (; it != end; ++it)
      {
      long nCells = getBlockNumCells(it->second);
      long nPts = getBlockNumPoints(it->second);

      metadata->BlockNumCells.push_back(nCells);
      metadata->BlockNumPoints.push_back(nPts);

      if (id == 1) // unctructured only
        metadata->BlockCellArraySize.push_back(9*nCells);
      }
    }

  if (metadata->Flags.BlockDecompSet())
    {
    auto it = this->Internals->BlockExtents.begin();
    auto end = this->Internals->BlockExtents.end();
    for (; it != end; ++it)
      {
      metadata->BlockOwner.push_back(rank);
      metadata->BlockIds.push_back(it->first);
      }
    }

  if (metadata->Flags.BlockArrayRangeSet())
    {
    float gmin = std::numeric_limits<float>::max();
    float gmax = std::numeric_limits<float>::lowest();
    std::map<long, float*>::iterator it = this->Internals->BlockData.begin();
    std::map<long, float*>::iterator end = this->Internals->BlockData.end();
    for (; it != end; ++it)
      {
      unsigned long nCells = getBlockNumCells(this->Internals->BlockExtents[it->first]);
      float *pdata = it->second;
      float bmin = std::numeric_limits<float>::max();
      float bmax = std::numeric_limits<float>::lowest();
      for (unsigned long i = 0; i < nCells; ++i)
        {
        bmin = std::min(bmin, pdata[i]);
        bmax = std::max(bmax, pdata[i]);
        }
      gmin = std::min(gmin, bmin);
      gmax = std::max(gmax, bmax);
      std::vector<std::array<double,2>> blkRange{{bmin,bmax}};
      metadata->BlockArrayRange.push_back(blkRange);
      }
    metadata->ArrayRange.push_back({gmin, gmax});
    }

  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::ReleaseData()
{
  return 0;
}

}
