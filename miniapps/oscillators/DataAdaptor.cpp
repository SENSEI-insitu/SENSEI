#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "Error.h"

#include <svtkCellArray.h>
#include <svtkCellData.h>
#include <svtkPointData.h>
#include <svtkFloatArray.h>
#include <svtkIdTypeArray.h>
#include <svtkIntArray.h>
#include <svtkImageData.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkObjectFactory.h>
#include <svtkPoints.h>
#include <svtkSmartPointer.h>
#include <svtkUnsignedCharArray.h>
#include <svtkUnstructuredGrid.h>
#include <svtkPolyData.h>
#include <svtkNew.h>

#include <sdiy/master.hpp>


static
long getBlockNumCells(const sdiy::DiscreteBounds &ext)
{
  return (ext.max[0] - ext.min[0] + 1)*
   (ext.max[1] - ext.min[1] + 1)*(ext.max[2] - ext.min[2] + 1);
}

static
long getBlockNumPoints(const sdiy::DiscreteBounds &ext)
{
  return (ext.max[0] - ext.min[0] + 2)*
   (ext.max[1] - ext.min[1] + 2)*(ext.max[2] - ext.min[2] + 2);
}

static
void getBlockBounds(const sdiy::DiscreteBounds &ext,
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
void getBlockExtent(const sdiy::DiscreteBounds &db, int *ext)
{
  // converts from DIY layout to SVTK
  ext[0] = db.min[0];
  ext[1] = db.max[0];
  ext[2] = db.min[1];
  ext[3] = db.max[1];
  ext[4] = db.min[2];
  ext[5] = db.max[2];
}

static
svtkImageData *newCartesianBlock(double *origin,
  double *spacing, const sdiy::DiscreteBounds &cellExts,
  bool structureOnly)
{
  svtkImageData *id = svtkImageData::New();

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
svtkUnstructuredGrid *newUnstructuredBlock(const double *origin,
  const double *spacing, const sdiy::DiscreteBounds &cellExts,
  bool structureOnly)
{
  svtkUnstructuredGrid *ug = svtkUnstructuredGrid::New();

  if (!structureOnly)
    {
    // Add points.
    int nx = cellExts.max[0] - cellExts.min[0] + 1 + 1;
    int ny = cellExts.max[1] - cellExts.min[1] + 1 + 1;
    int nz = cellExts.max[2] - cellExts.min[2] + 1 + 1;

    svtkPoints *pts = svtkPoints::New();
    pts->SetDataTypeToDouble();
    pts->SetNumberOfPoints(nx*ny*nz);

    svtkIdType idx = 0;

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
    svtkIdType ncells = ncx*ncy*ncz;

    svtkIdTypeArray *nlist = svtkIdTypeArray::New();
    nlist->SetNumberOfValues(ncells * 8);

    svtkUnsignedCharArray *cellTypes = svtkUnsignedCharArray::New();
    cellTypes->SetNumberOfValues(ncells);

    svtkIdTypeArray *cellLocations = svtkIdTypeArray::New();
    cellLocations->SetNumberOfValues(ncells + 1);

    svtkIdType *nl = nlist->GetPointer(0);
    unsigned char *ct = cellTypes->GetPointer(0);
    svtkIdType *cl = cellLocations->GetPointer(0);
    int nxny = nx*ny;
    int offset = 0;
    for(int k = 0; k < ncz; ++k)
    for(int j = 0; j < ncy; ++j)
    for(int i = 0; i < ncx; ++i)
      {
      *ct++ = SVTK_HEXAHEDRON;

      *cl++ = offset;
      offset += 8;

      nl[0] = (k) * nxny + j*nx + i;
      nl[1] = (k+1) * nxny + j*nx + i;
      nl[2] = (k+1) * nxny + j*nx + i + 1;
      nl[3] = (k) * nxny + j*nx + i + 1;
      nl[4] = (k) * nxny + (j+1)*nx + i;
      nl[5] = (k+1) * nxny + (j+1)*nx + i;
      nl[6] = (k+1) * nxny + (j+1)*nx + i + 1;
      nl[7] = (k) * nxny + (j+1)*nx + i + 1;

      nl += 8;
      }

    // new svtk layout, always 1 extra value
    *cl = offset;

    svtkCellArray *cells = svtkCellArray::New();
    cells->SetData(cellLocations, nlist);

    ug->SetCells(cellTypes, cells);

    nlist->Delete();
    cellTypes->Delete();
    cellLocations->Delete();
    cells->Delete();
  }

  return ug;
}

static
svtkPolyData *newParticleBlock(const std::vector<Particle> *particles,
  bool structureOnly)
{
  svtkPolyData *block = svtkPolyData::New();

  if (structureOnly)
    return block;

  svtkNew<svtkPoints> points;
  svtkNew<svtkCellArray> cells;
  points->Allocate(particles->size());
  cells->Allocate(particles->size());

  svtkIdType pointId = 0;
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
  const std::string &arrayName, svtkFloatArray *&fa)
{
  enum {PID, VEL, VELMAG};

  fa = svtkFloatArray::New();

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
svtkUnsignedCharArray *newGhostCellsArray(int *shape,
  sdiy::DiscreteBounds &cellExt, int ng)
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

    svtkUnsignedCharArray *g = svtkUnsignedCharArray::New();
    g->SetNumberOfTuples(ncells);
    memset(g->GetVoidPointer(0), 0, sizeof(unsigned char) * ncells);
    g->SetName("svtkGhostType");
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


  using BlockExtentMap = std::map<long, sdiy::DiscreteBounds>;
  using BlockDataMap = std::map<long, float*>;

  long NumBlocks;                                    // total number of blocks on all ranks
  sdiy::DiscreteBounds DomainExtent;                 // global index space
  BlockExtentMap BlockExtents;                       // local block extents, indexed by global block id
  BlockDataMap BlockData;                            // local data array, indexed by block id
  std::map<long, const std::vector<Particle>*> ParticleData;
  OscillatorArray Oscillators;                       // global list of oscillators

  double Origin[3];                                  // lower left corner of simulation domain
  double Spacing[3];                                 // mesh spacing

  int Shape[3];
  int NumGhostCells;                                 // number of ghost cells
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
void DataAdaptor::SetOscillators(const OscillatorArray &oscillators)
{
  this->Internals->Oscillators = oscillators;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetMesh(const std::string &meshName, bool structureOnly,
    svtkDataObject *&mesh)
{
  mesh = nullptr;

  if ((meshName != "mesh") && (meshName != "ucdmesh") &&
    (meshName != "particles") && (meshName != "oscillators"))
    {
    SENSEI_ERROR("the miniapp provides meshes named \"mesh\", \"ucdmesh\","
      ", \"particles\", and \"oscillators\". you requested \"" << meshName << "\"")
    return -1;
    }

  svtkMultiBlockDataSet *mb = svtkMultiBlockDataSet::New();
  mesh = mb;

  if (meshName == "oscillators")
    {
    // the oscillators only send on rank 0
    mb->SetNumberOfBlocks(1);

    int rank = 0;
    MPI_Comm_rank(this->GetCommunicator(), &rank);
    if (rank == 0)
      {
      size_t numPts = this->Internals->Oscillators.Size();

      svtkPolyData *pd = svtkPolyData::New();

      svtkPoints *pts = svtkPoints::New();
      pts->SetDataTypeToFloat();
      pts->SetNumberOfPoints(numPts);
      for (size_t cc=0; cc < numPts; ++cc)
      {
        const Oscillator &o = this->Internals->Oscillators[cc];
        pts->SetPoint(cc, o.center_x, o.center_y, o.center_z);
      }
      pd->SetPoints(pts);
      pts->Delete();

      mb->SetBlock(0, pd);
      pd->Delete();
      }
    }
  else
    {
    // the other meshes that have data per block
    int particleBlocks = meshName == "particles";
    int unstructuredBlocks = particleBlocks ? 0 : meshName == "ucdmesh";

    mb->SetNumberOfBlocks(this->Internals->NumBlocks);

    auto it = this->Internals->BlockExtents.begin();
    auto end = this->Internals->BlockExtents.end();
    for (; it != end; ++it)
      {
      if (particleBlocks)
        {
        svtkPolyData *pd =
          newParticleBlock(this->Internals->ParticleData[it->first],
          structureOnly);

        mb->SetBlock(it->first, pd);
        pd->Delete();
        }
      else if (unstructuredBlocks)
        {
        svtkUnstructuredGrid *ug = newUnstructuredBlock(this->Internals->Origin,
          this->Internals->Spacing, it->second, structureOnly);

        mb->SetBlock(it->first, ug);
        ug->Delete();
        }
      else
        {
        svtkImageData *id = newCartesianBlock(this->Internals->Origin,
          this->Internals->Spacing, it->second, structureOnly);

        mb->SetBlock(it->first, id);
        id->Delete();
        }
      }
    }

  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::AddArray(svtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName)
{
  svtkMultiBlockDataSet *mb = dynamic_cast<svtkMultiBlockDataSet*>(mesh);
  if (!mb)
    {
    SENSEI_ERROR("unexpected mesh type "
      << (mesh ? mesh->GetClassName() : "nullptr"))
    return -1;
    }

  if (meshName == "oscillators")
    {
    int rank = 0;
    MPI_Comm_rank(this->GetCommunicator(), &rank);
    if (rank == 0)
      {
      svtkPolyData *pd = dynamic_cast<svtkPolyData*>(mb->GetBlock(0));
      size_t numPts = this->Internals->Oscillators.Size();

      if (arrayName == "radius")
        {
        svtkFloatArray *radius = svtkFloatArray::New();
        radius->SetNumberOfTuples(numPts);
        radius->SetName("radius");
        pd->GetPointData()->AddArray(radius);
        radius->Delete();
        for (size_t i = 0; i < numPts; ++i)
          {
          const Oscillator &o = this->Internals->Oscillators[i];
          radius->SetTypedComponent(i, 0, o.radius);
          }
        }

      if (arrayName == "omega0")
        {
        svtkFloatArray *omega0 = svtkFloatArray::New();
        omega0->SetNumberOfTuples(numPts);
        omega0->SetName("omega0");
        pd->GetPointData()->AddArray(omega0);
        omega0->Delete();
        for (size_t i = 0; i < numPts; ++i)
          {
          const Oscillator &o = this->Internals->Oscillators[i];
          omega0->SetTypedComponent(i, 0, o.omega0);
          }
        }

      if (arrayName == "zeta")
        {
        svtkFloatArray *zeta = svtkFloatArray::New();
        zeta->SetNumberOfTuples(numPts);
        zeta->SetName("zeta");
        pd->GetPointData()->AddArray(zeta);
        zeta->Delete();
        for (size_t i = 0; i < numPts; ++i)
          {
          const Oscillator &o = this->Internals->Oscillators[i];
          zeta->SetTypedComponent(i, 0, o.zeta);
          }
        }

      if (arrayName == "type")
        {
        svtkIntArray *type = svtkIntArray::New();
        type->SetNumberOfTuples(numPts);
        type->SetName("type");
        pd->GetPointData()->AddArray(type);
        type->Delete();
        for (size_t i = 0; i < numPts; ++i)
          {
          const Oscillator &o = this->Internals->Oscillators[i];
          type->SetTypedComponent(i, 0, static_cast<int>(o.type));
          }
        }
      }
    }
  else
    {
    enum {BLOCK, PARTICLE};
    int meshId = BLOCK;
    if ((meshName == "mesh") || (meshName == "ucdmesh"))
      {
      meshId = BLOCK;
      if ((arrayName != "data") || (association != svtkDataObject::CELL))
        {
        SENSEI_ERROR("mesh \"" << meshName
          << "\" only has cell data array named \"data\"")
        return -1;
        }
      }
    else if (meshName == "particles")
      {
      meshId = PARTICLE;
      if (association != svtkDataObject::POINT)
        {
        SENSEI_ERROR("mesh \"particles\" only has point data")
        return -1;
        }
      if ((arrayName != "velocity") && (arrayName != "velocityMagnitude") &&
        (arrayName != "pid"))
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
      svtkDataObject *blk = mb->GetBlock(it->first);
      if (!blk)
        {
        SENSEI_ERROR("encountered empty block at index " << it->first)
        return -1;
        }

      svtkFloatArray *fa = nullptr;
      svtkDataSetAttributes *dsa = nullptr;

      if (meshId == BLOCK)
        {
        dsa = blk->GetAttributes(svtkDataObject::CELL);
        svtkIdType nCells = getBlockNumCells(this->Internals->BlockExtents[it->first]);

        // zero coopy the array
        fa = svtkFloatArray::New();
        fa->SetName("data");
        fa->SetArray(it->second, nCells, 1);
        }
      else
        {
        dsa = blk->GetAttributes(svtkDataObject::POINT);
        newParticleArray(*this->Internals->ParticleData[it->first], arrayName, fa);
        }

      dsa->AddArray(fa);
      fa->Delete();
      }
    }

  return 0;
}

//----------------------------------------------------------------------------
int DataAdaptor::AddGhostCellsArray(svtkDataObject *mesh, const std::string &meshName)
{
  if ((meshName != "mesh") && (meshName != "ucdmesh") &&
    (meshName != "particles") && (meshName != "oscillators"))
    {
    SENSEI_ERROR("the miniapp provides meshes named \"mesh\", \"ucdmesh\","
      ", \"particles\", and \"oscillators\". you requested \"" << meshName << "\"")
    return -1;
    }

  svtkMultiBlockDataSet *mb = dynamic_cast<svtkMultiBlockDataSet*>(mesh);
  if (!mb)
    {
    SENSEI_ERROR("unexpected mesh type "
      << (mesh ? mesh->GetClassName() : "nullptr"))
    return -1;
    }

  if (meshName == "oscillators")
    {
    svtkUnsignedCharArray *gh = svtkUnsignedCharArray::New();
    gh->SetNumberOfTuples(Internals->Oscillators.Size());
    gh->Fill(0);
    gh->SetName("svtkGhostType");

    svtkDataObject *blk = mb->GetBlock(0);
    svtkDataSetAttributes *dsa = blk->GetAttributes(svtkDataObject::CELL);
    dsa->AddArray(gh);
    gh->Delete();
    }
  else
    {
    auto it = this->Internals->BlockExtents.begin();
    auto end = this->Internals->BlockExtents.end();
    for (; it != end; ++it)
      {
      // this code is the same for the Cartesian and unstructured blocks
      // because they both have the same number of cells and are in the
      // same order
      svtkDataObject *blk = mb->GetBlock(it->first);
      if (!blk)
        {
        SENSEI_ERROR("encountered empty block at index " << it->first)
        return -1;
        }

      svtkDataSetAttributes *dsa = blk->GetAttributes(svtkDataObject::CELL);

      svtkUnsignedCharArray *ga = newGhostCellsArray(this->Internals->Shape,
        it->second, this->Internals->NumGhostCells);

      dsa->AddArray(ga);
      ga->Delete();
      }
    }

  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = 4;
  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &metadata)
{
  if (id > 3)
    {
    SENSEI_ERROR("invalid mesh id " << id)
    return -1;
    }

  int rank = 0;
  int nRanks = 1;

  MPI_Comm_rank(this->GetCommunicator(), &rank);
  MPI_Comm_size(this->GetCommunicator(), &nRanks);
  if (id == 2)
    {
    metadata->GlobalView = 1;
    metadata->MeshName = "oscillators";
    metadata->MeshType = SVTK_MULTIBLOCK_DATA_SET;
    metadata->BlockType = SVTK_POLY_DATA;
    metadata->CoordinateType = SVTK_FLOAT;
    metadata->NumBlocks = 1;
    metadata->NumBlocksLocal = {(rank ? 0 : 1)};
    metadata->NumGhostCells = 0;
    metadata->NumArrays = 4;
    metadata->ArrayName = {"radius", "omega0", "zeta", "type"};
    metadata->ArrayCentering = {svtkDataObject::POINT, svtkDataObject::POINT, svtkDataObject::POINT, svtkDataObject::POINT};
    metadata->ArrayComponents = {1, 1, 1, 1};
    metadata->ArrayType = {SVTK_FLOAT, SVTK_FLOAT, SVTK_FLOAT, SVTK_INT};
    metadata->StaticMesh = 1;

    if (metadata->Flags.BlockBoundsSet())
      {
      std::array<double,6> bds = {
        std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()};

      for (unsigned long i = 0; i < Internals->Oscillators.Size(); ++i)
        {
        const Oscillator &o = this->Internals->Oscillators[i];
        bds[0] = std::min(bds[0], (double)o.center_x);
        bds[1] = std::max(bds[1], (double)o.center_x);
        bds[2] = std::min(bds[2], (double)o.center_y);
        bds[3] = std::max(bds[3], (double)o.center_y);
        bds[4] = std::min(bds[4], (double)o.center_z);
        bds[5] = std::max(bds[5], (double)o.center_z);
        }

      metadata->Bounds = bds;
      metadata->BlockBounds.push_back(bds);
      }

    if (metadata->Flags.BlockSizeSet())
      {
      unsigned long nOsc = Internals->Oscillators.Size();
      metadata->BlockNumCells.push_back(nOsc);
      metadata->BlockNumPoints.push_back(nOsc);
      metadata->BlockCellArraySize.push_back(nOsc);
      }

    if (metadata->Flags.BlockDecompSet())
      {
      metadata->BlockOwner.push_back(0);
      metadata->BlockIds.push_back(0);
      }

    if (metadata->Flags.BlockArrayRangeSet())
      {
      std::vector<std::array<double,2>> bar(4,
        {std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()});

      for (unsigned long i = 0; i < Internals->Oscillators.Size(); ++i)
        {
        const Oscillator &o = this->Internals->Oscillators[i];
        bar[0][0] = std::min(bar[0][0], (double)o.radius);
        bar[0][1] = std::max(bar[0][1], (double)o.radius);
        bar[1][0] = std::min(bar[1][0], (double)o.omega0);
        bar[1][1] = std::max(bar[1][1], (double)o.omega0);
        bar[2][0] = std::min(bar[2][0], (double)o.zeta);
        bar[2][1] = std::max(bar[2][1], (double)o.zeta);
        bar[3][0] = std::min(bar[3][0], (double)o.type);
        bar[3][1] = std::max(bar[3][1], (double)o.type);
        }

      metadata->ArrayRange = bar;
      metadata->BlockArrayRange.push_back(bar);
      }
    }
  else if (id == 3)
    {
    int nBlocks = this->Internals->ParticleData.size();

    metadata->MeshName = "particles";
    metadata->MeshType = SVTK_MULTIBLOCK_DATA_SET;
    metadata->BlockType = SVTK_POLY_DATA;
    metadata->CoordinateType = SVTK_FLOAT;
    metadata->NumBlocks = this->Internals->NumBlocks;;
    metadata->NumBlocksLocal = {nBlocks};
    metadata->NumGhostCells = 0;
    metadata->NumArrays = 3;
    metadata->ArrayName = {"pid", "velocity", "velocityMagnitude"};
    metadata->ArrayCentering = {svtkDataObject::POINT, svtkDataObject::POINT, svtkDataObject::POINT};
    metadata->ArrayComponents = {1, 3, 1};
    metadata->ArrayType = {SVTK_INT, SVTK_FLOAT, SVTK_FLOAT};
    metadata->StaticMesh = 1;

    using ParticleMapIterator = std::map<long, const std::vector<Particle>*>::iterator;
    ParticleMapIterator end = this->Internals->ParticleData.end();

    if (metadata->Flags.BlockBoundsSet())
      {
      for (ParticleMapIterator it = this->Internals->ParticleData.begin(); it != end; ++it)
        {
          std::array<double,6> bds = {
            std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
            std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
            std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()};
        auto particles = it->second;
        for (auto particle: *particles)
          {
            bds[0] = std::min(bds[0], (double)particle.position[0]);
            bds[1] = std::max(bds[1], (double)particle.position[0]);
            bds[2] = std::min(bds[2], (double)particle.position[1]);
            bds[3] = std::max(bds[3], (double)particle.position[1]);
            bds[4] = std::min(bds[4], (double)particle.position[2]);
            bds[5] = std::max(bds[5], (double)particle.position[2]);
          }
        metadata->Bounds = bds;
        metadata->BlockBounds.push_back(bds);
        }
      }

    if (metadata->Flags.BlockArrayRangeSet())
      {
      for (ParticleMapIterator it = this->Internals->ParticleData.begin(); it != end; ++it)
        {
          std::vector<std::array<double,2>> bar(3,
            {std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()});
        auto particles = it->second;
        for (auto particle: *particles)
          {
          bar[0][0] = std::min(bar[0][0], (double)particle.velocity[0]);
          bar[0][1] = std::max(bar[0][1], (double)particle.velocity[0]);
          bar[1][0] = std::min(bar[1][0], (double)particle.velocity[1]);
          bar[1][1] = std::max(bar[1][1], (double)particle.velocity[1]);
          bar[2][0] = std::min(bar[2][0], (double)particle.velocity[2]);
          bar[2][1] = std::max(bar[2][1], (double)particle.velocity[2]);
          }
        metadata->ArrayRange = bar;
        metadata->BlockArrayRange.push_back(bar);
        }
      }

    if (metadata->Flags.BlockSizeSet())
      {
      for (ParticleMapIterator it = this->Internals->ParticleData.begin(); it != end; ++it)
        {
        auto particles = it->second;
        long nCells = particles->size();
        long nPts = nCells;

        metadata->BlockNumCells.push_back(nCells);
        metadata->BlockCellArraySize.push_back(nCells);
        metadata->BlockNumPoints.push_back(nPts);
        }
      }

    if (metadata->Flags.BlockDecompSet())
      {
      for (ParticleMapIterator it = this->Internals->ParticleData.begin(); it != end; ++it)
        {
        metadata->BlockOwner.push_back(rank);
        metadata->BlockIds.push_back(it->first);
        }
      }
    }
  else
    {
    // this exercises the multimesh api
    // mesh 0 is a multiblock with uniform Cartesian blocks
    // mesh 1 is a multiblock with unstructured blocks
    // otherwise the meshes are identical
    int nBlocks = this->Internals->BlockData.size();

    metadata->MeshName = (id == 0 ? "mesh" : "ucdmesh");

    metadata->MeshType = SVTK_MULTIBLOCK_DATA_SET;
    metadata->BlockType = (id == 0 ? SVTK_IMAGE_DATA : SVTK_UNSTRUCTURED_GRID);
    metadata->CoordinateType = SVTK_DOUBLE;
    metadata->NumBlocks = this->Internals->NumBlocks;
    metadata->NumBlocksLocal = {nBlocks};
    metadata->NumGhostCells = this->Internals->NumGhostCells;
    metadata->NumArrays = 1;
    metadata->ArrayName = {"data"};
    metadata->ArrayCentering = {svtkDataObject::CELL};
    metadata->ArrayComponents = {1};
    metadata->ArrayType = {SVTK_FLOAT};
    metadata->StaticMesh = 1;

    using ExtentIterator = InternalsType::BlockExtentMap::iterator;

    if ((id == 0) && metadata->Flags.BlockExtentsSet())
      {
      std::array<int,6> ext;
      getBlockExtent(this->Internals->DomainExtent, ext.data());
      metadata->Extent = std::move(ext);

      metadata->BlockExtents.reserve(nBlocks);

      ExtentIterator it = this->Internals->BlockExtents.begin();
      ExtentIterator end = this->Internals->BlockExtents.end();
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

      ExtentIterator it = this->Internals->BlockExtents.begin();
      ExtentIterator end = this->Internals->BlockExtents.end();
      for (; it != end; ++it)
        {
        getBlockBounds(it->second, this->Internals->Origin,
          this->Internals->Spacing, bounds.data());
        metadata->BlockBounds.emplace_back(std::move(bounds));
        }
      }

    if (metadata->Flags.BlockSizeSet())
      {
      ExtentIterator it = this->Internals->BlockExtents.begin();
      ExtentIterator end = this->Internals->BlockExtents.end();
      for (; it != end; ++it)
        {
        long nCells = getBlockNumCells(it->second);
        long nPts = getBlockNumPoints(it->second);

        metadata->BlockNumCells.push_back(nCells);
        metadata->BlockNumPoints.push_back(nPts);

        if (id == 1) // unctructured only
          metadata->BlockCellArraySize.push_back(8*nCells);
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
    }

  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::ReleaseData()
{
  return 0;
}

}
