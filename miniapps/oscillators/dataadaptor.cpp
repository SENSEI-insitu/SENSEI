#include "dataadaptor.h"
#include "Error.h"

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>
#include <vtkIntArray.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>

#include <diy/master.hpp>

namespace oscillators
{

struct DataAdaptor::DInternals
{
  std::vector<diy::DiscreteBounds> CellExtents;
  std::vector<float*> Data;
  vtkSmartPointer<vtkMultiBlockDataSet> Mesh;
  std::vector<vtkSmartPointer<vtkImageData> > BlockMesh;
  vtkSmartPointer<vtkMultiBlockDataSet> uMesh;
  std::vector<vtkSmartPointer<vtkUnstructuredGrid> > UnstructuredMesh;
  std::vector<int> DataExtent;
  int shape[3];
  int ghostLevels;

  std::vector<const Particles*> ParticlesData;
  vtkSmartPointer<vtkMultiBlockDataSet> ParticlesDataSet;
  std::vector<vtkSmartPointer<vtkPolyData>> ParticleBlocks;
};

inline bool areBoundsValid(const diy::DiscreteBounds& bds)
{
  return ((bds.min[0] <= bds.max[0]) && (bds.min[1] <= bds.max[1])
    && (bds.min[2] <= bds.max[2]));
}

//-----------------------------------------------------------------------------
senseiNewMacro(DataAdaptor);

//-----------------------------------------------------------------------------
DataAdaptor::DataAdaptor() :
  Internals(new DataAdaptor::DInternals())
{
}

//-----------------------------------------------------------------------------
DataAdaptor::~DataAdaptor()
{
  delete this->Internals;
}

//-----------------------------------------------------------------------------
void DataAdaptor::Initialize(size_t nblocks, const int *shape_, int ghostLevels_)
{
  DInternals& internals = (*this->Internals);
  internals.CellExtents.resize(nblocks);
  internals.Data.resize(nblocks);
  internals.BlockMesh.resize(nblocks);
  internals.UnstructuredMesh.resize(nblocks);
  for (size_t cc=0; cc < nblocks; cc++)
    {
    internals.CellExtents[cc].min[0] = 0;
    internals.CellExtents[cc].min[1] = 0;
    internals.CellExtents[cc].min[2] = 0;
    internals.CellExtents[cc].max[0] = -1;
    internals.CellExtents[cc].max[1] = -1;
    internals.CellExtents[cc].max[2] = -1;
    }

  internals.shape[0] = shape_[0];
  internals.shape[1] = shape_[1];
  internals.shape[2] = shape_[2];
  internals.ghostLevels = ghostLevels_;

  internals.ParticlesData.resize(nblocks);
  internals.ParticleBlocks.resize(nblocks);

  this->ReleaseData();
}

//-----------------------------------------------------------------------------
void DataAdaptor::SetBlockExtent(int gid,
  int xmin, int xmax,
  int ymin, int ymax,
  int zmin, int zmax)
{
  DInternals& internals = (*this->Internals);
  internals.CellExtents[gid].min[0] = xmin;
  internals.CellExtents[gid].min[1] = ymin;
  internals.CellExtents[gid].min[2] = zmin;

  internals.CellExtents[gid].max[0] = xmax;
  internals.CellExtents[gid].max[1] = ymax;
  internals.CellExtents[gid].max[2] = zmax;
}

//-----------------------------------------------------------------------------
void DataAdaptor::SetDataExtent(int ext[6])
{
  this->Internals->DataExtent.assign(ext, ext+6);

  this->GetInformation()->Set(vtkDataObject::DATA_EXTENT(),
      this->Internals->DataExtent.data(), 6);
}

//-----------------------------------------------------------------------------
void DataAdaptor::SetBlockData(int gid, float* data)
{
  DInternals& internals = (*this->Internals);
  internals.Data[gid] = data;
}

//-----------------------------------------------------------------------------
void DataAdaptor::SetParticles(int gid, const Particles& particles)
{
  this->Internals->ParticlesData[gid] = &particles;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetMesh(const std::string &meshName, bool structureOnly,
    vtkDataObject *&mesh)
{
  if (meshName != "mesh" && meshName != "ucdmesh" && meshName != "particles")
    {
    SENSEI_ERROR("the miniapp provides meshes named \"mesh\", \"ucdmesh\", and \"particles\""
       " you requested \"" << meshName << "\"")
    return -1;
    }

  DInternals& internals = (*this->Internals);

  if (meshName == "ucdmesh")
    {
    if (!internals.uMesh)
      {
      internals.uMesh = vtkSmartPointer<vtkMultiBlockDataSet>::New();
      internals.uMesh->SetNumberOfBlocks(static_cast<unsigned int>(internals.CellExtents.size()));
      for (size_t cc=0; cc < internals.CellExtents.size(); ++cc)
        internals.uMesh->SetBlock(static_cast<unsigned int>(cc), nullptr);
      }
    // Either create empty vtkUnstructuredGrid objects or let us replace
    // empty ones with new ones that have the right data.
    for (size_t cc=0; cc < internals.CellExtents.size(); ++cc)
      {
      unsigned int bid = static_cast<unsigned int>(cc);
      vtkUnstructuredGrid *g = vtkUnstructuredGrid::SafeDownCast(
          internals.uMesh->GetBlock(bid));
      if(g == nullptr)
        {
        g = (vtkUnstructuredGrid *)this->GetUnstructuredMesh(cc, structureOnly);
        //cout << "Setting uMesh[" << bid << "] structureOnly=" << structureOnly << ", g=" << (void*)g << ", ncells=" << (g ? g->GetNumberOfCells() : 0) << endl;
        internals.uMesh->SetBlock(bid, g);
        }
      else if(!structureOnly && g->GetNumberOfCells() == 0)
        {
        g = (vtkUnstructuredGrid *)this->GetUnstructuredMesh(cc, structureOnly);
        //cout << "Replacing uMesh[" << bid << "] structureOnly=" << structureOnly << ", g=" << (void*)g << ", ncells=" << (g ? g->GetNumberOfCells() : 0) << endl;
        internals.uMesh->SetBlock(bid, g);
        }
      }

    mesh = internals.uMesh;
    }
  else if (meshName == "mesh")
    {
    if (!internals.Mesh)
      {
      internals.Mesh = vtkSmartPointer<vtkMultiBlockDataSet>::New();
      internals.Mesh->SetNumberOfBlocks(static_cast<unsigned int>(internals.CellExtents.size()));
      for (size_t cc=0; cc < internals.CellExtents.size(); ++cc)
        internals.Mesh->SetBlock(static_cast<unsigned int>(cc), this->GetBlockMesh(cc));
      }
      mesh = internals.Mesh;
    }
  else if (meshName == "particles")
    {
    if (!internals.ParticlesDataSet)
      {
      internals.ParticlesDataSet = vtkSmartPointer<vtkMultiBlockDataSet>::New();
      internals.ParticlesDataSet->SetNumberOfBlocks(static_cast<unsigned int>(internals.ParticlesData.size()));
      for (size_t cc=0; cc < internals.ParticlesData.size(); ++cc)
        {
        internals.ParticlesDataSet->SetBlock(
          static_cast<unsigned int>(cc),  this->GetParticlesBlock(cc, structureOnly));
        }
      }
      mesh = internals.ParticlesDataSet;
    }

  return 0;
}

//-----------------------------------------------------------------------------
vtkDataObject* DataAdaptor::GetBlockMesh(int gid)
{
  DInternals& internals = (*this->Internals);
  vtkSmartPointer<vtkImageData>& blockMesh = internals.BlockMesh[gid];
  const diy::DiscreteBounds& cellExts = internals.CellExtents[gid];
  if (!blockMesh && areBoundsValid(cellExts))
    {
    blockMesh = vtkSmartPointer<vtkImageData>::New();
    blockMesh->SetExtent(
      cellExts.min[0], cellExts.max[0]+1,
      cellExts.min[1], cellExts.max[1]+1,
      cellExts.min[2], cellExts.max[2]+1);
    }
  return blockMesh;
}

//-----------------------------------------------------------------------------
vtkDataObject* DataAdaptor::GetUnstructuredMesh(int gid, bool structureOnly)
{
  DInternals& internals = (*this->Internals);
  vtkSmartPointer<vtkUnstructuredGrid>& uMesh = internals.UnstructuredMesh[gid];
  const diy::DiscreteBounds& cellExts = internals.CellExtents[gid];
  if(areBoundsValid(cellExts))
  {
      if (uMesh == nullptr)
      {
          uMesh = vtkSmartPointer<vtkUnstructuredGrid>::New();
      }

      if(structureOnly == false &&
         uMesh != nullptr &&
         uMesh->GetNumberOfCells() == 0)
      {
          // Add points.
          int nx = cellExts.max[0] - cellExts.min[0] + 1+1;
          int ny = cellExts.max[1] - cellExts.min[1] + 1+1;
          int nz = cellExts.max[2] - cellExts.min[2] + 1+1;
          vtkPoints *pts = vtkPoints::New();
          pts->SetNumberOfPoints(nx*ny*nz);
          vtkIdType idx = 0;
          for(int k = cellExts.min[2]; k <= cellExts.max[2]+1; ++k)
          for(int j = cellExts.min[1]; j <= cellExts.max[1]+1; ++j)
          for(int i = cellExts.min[0]; i <= cellExts.max[0]+1; ++i)
          {
              pts->SetPoint(idx++, i,j,k);
          }
          uMesh->SetPoints(pts);
          pts->Delete();

          // Add cells
          vtkIdType ncells = (nx-1)*(ny-1)*(nz-1);
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
          for(int k = 0; k < nz-1; ++k)
          for(int j = 0; j < ny-1; ++j)
          for(int i = 0; i < nx-1; ++i)
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
          uMesh->SetCells(cellTypes, cellLocations, cells);
          cellTypes->Delete();
          cellLocations->Delete();
          cells->Delete();
      }
  }
  return uMesh;
}

vtkDataObject* DataAdaptor::GetParticlesBlock(int gid, bool structureOnly)
{
  DInternals& internals = (*this->Internals);
  auto particles = internals.ParticlesData[gid];
  if (!particles)
    {
    return NULL;
    }

  vtkSmartPointer<vtkPolyData>& block = internals.ParticleBlocks[gid];
  if (!block)
    {
    block = vtkSmartPointer<vtkPolyData>::New();
    }
  if (!structureOnly && block->GetNumberOfCells() == 0 && !particles->empty())
    {
    block->Initialize();

    vtkNew<vtkPoints> points;
    vtkNew<vtkCellArray> cells;
    points->Allocate(particles->size());
    cells->Allocate(particles->size());

    vtkIdType pointId = 0;
    for (const auto& p : *particles)
      {
      points->InsertNextPoint(p.position[0], p.position[1], p.position[2]);
      cells->InsertNextCell(1, &pointId);
      ++pointId;
      }
    block->SetPoints(points.Get());
    block->SetVerts(cells.Get());
    }

  return block;
}

//-----------------------------------------------------------------------------
int DataAdaptor::AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName)
{
  DInternals& internals = (*this->Internals);
  vtkMultiBlockDataSet* md = vtkMultiBlockDataSet::SafeDownCast(mesh);

  if ((meshName == "mesh" || meshName == "ucdmesh") && arrayName == "data" &&
      association == vtkDataObject::FIELD_ASSOCIATION_CELLS)
    {
    for (unsigned int cc=0, max=md->GetNumberOfBlocks(); cc < max; ++cc)
      {
      if (!internals.Data[cc]) // Exclude NULL datasets
        {
        continue;
        }
      vtkCellData *cd = nullptr;
      if(meshName == "mesh")
         {
         vtkSmartPointer<vtkImageData>& blockMesh = internals.BlockMesh[cc];
         cd = (blockMesh? blockMesh->GetCellData() : nullptr);
         }
      else if(meshName == "ucdmesh")
         {
         vtkSmartPointer<vtkUnstructuredGrid>& uMesh = internals.UnstructuredMesh[cc];
         cd = (uMesh? uMesh->GetCellData() : nullptr);
         }
      const diy::DiscreteBounds &ce = internals.CellExtents[cc];

      if (cd && !cd->GetArray(arrayName.c_str()))
        {
        vtkIdType ncells = (ce.max[0] - ce.min[0] + 1)*
          (ce.max[1] - ce.min[1] + 1)*(ce.max[2] - ce.min[2] + 1);

        vtkFloatArray *fa = vtkFloatArray::New();
        fa->SetName(arrayName.c_str());
        fa->SetArray(internals.Data[cc], ncells, 1);
        cd->SetScalars(fa);
        cd->SetActiveScalars("data");
        fa->Delete();
	}
      }
    }
  else if (meshName == "particles" && association == vtkDataObject::FIELD_ASSOCIATION_POINTS &&
           (arrayName == "uniqueGlobalId" || arrayName == "velocity" || arrayName == "velocityMagnitude"))
    {
    for (unsigned int cc=0, max=md->GetNumberOfBlocks(); cc < max; ++cc)
      {
      if (!internals.ParticlesData[cc])
        {
        continue;
        }

      const Particles& particles = *internals.ParticlesData[cc];

      vtkPolyData* block = internals.ParticleBlocks[cc].Get();
      auto pd = block ? block->GetPointData() : NULL;
      if (pd && pd->GetArray(arrayName.c_str()) == NULL)
        {
        if (arrayName == "uniqueGlobalId")
          {
          vtkNew<vtkIdTypeArray> array;
          array->SetName(arrayName.c_str());
          array->SetNumberOfComponents(1);
          array->SetNumberOfTuples(particles.size());
          for (vtkIdType i = 0; i < array->GetNumberOfTuples(); ++i)
            {
            vtkIdType ugid = particles[i].id;
            array->SetValue(i, ugid);
            }
          pd->AddArray(array.Get());
          }
        else if (arrayName == "velocity")
          {
          vtkNew<vtkFloatArray> array;
          array->SetName(arrayName.c_str());
          array->SetNumberOfComponents(3);
          array->SetNumberOfTuples(particles.size());
          float vel[3] = { 0, 0, 0 };
          for (vtkIdType i = 0; i < array->GetNumberOfTuples(); ++i)
            {
            vel[0] = particles[i].velocity[0];
            vel[1] = particles[i].velocity[1];
            vel[2] = particles[i].velocity[2];
            array->SetTuple(i, vel);
            }
          pd->SetScalars(array.Get());
          pd->SetActiveScalars(array->GetName());
          }
        else if (arrayName == "velocityMagnitude")
          {
          vtkNew<vtkFloatArray> array;
          array->SetName(arrayName.c_str());
          array->SetNumberOfComponents(1);
          array->SetNumberOfTuples(particles.size());
          for (vtkIdType i = 0; i < array->GetNumberOfTuples(); ++i)
            {
            float magnitude = std::sqrt(particles[i].velocity.norm());
            array->SetValue(i, magnitude);
            }
          pd->SetScalars(array.Get());
          pd->SetActiveScalars(array->GetName());
          }
        }
      }
    }
#ifndef NDEBUG
    else
    {
    SENSEI_ERROR("the miniapp provides a cell centered array named \"data\" "
      "on meshes named \"mesh\" or \"ucdmesh\"; or point centered arrays named "
      "\"uniqueGlobalId\", \"velocity\" and \"velocityMagnitude\" on a mesh named \"particles\"")
    return -1;
    }
#endif

  return 0;
}

//----------------------------------------------------------------------------
int DataAdaptor::GetMeshHasGhostCells(const std::string &/*meshName*/,
  int &nLayers)
{
  DInternals& internals = (*this->Internals);
  nLayers = internals.ghostLevels;
  return 0;
}

//----------------------------------------------------------------------------
vtkDataArray *
DataAdaptor::CreateGhostCellsArray(int cc) const
{
    // This sim is always 3D.
    const DInternals& internals = (*this->Internals);
    int imin = internals.CellExtents[cc].min[0];
    int jmin = internals.CellExtents[cc].min[1];
    int kmin = internals.CellExtents[cc].min[2];
    int imax = internals.CellExtents[cc].max[0];
    int jmax = internals.CellExtents[cc].max[1];
    int kmax = internals.CellExtents[cc].max[2];
    int nx = imax - imin + 1;
    int ny = jmax - jmin + 1;
    int nz = kmax - kmin + 1;
    int nxny = nx*ny;
    int ncells = nx * ny *nz;
    int ng = internals.ghostLevels;

#define GCTYPE unsigned char
#define GCVTKARRAY vtkUnsignedCharArray
    GCVTKARRAY *g = GCVTKARRAY::New();
    g->SetNumberOfTuples(ncells);
    memset(g->GetVoidPointer(0), 0, sizeof(GCTYPE) * ncells);
    g->SetName("vtkGhostType");
    GCTYPE *gptr = (GCTYPE *)g->GetVoidPointer(0);
    GCTYPE ghost = 1;

    if(imin > 0)
    {
        // Set the low I faces to ghosts.
        for(int k = 0; k < nz; ++k)
        for(int j = 0; j < ny; ++j)
        for(int i = 0; i < ng; ++i)
            gptr[k * nxny + j*nx + i] = ghost;
    }
    if(imax < internals.shape[0]-1)
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
    if(jmax < internals.shape[1]-1)
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
    if(kmax < internals.shape[2]-1)
    {
        // Set the high K faces to ghosts.
        for(int k = nz-ng; k < nz; ++k)
        for(int j = 0; j < ny; ++j)
        for(int i = 0; i < nx; ++i)
            gptr[k * nxny + j*nx + i] = ghost;
    }

    return g;
}

//----------------------------------------------------------------------------
int DataAdaptor::AddGhostCellsArray(vtkDataObject *mesh, const std::string &meshName)
{
  if (meshName != "mesh" && meshName != "ucdmesh")
    {
    SENSEI_ERROR("the miniapp provides meshes \"mesh\" and \"ucdmesh\".")
    return -1;
    }

  DInternals& internals = (*this->Internals);
  vtkMultiBlockDataSet* md = vtkMultiBlockDataSet::SafeDownCast(mesh);
  for (unsigned int cc=0, max=md->GetNumberOfBlocks(); cc < max; ++cc)
    {
    vtkCellData *cd = nullptr;
    if(meshName == "mesh")
       {
       vtkSmartPointer<vtkImageData>& blockMesh = internals.BlockMesh[cc];
       cd = (blockMesh? blockMesh->GetCellData() : nullptr);
       }
    else if(meshName == "ucdmesh")
       {
       vtkSmartPointer<vtkUnstructuredGrid>& uMesh = internals.UnstructuredMesh[cc];
       cd = (uMesh? uMesh->GetCellData() : nullptr);
       }

    if (cd && !cd->GetArray("vtkGhostType"))
      {
      vtkDataArray *g = CreateGhostCellsArray(cc);
      cd->AddArray(g);
      g->Delete();
      }
    }

  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = 3;
  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetMeshName(unsigned int id, std::string &meshName)
{
  if (id == 0)
    {
    meshName = "mesh";
    return 0;
    }
  else if (id == 1)
    {
    meshName = "ucdmesh";
    return 0;
    }
  else if (id == 2)
    {
    meshName = "particles";
    return 0;
    }

  SENSEI_ERROR("Failed to get mesh name")
  return -1;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetNumberOfArrays(const std::string &meshName, int association,
    unsigned int &numberOfArrays)
{
  numberOfArrays = 0;
  if ((meshName == "mesh" || meshName == "ucdmesh") && (association == vtkDataObject::CELL))
    {
    numberOfArrays = 1;
    return 0;
    }
  else if ((meshName == "particles") && (association == vtkDataObject::POINT))
    {
    numberOfArrays = 2;
    return 0;
    }
  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetArrayName(const std::string &meshName, int association,
    unsigned int index, std::string &arrayName)
{
  if ((meshName == "mesh" || meshName == "ucdmesh") &&
    (association == vtkDataObject::CELL) && (index == 0))
    {
    arrayName = "data";
    return 0;
    }
  else if ((meshName == "particles") && (association == vtkDataObject::POINT))
    {
    static const char *names[] = { "uniqueGlobalId", "velocity" };
    arrayName = names[index];
    return 0;
    }

  SENSEI_ERROR("Failed to get array name")
  return -1;
}

//-----------------------------------------------------------------------------
int DataAdaptor::ReleaseData()
{
  DInternals& internals = (*this->Internals);
  internals.Mesh = nullptr;
  internals.uMesh = nullptr;
  internals.ParticlesDataSet = nullptr;
  for (auto i : internals.CellExtents)
    {
    i.min[0] = i.min[1] = i.min[2] = 0;
    i.max[0] = i.max[1] = i.max[2] = -1;
    }
  for (size_t cc=0, max = internals.Data.size(); cc < max; ++cc)
    {
    internals.Data[cc] = nullptr;
    internals.BlockMesh[cc] = nullptr;
    internals.UnstructuredMesh[cc] = nullptr;
    internals.ParticlesData[cc] = nullptr;
    internals.ParticleBlocks[cc] = nullptr;
    }
  return 0;
}

}
