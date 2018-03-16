#include "dataadaptor.h"
#include "Error.h"

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkPoints.h>
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
void DataAdaptor::Initialize(size_t nblocks)
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
int DataAdaptor::GetMesh(const std::string &meshName, bool structureOnly,
    vtkDataObject *&mesh)
{
  if (meshName != "mesh" && meshName != "ucdmesh")
    {
    SENSEI_ERROR("the miniapp provides meshes named \"mesh\" and \"ucdmesh\"" 
       " you requested \"" << meshName << "\"")
    return -1;
    }

  DInternals& internals = (*this->Internals);

  if(meshName == "ucdmesh")
    {
    if (!internals.uMesh)
      {
      internals.uMesh = vtkSmartPointer<vtkMultiBlockDataSet>::New();
      internals.uMesh->SetNumberOfBlocks(static_cast<unsigned int>(internals.CellExtents.size()));
      }
    // Either create empty vtkUnstructuredGrid objects or let us replace
    // empty ones with new ones that have the right data.
    for (size_t cc=0; cc < internals.CellExtents.size(); ++cc)
      {
      unsigned int bid = static_cast<unsigned int>(cc);
      vtkUnstructuredGrid *g = vtkUnstructuredGrid::SafeDownCast(
          internals.uMesh->GetBlock(bid));
      if((structureOnly && g == NULL) || 
         (!structureOnly && g != NULL && g->GetNumberOfCells() == 0))
        {
        internals.uMesh->SetBlock(bid, this->GetUnstructuredMesh(cc, structureOnly));
        }
      }
    mesh = internals.uMesh;
    }
  else
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

//-----------------------------------------------------------------------------
int DataAdaptor::AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName)
{
#ifndef NDEBUG
  if ((association != vtkDataObject::FIELD_ASSOCIATION_CELLS) ||
    (arrayName != "data") || (meshName != "mesh" && meshName != "ucdmesh"))
    {
    SENSEI_ERROR("the miniapp provides a cell centered array named \"data\" "
      " on a mesh named \"mesh\"")
    return 1;
    }
#else
  (void)meshName;
  (void)association;
  (void)arrayName;
#endif
  int retVal = 1;
  DInternals& internals = (*this->Internals);
  vtkMultiBlockDataSet* md = vtkMultiBlockDataSet::SafeDownCast(mesh);
  for (unsigned int cc=0, max=md->GetNumberOfBlocks(); cc < max; ++cc)
    {
    if (!internals.Data[cc])
      {
      continue;
      }
    vtkCellData *cd = NULL;
    vtkIdType ncells = 0;
    if(meshName == "mesh")
       {
       vtkSmartPointer<vtkImageData>& blockMesh = internals.BlockMesh[cc];
       cd = (blockMesh? blockMesh->GetCellData() : NULL);
       ncells = blockMesh->GetNumberOfCells();
       }
    else if(meshName == "ucdmesh")
       {
       vtkSmartPointer<vtkUnstructuredGrid>& uMesh = internals.UnstructuredMesh[cc];
       cd = (uMesh? uMesh->GetCellData() : NULL);
       ncells = uMesh->GetNumberOfCells();
       }
    if (cd != NULL)
      {
      if (cd->GetArray(arrayName.c_str()) == NULL)
        {
        vtkFloatArray* fa = vtkFloatArray::New();
        fa->SetName(arrayName.c_str());
        fa->SetArray(internals.Data[cc], ncells, 1);
        cd->SetScalars(fa);
        cd->SetActiveScalars("data");
        fa->FastDelete();
        }
      retVal = 0;
      }
    }
  return retVal;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = 2;
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

  SENSEI_ERROR("Failed to get mesh name")
  return -1;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetNumberOfArrays(const std::string &meshName, int association,
    unsigned int &numberOfArrays)
{
  if ((meshName == "mesh" || meshName == "ucdmesh") && (association == vtkDataObject::CELL))
    {
    numberOfArrays = 1;
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

  SENSEI_ERROR("Failed to get array name")
  return -1;
}

//-----------------------------------------------------------------------------
int DataAdaptor::ReleaseData()
{
  DInternals& internals = (*this->Internals);
  internals.Mesh = NULL;
  internals.uMesh = NULL;
  for (auto i : internals.CellExtents)
    {
    i.min[0] = i.min[1] = i.min[2] = 0;
    i.max[0] = i.max[1] = i.max[2] = -1;
    }
  for (size_t cc=0, max = internals.Data.size(); cc < max; ++cc)
    {
    internals.Data[cc] = NULL;
    internals.BlockMesh[cc] = NULL;
    internals.UnstructuredMesh[cc] = NULL;
    }
  return 0;
}

}
