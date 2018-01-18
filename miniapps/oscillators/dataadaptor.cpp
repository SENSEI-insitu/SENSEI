#include "dataadaptor.h"
#include "Error.h"

#include <vtkInformation.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkCellData.h>
#include <vtkSmartPointer.h>

#include <diy/master.hpp>

namespace oscillators
{

struct DataAdaptor::DInternals
{
  std::vector<diy::DiscreteBounds> CellExtents;
  std::vector<float*> Data;
  vtkSmartPointer<vtkMultiBlockDataSet> Mesh;
  std::vector<vtkSmartPointer<vtkImageData> > BlockMesh;
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
  (void)structureOnly;

  if (meshName != "mesh")
    {
    SENSEI_ERROR("the miniapp provides a mesh named \"mesh\"" 
       " you requested \"" << meshName << "\"")
    return -1;
    }

  DInternals& internals = (*this->Internals);

  if (!internals.Mesh)
    {
    internals.Mesh = vtkSmartPointer<vtkMultiBlockDataSet>::New();
    internals.Mesh->SetNumberOfBlocks(static_cast<unsigned int>(internals.CellExtents.size()));
    for (size_t cc=0; cc < internals.CellExtents.size(); ++cc)
      {
      internals.Mesh->SetBlock(static_cast<unsigned int>(cc), this->GetBlockMesh(cc));
      }
    }

  mesh = internals.Mesh;

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
int DataAdaptor::AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName)
{
#ifndef NDEBUG
  if ((association != vtkDataObject::FIELD_ASSOCIATION_CELLS) ||
    (arrayName != "data") || (meshName != "mesh"))
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
    vtkSmartPointer<vtkImageData>& blockMesh = internals.BlockMesh[cc];
    if (vtkCellData* cd = (blockMesh? blockMesh->GetCellData(): NULL))
      {
      if (cd->GetArray(arrayName.c_str()) == NULL)
        {
        vtkFloatArray* fa = vtkFloatArray::New();
        fa->SetName(arrayName.c_str());
        fa->SetArray(internals.Data[cc], blockMesh->GetNumberOfCells(), 1);
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
  numMeshes = 1;
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

  SENSEI_ERROR("Failed to get mesh name")
  return -1;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetNumberOfArrays(const std::string &meshName, int association,
    unsigned int &numberOfArrays)
{
  if ((meshName == "mesh") && (association == vtkDataObject::CELL))
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
  if ((meshName == "mesh") &&
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
  for (auto i : internals.CellExtents)
    {
    i.min[0] = i.min[1] = i.min[2] = 0;
    i.max[0] = i.max[1] = i.max[2] = -1;
    }
  for (size_t cc=0, max = internals.Data.size(); cc < max; ++cc)
    {
    internals.Data[cc] = NULL;
    internals.BlockMesh[cc] = NULL;
    }
  return 0;
}

}
