#include "dataadaptor.h"

#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkCellData.h>
#include <vtkSmartPointer.h>

#include <diy/master.hpp>

namespace oscillators
{

class DataAdaptor::DInternals
{
public:
  std::vector<diy::DiscreteBounds> CellExtents;
  std::vector<float*> Data;
  vtkSmartPointer<vtkMultiBlockDataSet> Mesh;
  std::vector<vtkSmartPointer<vtkImageData> > BlockMesh;
};

inline bool areBoundsValid(const diy::DiscreteBounds& bds)
{
  return (bds.min[0] <= bds.max[0] && bds.min[1] <= bds.max[1] && bds.min[2] <= bds.max[2]);
}

vtkStandardNewMacro(DataAdaptor);
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
void DataAdaptor::SetBlockData(int gid, float* data)
{
  DInternals& internals = (*this->Internals);
  internals.Data[gid] = data;
}

//-----------------------------------------------------------------------------
vtkDataObject* DataAdaptor::GetMesh(bool vtkNotUsed(structure_only))
{
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
  return internals.Mesh;
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
bool DataAdaptor::AddArray(vtkDataObject* mesh, int association, const char* arrayname)
{
  if (association != vtkDataObject::FIELD_ASSOCIATION_CELLS ||
      arrayname == NULL ||
      strcmp(arrayname, "data") != 0)
    {
    return false;
    }

  bool retVal = false;
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
      if (cd->GetArray(arrayname) == NULL)
        {
        vtkFloatArray* fa = vtkFloatArray::New();
        fa->SetName(arrayname);
        fa->SetArray(internals.Data[cc], blockMesh->GetNumberOfCells(), 1);
        cd->SetScalars(fa);
        cd->SetActiveScalars("data");
        fa->FastDelete();
        }
      retVal = true;
      }
    }
  return retVal;
}

//-----------------------------------------------------------------------------
void DataAdaptor::ReleaseData()
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
}

}
