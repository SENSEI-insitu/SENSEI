#include "nyx_sensei_dataadaptor.h"

#if defined(BL_USE_FLOAT) || !defined(NYX_SENSEI_NO_COPY)
#include <vtkFloatArray.h>
#else
#include <vtkDoubleArray.h>
#endif
#ifdef NYX_SENSEI_NO_COPY
#include <vtkUnsignedCharArray.h>
#endif
#include <vtkImageData.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkCellData.h>
#include <vtkSmartPointer.h>
#include <vtkVector.h>
#include <vtkVectorOperators.h>

#include <cassert>

#include <diy/master.hpp>

namespace nyx_sensei_bridge
{

class DataAdaptor::DInternals
{
  public:
  std::vector<diy::DiscreteBounds> CellExtents;
#ifdef NYX_SENSEI_NO_COPY
  std::vector<diy::DiscreteBounds> ValidCellExtents;
  std::vector<const amrex::Real*> Data;
#else
  std::vector<float*> Data;
#endif
  vtkSmartPointer<vtkMultiBlockDataSet> Mesh;
  std::vector<vtkSmartPointer<vtkImageData> > BlockMesh;
  std::vector<int> DataExtent;
  std::vector<double> BoundingBox;
  vtkVector3d Origin;
  vtkVector3d Spacing;
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
#ifdef NYX_SENSEI_NO_COPY
  internals.ValidCellExtents.resize(nblocks);
#endif
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

#ifdef NYX_SENSEI_NO_COPY

//-----------------------------------------------------------------------------
void DataAdaptor::SetValidBlockExtent(int gid,
  int xmin, int xmax,
  int ymin, int ymax,
  int zmin, int zmax)
{
  DInternals& internals = (*this->Internals);
  internals.ValidCellExtents[gid].min[0] = xmin;
  internals.ValidCellExtents[gid].min[1] = ymin;
  internals.ValidCellExtents[gid].min[2] = zmin;

  internals.ValidCellExtents[gid].max[0] = xmax;
  internals.ValidCellExtents[gid].max[1] = ymax;
  internals.ValidCellExtents[gid].max[2] = zmax;
}

#endif

//-----------------------------------------------------------------------------
void DataAdaptor::SetDataExtent(int ext[6])
{
  // TODO -- this key holds a int**, it should copy the data
  this->Internals->DataExtent.assign(ext, ext+6);
  this->GetInformation()->Set(vtkDataObject::DATA_EXTENT(),
      this->Internals->DataExtent.data(), 6);
}
//
//-----------------------------------------------------------------------------
void DataAdaptor::SetPhysicalExtents(double pext[6])
{
  // Not needed.But still add meta data
  this->Internals->BoundingBox.assign(pext, pext+6);
  // note vtkDoubleArray::BOUNDING_BOX() is double vector. It keeps
  // a copy (unlike vtkDataObject::DATA_EXTENT() which keep the pointer to the ext).
  this->GetInformation()->Set(vtkDataObject::BOUNDING_BOX(),
      this->Internals->BoundingBox.data(), 6);
}

//-----------------------------------------------------------------------------

// Computes Origin and Spacing using DataExtent and BoundingBox (assuming both
// are specified for the full domain (not just what's present on local
// partition). Assumes both DataExtent and BoundingBox is set.
void DataAdaptor::ComputeSpacingAndOrigin()
{
  assert(this->Internals->BoundingBox.size() == 6 && this->Internals->DataExtent.size() == 6);
  vtkVector3d minpt = vtkVector3d(
          this->Internals->BoundingBox[0],
          this->Internals->BoundingBox[2],
          this->Internals->BoundingBox[4]);
  vtkVector3d maxpt = vtkVector3d(
          this->Internals->BoundingBox[1],
          this->Internals->BoundingBox[3],
          this->Internals->BoundingBox[5]);
  this->Internals->Origin = minpt;

  vtkVector3d pt_ext = vtkVector3d(
          this->Internals->DataExtent[1] - this->Internals->DataExtent[0],
          this->Internals->DataExtent[3] - this->Internals->DataExtent[2],
          this->Internals->DataExtent[5] - this->Internals->DataExtent[4]);
  // convert cell extents to point extents.
  pt_ext = pt_ext + vtkVector3d(1.0, 1.0, 1.0);
  this->Internals->Spacing = (maxpt - minpt) / pt_ext;
}

//-----------------------------------------------------------------------------
#ifdef NYX_SENSEI_NO_COPY
void DataAdaptor::SetBlockData(int gid, const amrex::Real* data)
#else
void DataAdaptor::SetBlockData(int gid, float* data)
#endif
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
  this->AddArray(internals.Mesh,
      vtkDataObject::FIELD_ASSOCIATION_CELLS, "data");
  return internals.Mesh;
}

//-----------------------------------------------------------------------------
vtkDataObject* DataAdaptor::GetBlockMesh(int gid)
{
  DInternals& internals = (*this->Internals);
  vtkSmartPointer<vtkImageData>& blockMesh = internals.BlockMesh[gid];
  const diy::DiscreteBounds& cellExts = internals.CellExtents[gid];
#ifdef NYX_SENSEI_NO_COPY
  const diy::DiscreteBounds& validCellExts = internals.ValidCellExtents[gid];
#endif
  if (!blockMesh && areBoundsValid(cellExts))
    {
    blockMesh = vtkSmartPointer<vtkImageData>::New();
    blockMesh->SetOrigin(this->Internals->Origin.GetData());
    blockMesh->SetSpacing(this->Internals->Spacing.GetData());
    blockMesh->SetExtent(
      cellExts.min[0], cellExts.max[0]+1,
      cellExts.min[1], cellExts.max[1]+1,
      cellExts.min[2], cellExts.max[2]+1);
#ifdef NYX_SENSEI_NO_COPY
    vtkSmartPointer<vtkUnsignedCharArray> ghostArray = vtkSmartPointer<vtkUnsignedCharArray>::New();
    ghostArray->SetNumberOfTuples(blockMesh->GetNumberOfCells());
    ghostArray->SetName(vtkDataSetAttributes::GhostArrayName());

    // Easier (though more memory access): First blank everything out and then just
    // "unlbank" out valid region
    for (vtkIdType tuple = 0; tuple < ghostArray->GetNumberOfTuples(); ++tuple)
      ghostArray->SetTuple1(tuple, vtkDataSetAttributes::DUPLICATECELL);

    int ijk[3];
    for (ijk[0] = validCellExts.min[0]; ijk[0] <= validCellExts.max[0]; ++ijk[0])
      for (ijk[1] = validCellExts.min[1]; ijk[1] <= validCellExts.max[1]; ++ijk[1])
        for (ijk[2] = validCellExts.min[2]; ijk[2] <= validCellExts.max[2]; ++ijk[2])
          {
          ghostArray->SetTuple1(blockMesh->ComputeCellId(ijk), 0);
          }

    blockMesh->GetCellData()->AddArray(ghostArray);
#endif
    }
  return blockMesh;
}

//-----------------------------------------------------------------------------
bool DataAdaptor::AddArray(vtkDataObject* mesh, int association,
                           const std::string& arrayname)
{
#ifndef NDEBUG
  if (association != vtkDataObject::FIELD_ASSOCIATION_CELLS ||
      arrayname != "data")
    {
    return false;
    }
#endif
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
      if (cd->GetArray(arrayname.c_str()) == NULL)
        {

#if defined(BL_USE_FLOAT) || !defined(NYX_SENSEI_NO_COPY)
        vtkFloatArray* fa = vtkFloatArray::New();
#else
        vtkDoubleArray* fa = vtkDoubleArray::New();
#endif
        fa->SetName(arrayname.c_str());
#ifdef NYX_SENSEI_NO_COPY
        fa->SetArray(const_cast<amrex::Real*>(internals.Data[cc]), blockMesh->GetNumberOfCells(), 1);
        fa->Modified();
#else
        fa->SetArray(internals.Data[cc], blockMesh->GetNumberOfCells(), 0);
#endif
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
