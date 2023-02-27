#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overread"
/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkHyperTreeGrid.cxx

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkHyperTreeGrid.h"

#include "svtkBitArray.h"
#include "svtkBoundingBox.h"
#include "svtkCollection.h"
#include "svtkDoubleArray.h"
#include "svtkFieldData.h"
#include "svtkGenericCell.h"
#include "svtkHyperTree.h"
#include "svtkHyperTreeGridNonOrientedCursor.h"
#include "svtkHyperTreeGridNonOrientedGeometryCursor.h"
#include "svtkHyperTreeGridNonOrientedMooreSuperCursor.h"
#include "svtkHyperTreeGridNonOrientedMooreSuperCursorLight.h"
#include "svtkHyperTreeGridNonOrientedVonNeumannSuperCursor.h"
#include "svtkHyperTreeGridNonOrientedVonNeumannSuperCursorLight.h"
#include "svtkHyperTreeGridOrientedCursor.h"
#include "svtkHyperTreeGridOrientedGeometryCursor.h"
#include "svtkHyperTreeGridScales.h"
#include "svtkIdList.h"
#include "svtkIdTypeArray.h"
#include "svtkInformation.h"
#include "svtkInformationDoubleVectorKey.h"
#include "svtkInformationIntegerKey.h"
#include "svtkInformationVector.h"
#include "svtkMath.h"
#include "svtkNew.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkSmartPointer.h"
#include "svtkStructuredData.h"
#include "svtkUnsignedCharArray.h"

#include <array>
#include <cassert>
#include <deque>

svtkInformationKeyMacro(svtkHyperTreeGrid, LEVELS, Integer);
svtkInformationKeyMacro(svtkHyperTreeGrid, DIMENSION, Integer);
svtkInformationKeyMacro(svtkHyperTreeGrid, ORIENTATION, Integer);
svtkInformationKeyRestrictedMacro(svtkHyperTreeGrid, SIZES, DoubleVector, 3);

svtkStandardNewMacro(svtkHyperTreeGrid);
svtkCxxSetObjectMacro(svtkHyperTreeGrid, XCoordinates, svtkDataArray);
svtkCxxSetObjectMacro(svtkHyperTreeGrid, YCoordinates, svtkDataArray);
svtkCxxSetObjectMacro(svtkHyperTreeGrid, ZCoordinates, svtkDataArray);

void svtkHyperTreeGrid::CopyCoordinates(const svtkHyperTreeGrid* output)
{
  this->SetXCoordinates(output->XCoordinates);
  this->SetYCoordinates(output->YCoordinates);
  this->SetZCoordinates(output->ZCoordinates);
}

void svtkHyperTreeGrid::SetFixedCoordinates(unsigned int axis, double value)
{
  svtkNew<svtkDoubleArray> zeros;
  zeros->SetNumberOfValues(1);
  zeros->SetValue(0, value);
  switch (axis)
  {
    case 0:
    {
      this->SetXCoordinates(zeros);
      break;
    }
    case 1:
    {
      this->SetYCoordinates(zeros);
      break;
    }
    case 2:
    {
      this->SetZCoordinates(zeros);
      break;
    }
    default:
    {
      assert("pre: invalid_axis" && axis < 3);
    }
  }
}

void svtkHyperTreeGrid::SetMask(svtkBitArray* _arg)
{
  svtkSetObjectBodyMacro(Mask, svtkBitArray, _arg);

  this->InitPureMask = false;
  if (this->PureMask)
  {
    this->PureMask->Delete();
    this->PureMask = nullptr;
  }
}

// Helper macros to quickly fetch a HT at a given index or iterator
#define GetHyperTreeFromOtherMacro(_obj_, _index_)                                                 \
  (static_cast<svtkHyperTree*>(_obj_->HyperTrees.find(_index_) != _obj_->HyperTrees.end()           \
      ? _obj_->HyperTrees[_index_]                                                                 \
      : nullptr))
#define GetHyperTreeFromThisMacro(_index_) GetHyperTreeFromOtherMacro(this, _index_)

//-----------------------------------------------------------------------------
svtkHyperTreeGrid::svtkHyperTreeGrid()
{
  // Default state
  this->ModeSqueeze = nullptr;
  this->FreezeState = false;

  // Grid topology
  this->TransposedRootIndexing = false;

  // Invalid default grid parameters to force actual initialization
  this->Orientation = UINT_MAX;
  this->BranchFactor = 0;
  this->NumberOfChildren = 0;

  // Depth limiter
  this->DepthLimiter = UINT_MAX;

  // Masked primal leaves
  this->Mask = nullptr;
  this->PureMask = nullptr;
  this->InitPureMask = false;

  // No interface by default
  this->HasInterface = false;

  // Interface array names
  this->InterfaceNormalsName = nullptr;
  this->InterfaceInterceptsName = nullptr;

  // Primal grid geometry
  this->WithCoordinates = true;
  this->XCoordinates = svtkDoubleArray::New();
  this->XCoordinates->SetNumberOfTuples(1);
  this->XCoordinates->SetTuple1(0, 0.0);

  this->YCoordinates = svtkDoubleArray::New();
  this->YCoordinates->SetNumberOfTuples(1);
  this->YCoordinates->SetTuple1(0, 0.0);

  this->ZCoordinates = svtkDoubleArray::New();
  this->ZCoordinates->SetNumberOfTuples(1);
  this->ZCoordinates->SetTuple1(0, 0.0);

  this->TreeGhostArrayCached = false;

  // -----------------------------------------------
  // RectilinearGrid
  // -----------------------------------------------
  // Invalid default grid parameters to force actual initialization
  this->Dimension = 0;
  this->Dimensions[0] = 0; // Just used by GetDimensions
  this->Dimensions[1] = 0;
  this->Dimensions[2] = 0;

  this->CellDims[0] = 0; // Just used by GetCellDims
  this->CellDims[1] = 0;
  this->CellDims[2] = 0;

  this->Axis[0] = UINT_MAX;
  this->Axis[1] = UINT_MAX;

  int extent[6] = { 0, -1, 0, -1, 0, -1 };
  memcpy(this->Extent, extent, 6 * sizeof(int));

  this->DataDescription = SVTK_EMPTY;

  this->Information->Set(svtkDataObject::DATA_EXTENT_TYPE(), SVTK_3D_EXTENT);
  this->Information->Set(svtkDataObject::DATA_EXTENT(), this->Extent, 6);

  // Generate default information
  this->Bounds[0] = 0.0;
  this->Bounds[1] = -1.0;
  this->Bounds[2] = 0.0;
  this->Bounds[3] = -1.0;
  this->Bounds[4] = 0.0;
  this->Bounds[5] = -1.0;

  this->Center[0] = 0.0;
  this->Center[1] = 0.0;
  this->Center[2] = 0.0;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::Initialize()
{
  this->Superclass::Initialize();
  // DataObject Initialize will not do PointData
  this->PointData->Initialize();
  // Delete existing trees
  this->HyperTrees.clear();

  // Default state
  this->ModeSqueeze = nullptr;
  this->FreezeState = false;

  // Grid topology
  this->TransposedRootIndexing = false;

  // Invalid default grid parameters to force actual initialization
  this->Orientation = UINT_MAX;
  this->BranchFactor = 0;
  this->NumberOfChildren = 0;

  // Depth limiter
  this->DepthLimiter = UINT_MAX;

  // Masked primal leaves
  svtkBitArray* mask = svtkBitArray::New();
  this->SetMask(mask);
  mask->FastDelete();

  // No interface by default
  this->HasInterface = false;

  // Interface array names
  this->InterfaceNormalsName = nullptr;
  this->InterfaceInterceptsName = nullptr;

  // Primal grid geometry
  this->WithCoordinates = true;

  // Might be better to set coordinates using this->SetXCoordinates(),
  // but there is currently a conflict with svtkUniformHyperTreeGrid
  // which inherits from svtkHyperTreeGrid.
  // To be fixed when a better inheritance tree is implemented.
  if (this->XCoordinates)
  {
    this->XCoordinates->Delete();
  }
  this->XCoordinates = svtkDoubleArray::New();
  this->XCoordinates->SetNumberOfTuples(1);
  this->XCoordinates->SetTuple1(0, 0.0);

  if (this->YCoordinates)
  {
    this->YCoordinates->Delete();
  }
  this->YCoordinates = svtkDoubleArray::New();
  this->YCoordinates->SetNumberOfTuples(1);
  this->YCoordinates->SetTuple1(0, 0.0);

  if (this->ZCoordinates)
  {
    this->ZCoordinates->Delete();
  }
  this->ZCoordinates = svtkDoubleArray::New();
  this->ZCoordinates->SetNumberOfTuples(1);
  this->ZCoordinates->SetTuple1(0, 0.0);

  // -----------------------------------------------
  // RectilinearGrid
  // -----------------------------------------------
  // Invalid default grid parameters to force actual initialization
  this->Dimension = 0;
  this->Dimensions[0] = 0; // Just used by GetDimensions
  this->Dimensions[1] = 0;
  this->Dimensions[2] = 0;

  this->CellDims[0] = 0; // Just used by GetCellDims
  this->CellDims[1] = 0;
  this->CellDims[2] = 0;

  this->Axis[0] = UINT_MAX;
  this->Axis[1] = UINT_MAX;

  int extent[6] = { 0, -1, 0, -1, 0, -1 };
  memcpy(this->Extent, extent, 6 * sizeof(int));

  this->DataDescription = SVTK_EMPTY;

  this->Information->Set(svtkDataObject::DATA_EXTENT_TYPE(), SVTK_3D_EXTENT);
  this->Information->Set(svtkDataObject::DATA_EXTENT(), this->Extent, 6);

  // Generate default information
  this->Bounds[0] = 0.0;
  this->Bounds[1] = -1.0;
  this->Bounds[2] = 0.0;
  this->Bounds[3] = -1.0;
  this->Bounds[4] = 0.0;
  this->Bounds[5] = -1.0;

  this->Center[0] = 0.0;
  this->Center[1] = 0.0;
  this->Center[2] = 0.0;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::Squeeze()
{
  if (!this->FreezeState)
  {
    svtkHyperTreeGridIterator itIn;
    InitializeTreeIterator(itIn);
    svtkIdType indexIn;
    while (svtkHyperTree* ht = itIn.GetNextTree(indexIn))
    {
      svtkHyperTree* htfreeze = ht->Freeze(this->GetModeSqueeze());
      if (htfreeze != ht)
      {
        this->SetTree(indexIn, htfreeze);
        htfreeze->UnRegister(this);
      }
    }
    this->FreezeState = true;
  }
}

//-----------------------------------------------------------------------------
svtkHyperTreeGrid::~svtkHyperTreeGrid()
{
  if (this->ModeSqueeze)
  {
    delete[] this->ModeSqueeze;
    this->ModeSqueeze = nullptr;
  }

  if (this->Mask)
  {
    this->Mask->Delete();
    this->Mask = nullptr;
  }

  if (this->PureMask)
  {
    this->PureMask->Delete();
    this->PureMask = nullptr;
  }

  if (this->XCoordinates)
  {
    this->XCoordinates->Delete();
    this->XCoordinates = nullptr;
  }

  if (this->YCoordinates)
  {
    this->YCoordinates->Delete();
    this->YCoordinates = nullptr;
  }

  if (this->ZCoordinates)
  {
    this->ZCoordinates->Delete();
    this->ZCoordinates = nullptr;
  }
  this->SetInterfaceNormalsName(nullptr);
  this->SetInterfaceInterceptsName(nullptr);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Frozen: " << this->FreezeState << endl;
  os << indent << "Dimension: " << this->Dimension << endl;
  os << indent << "Orientation: " << this->Orientation << endl;
  os << indent << "BranchFactor: " << this->BranchFactor << endl;
  os << indent << "Dimensions: " << this->Dimensions[0] << "," << this->Dimensions[1] << ","
     << this->Dimensions[2] << endl;
  os << indent << "Extent: " << this->Extent[0] << "," << this->Extent[1] << "," << this->Extent[2]
     << "," << this->Extent[3] << "," << this->Extent[4] << "," << this->Extent[5] << endl;
  os << indent << "CellDims: " << this->CellDims[0] << "," << this->CellDims[1] << ","
     << this->CellDims[2] << endl;
  os << indent << "Axis: " << this->Axis[0] << "," << this->Axis[1] << endl;
  os << indent << "Mask:\n";
  if (this->Mask)
  {
    this->Mask->PrintSelf(os, indent.GetNextIndent());
  }
  if (this->PureMask)
  {
    this->PureMask->PrintSelf(os, indent.GetNextIndent());
  }
  os << indent << "InitPureMask: " << (this->InitPureMask ? "true" : "false") << endl;

  os << indent << "HasInterface: " << (this->HasInterface ? "true" : "false") << endl;
  if (this->WithCoordinates)
  {
    os << indent << "XCoordinates:" << endl;
    if (this->XCoordinates)
    {
      this->XCoordinates->PrintSelf(os, indent.GetNextIndent());
    }
    os << indent << "YCoordinates:" << endl;
    if (this->YCoordinates)
    {
      this->YCoordinates->PrintSelf(os, indent.GetNextIndent());
    }
    os << indent << "ZCoordinates:" << endl;
    if (this->ZCoordinates)
    {
      this->ZCoordinates->PrintSelf(os, indent.GetNextIndent());
    }
  }
  else
  {
    os << indent << "Non explicit coordinates" << endl;
  }
  os << indent << "HyperTrees: " << this->HyperTrees.size() << endl;

  os << indent << "PointData:" << endl;
  this->PointData->PrintSelf(os, indent.GetNextIndent());
}

//----------------------------------------------------------------------------
svtkHyperTreeGrid* svtkHyperTreeGrid::GetData(svtkInformation* info)
{
  return info ? svtkHyperTreeGrid::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkHyperTreeGrid* svtkHyperTreeGrid::GetData(svtkInformationVector* v, int i)
{
  return svtkHyperTreeGrid::GetData(v->GetInformationObject(i));
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::CopyEmptyStructure(svtkDataObject* ds)
{
  assert("pre: ds_exists" && ds != nullptr);
  svtkHyperTreeGrid* htg = svtkHyperTreeGrid::SafeDownCast(ds);
  assert("pre: same_type" && htg != nullptr);

  // RectilinearGrid
  memcpy(this->Dimensions, htg->GetDimensions(), 3 * sizeof(unsigned int));
  this->SetExtent(htg->GetExtent());
  memcpy(this->CellDims, htg->GetCellDims(), 3 * sizeof(unsigned int));
  this->DataDescription = htg->DataDescription;

  this->WithCoordinates = htg->WithCoordinates;
  if (this->WithCoordinates)
  {
    this->SetXCoordinates(htg->XCoordinates);
    this->SetYCoordinates(htg->YCoordinates);
    this->SetZCoordinates(htg->ZCoordinates);
  }

  // Copy grid parameters
  this->ModeSqueeze = htg->ModeSqueeze;
  this->FreezeState = htg->FreezeState;
  this->BranchFactor = htg->BranchFactor;
  this->Dimension = htg->Dimension;
  this->Orientation = htg->Orientation;

  memcpy(this->Extent, htg->GetExtent(), 6 * sizeof(int));
  memcpy(this->Axis, htg->GetAxes(), 2 * sizeof(unsigned int));
  this->NumberOfChildren = htg->NumberOfChildren;
  this->DepthLimiter = htg->DepthLimiter;
  this->TransposedRootIndexing = htg->TransposedRootIndexing;
  this->InitPureMask = htg->InitPureMask;
  this->HasInterface = htg->HasInterface;
  this->SetInterfaceNormalsName(htg->InterfaceNormalsName);
  this->SetInterfaceInterceptsName(htg->InterfaceInterceptsName);

  this->PointData->CopyStructure(htg->GetPointData());
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::CopyStructure(svtkDataObject* ds)
{
  assert("pre: ds_exists" && ds != nullptr);
  svtkHyperTreeGrid* htg = svtkHyperTreeGrid::SafeDownCast(ds);
  assert("pre: same_type" && htg != nullptr);

  // RectilinearGrid
  memcpy(this->Dimensions, htg->GetDimensions(), 3 * sizeof(unsigned int));
  this->SetExtent(htg->GetExtent());
  memcpy(this->CellDims, htg->GetCellDims(), 3 * sizeof(unsigned int));
  this->DataDescription = htg->DataDescription;

  this->WithCoordinates = htg->WithCoordinates;
  if (this->WithCoordinates)
  {
    this->SetXCoordinates(htg->XCoordinates);
    this->SetYCoordinates(htg->YCoordinates);
    this->SetZCoordinates(htg->ZCoordinates);
  }

  // Copy grid parameters
  this->ModeSqueeze = htg->ModeSqueeze;
  this->FreezeState = htg->FreezeState;
  this->BranchFactor = htg->BranchFactor;
  this->Dimension = htg->Dimension;
  this->Orientation = htg->Orientation;

  memcpy(this->Extent, htg->GetExtent(), 6 * sizeof(int));
  memcpy(this->Axis, htg->GetAxes(), 2 * sizeof(unsigned int));
  this->NumberOfChildren = htg->NumberOfChildren;
  this->DepthLimiter = htg->DepthLimiter;
  this->TransposedRootIndexing = htg->TransposedRootIndexing;
  this->InitPureMask = htg->InitPureMask;
  this->HasInterface = htg->HasInterface;
  this->SetInterfaceNormalsName(htg->InterfaceNormalsName);
  this->SetInterfaceInterceptsName(htg->InterfaceInterceptsName);

  // Shallow copy masked if needed
  this->SetMask(htg->GetMask());
  svtkSetObjectBodyMacro(PureMask, svtkBitArray, htg->GetPureMask());

  this->PointData->CopyStructure(htg->GetPointData());

  // Search for hyper tree with given index
  this->HyperTrees.clear();

  for (auto it = htg->HyperTrees.begin(); it != htg->HyperTrees.end(); ++it)
  {
    svtkHyperTree* tree = svtkHyperTree::CreateInstance(this->BranchFactor, this->Dimension);
    assert("pre: same_type" && tree != nullptr);
    tree->CopyStructure(it->second);
    this->HyperTrees[it->first] = tree;
    tree->Delete();
  }
}

// ============================================================================
// BEGIN - RectilinearGrid common API
// ============================================================================

void svtkHyperTreeGrid::SetDimensions(const int dim[3])
{
  this->SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1);
}

//----------------------------------------------------------------------------
void svtkHyperTreeGrid::SetDimensions(int i, int j, int k)
{
  this->SetExtent(0, i - 1, 0, j - 1, 0, k - 1);
}

//----------------------------------------------------------------------------
void svtkHyperTreeGrid::SetDimensions(const unsigned int dim[3])
{
  this->SetExtent(0, static_cast<int>(dim[0]) - 1, 0, static_cast<int>(dim[1]) - 1, 0,
    static_cast<int>(dim[2]) - 1);
}

//----------------------------------------------------------------------------
void svtkHyperTreeGrid::SetDimensions(unsigned int i, unsigned int j, unsigned int k)
{
  this->SetExtent(
    0, static_cast<int>(i) - 1, 0, static_cast<int>(j) - 1, 0, static_cast<int>(k) - 1);
}

//----------------------------------------------------------------------------
const unsigned int* svtkHyperTreeGrid::GetDimensions() const
{
  return this->Dimensions;
}

//----------------------------------------------------------------------------
void svtkHyperTreeGrid::GetDimensions(int dim[3]) const
{
  dim[0] = static_cast<int>(this->Dimensions[0]);
  dim[1] = static_cast<int>(this->Dimensions[1]);
  dim[2] = static_cast<int>(this->Dimensions[2]);
}

//----------------------------------------------------------------------------
void svtkHyperTreeGrid::GetDimensions(unsigned int dim[3]) const
{
  dim[0] = this->Dimensions[0];
  dim[1] = this->Dimensions[1];
  dim[2] = this->Dimensions[2];
}

//----------------------------------------------------------------------------
const unsigned int* svtkHyperTreeGrid::GetCellDims() const
{
  return this->CellDims;
}

//----------------------------------------------------------------------------
void svtkHyperTreeGrid::GetCellDims(int cellDims[3]) const
{
  cellDims[0] = static_cast<int>(this->CellDims[0]);
  cellDims[1] = static_cast<int>(this->CellDims[1]);
  cellDims[2] = static_cast<int>(this->CellDims[2]);
}

//----------------------------------------------------------------------------
void svtkHyperTreeGrid::GetCellDims(unsigned int cellDims[3]) const
{
  cellDims[0] = this->CellDims[0];
  cellDims[1] = this->CellDims[1];
  cellDims[2] = this->CellDims[2];
}

//----------------------------------------------------------------------------
void svtkHyperTreeGrid::SetExtent(const int extent[6])
{
  assert("pre: valid_extent_0" && extent[0] == 0);
  assert("pre: valid_extent_1" && extent[1] >= -1); // -1 is the unset extent
  assert("pre: valid_extent_2" && extent[2] == 0);
  assert("pre: valid_extent_3" && extent[3] >= -1); // -1 is the unset extent
  assert("pre: valid_extent_4" && extent[4] == 0);
  assert("pre: valid_extent_5" && extent[5] >= -1); // -1 is the unset extent
  int description = svtkStructuredData::SetExtent(const_cast<int*>(extent), this->Extent);
  // why svtkStructuredData::SetExtent don't take const int* ?

  if (description < 0) // improperly specified
  {
    svtkErrorMacro(<< "Bad extent, retaining previous values");
    return;
  }

  this->Dimension = 0;
  this->Axis[0] = UINT_MAX;
  this->Axis[1] = UINT_MAX;
  for (unsigned int i = 0; i < 3; ++i)
  {
    this->Dimensions[i] = static_cast<unsigned int>(extent[2 * i + 1] - extent[2 * i] + 1);
    if (this->Dimensions[i] == 1)
    {
      this->CellDims[i] = 1;
    }
    else
    {
      this->CellDims[i] = this->Dimensions[i] - 1;
      if (this->Dimension == 2)
      {
        this->Axis[0] = UINT_MAX;
        this->Axis[1] = UINT_MAX;
      }
      else
      {
        this->Axis[this->Dimension] = i;
      }
      ++this->Dimension;
    }
  }

  assert("post: valid_axis" &&
    (this->Dimension != 3 || (this->Axis[0] == UINT_MAX && this->Axis[1] == UINT_MAX)));
  assert("post: valid_axis" &&
    (this->Dimension != 2 || (this->Axis[0] != UINT_MAX && this->Axis[1] != UINT_MAX)));
  assert("post: valid_axis" &&
    (this->Dimension != 1 || (this->Axis[0] != UINT_MAX && this->Axis[1] == UINT_MAX)));

  switch (this->Dimension)
  {
    case 1:
      this->Orientation = this->Axis[0];
      break;
    case 2:
      this->Orientation = 0;
      for (unsigned int i = 0; i < 2; ++i)
      {
        if (this->Orientation == this->Axis[i])
        {
          ++this->Orientation;
        }
      }
      // If normal to the HTG is y, we right now have HTG spanned by (x,y)
      // We swap them to have a direct frame spanning the HTG
      if (this->Orientation == 1)
      {
        std::swap(this->Axis[0], this->Axis[1]);
      }
      break;
  }

  assert("post: valid_axis" &&
    (this->Dimension != 2 ||
      (this->Axis[0] == (this->Orientation + 1) % 3 &&
        this->Axis[1] == (this->Orientation + 2) % 3)));

  // Make sure that number of children is factor^dimension
  this->NumberOfChildren = this->BranchFactor;
  for (unsigned int i = 1; i < this->Dimension; ++i)
  {
    this->NumberOfChildren *= this->BranchFactor;
  }
  if (description == SVTK_UNCHANGED)
  {
    return;
  }
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkHyperTreeGrid::SetExtent(int i0, int i1, int j0, int j1, int k0, int k1)
{
  int extent[6];

  extent[0] = i0;
  extent[1] = i1;
  extent[2] = j0;
  extent[3] = j1;
  extent[4] = k0;
  extent[5] = k1;

  this->SetExtent(extent);
}

// ============================================================================
// END - RectilinearGrid common API
// ============================================================================

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::SetBranchFactor(unsigned int factor)
{
  assert("pre: valid_factor" && factor >= 2 && factor <= 3);

  // Make sure that number of children is factor^dimension
  unsigned int num = factor;
  for (unsigned int i = 1; i < this->Dimension; ++i)
  {
    num *= factor;
  }

  // Bail out early if nothing was changed
  if (this->BranchFactor == factor && this->NumberOfChildren == num)
  {
    return;
  }

  // Otherwise modify as needed
  this->BranchFactor = factor;
  this->NumberOfChildren = num;
  this->Modified();
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGrid::HasMask()
{
  return this->Mask && this->Mask->GetNumberOfTuples() != 0;
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGrid::GetMaxNumberOfTrees()
{
  return this->CellDims[0] * this->CellDims[1] * this->CellDims[2];
}

//-----------------------------------------------------------------------------
unsigned int svtkHyperTreeGrid::GetNumberOfLevels(svtkIdType index)
{
  svtkHyperTree* tree = GetHyperTreeFromThisMacro(index);
  return tree ? tree->GetNumberOfLevels() : 0;
}

//-----------------------------------------------------------------------------
unsigned int svtkHyperTreeGrid::GetNumberOfLevels()
{
  svtkIdType nLevels = 0;

  // Iterate over all individual trees
  svtkHyperTreeGrid::svtkHyperTreeGridIterator it;
  this->InitializeTreeIterator(it);
  svtkHyperTree* tree = nullptr;
  while ((tree = it.GetNextTree()) != nullptr)
  {
    const svtkIdType nl = tree->GetNumberOfLevels();
    if (nl > nLevels)
    {
      nLevels = nl;
    }
  } // while (it.GetNextTree(inIndex))

  return nLevels;
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGrid::GetNumberOfVertices()
{
  svtkIdType nVertices = 0;

  // Iterate over all trees in grid
  svtkHyperTreeGridIterator it;
  it.Initialize(this);
  while (svtkHyperTree* tree = it.GetNextTree())
  {
    nVertices += tree->GetNumberOfVertices();
  }
  return nVertices;
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGrid::GetNumberOfLeaves()
{
  svtkIdType nLeaves = 0;

  // Iterate over all trees in grid
  svtkHyperTreeGridIterator it;
  it.Initialize(this);
  while (svtkHyperTree* tree = it.GetNextTree())
  {
    nLeaves += tree->GetNumberOfLeaves();
  }

  return nLeaves;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::InitializeTreeIterator(svtkHyperTreeGridIterator& it)
{
  it.Initialize(this);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::InitializeOrientedCursor(
  svtkHyperTreeGridOrientedCursor* cursor, svtkIdType index, bool create)
{
  cursor->Initialize(this, index, create);
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridOrientedCursor* svtkHyperTreeGrid::NewOrientedCursor(svtkIdType index, bool create)
{
  svtkHyperTreeGridOrientedCursor* cursor = svtkHyperTreeGridOrientedCursor::New();
  cursor->Initialize(this, index, create);
  return cursor;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::InitializeOrientedGeometryCursor(
  svtkHyperTreeGridOrientedGeometryCursor* cursor, svtkIdType index, bool create)
{
  cursor->Initialize(this, index, create);
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridOrientedGeometryCursor* svtkHyperTreeGrid::NewOrientedGeometryCursor(
  svtkIdType index, bool create)
{
  svtkHyperTreeGridOrientedGeometryCursor* cursor = svtkHyperTreeGridOrientedGeometryCursor::New();
  cursor->Initialize(this, index, create);
  return cursor;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::InitializeNonOrientedCursor(
  svtkHyperTreeGridNonOrientedCursor* cursor, svtkIdType index, bool create)
{
  cursor->Initialize(this, index, create);
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedCursor* svtkHyperTreeGrid::NewNonOrientedCursor(
  svtkIdType index, bool create)
{
  svtkHyperTreeGridNonOrientedCursor* cursor = svtkHyperTreeGridNonOrientedCursor::New();
  cursor->Initialize(this, index, create);
  return cursor;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::InitializeNonOrientedGeometryCursor(
  svtkHyperTreeGridNonOrientedGeometryCursor* cursor, svtkIdType index, bool create)
{
  cursor->Initialize(this, index, create);
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedGeometryCursor* svtkHyperTreeGrid::NewNonOrientedGeometryCursor(
  svtkIdType index, bool create)
{
  svtkHyperTreeGridNonOrientedGeometryCursor* cursor =
    svtkHyperTreeGridNonOrientedGeometryCursor::New();
  cursor->Initialize(this, index, create);
  return cursor;
}

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// #define TRACE

unsigned int svtkHyperTreeGrid::RecurseDichotomic(
  double value, svtkDoubleArray* coord, unsigned int ideb, unsigned int ifin) const
{
#ifdef TRACE
  std::cerr << "RecurseDichotomic: [" << ideb << "; " << ifin << "]" << std::endl;
#endif
  if (ideb == ifin - 1)
  {
    return ideb;
  }
  unsigned imil = ideb + (ifin - ideb) / 2;
#ifdef TRACE
  std::cerr << "RecurseDichotomic: [" << ideb << "; " << imil << "; " << ifin << "]" << std::endl;
  std::cerr << "RecurseDichotomic: imil# " << imil << " : " << coord->GetValue(imil) << std::endl;
#endif
  if (value < coord->GetValue(imil))
  {
    return this->RecurseDichotomic(value, coord, ideb, imil);
  }
  else
  {
    return this->RecurseDichotomic(value, coord, imil, ifin);
  }
}

unsigned int svtkHyperTreeGrid::FindDichotomic(double value, svtkDataArray* tmp) const
{
  svtkDoubleArray* coord = svtkDoubleArray::SafeDownCast(tmp);
  if (value < coord->GetValue(0) || value > coord->GetValue(coord->GetNumberOfTuples() - 1))
  {
    return UINT_MAX;
  }
  return RecurseDichotomic(value, coord, 0, coord->GetNumberOfTuples());
}

unsigned int svtkHyperTreeGrid::FindDichotomicX(double value) const
{
  assert("pre: exist_coordinates_explict" && this->WithCoordinates);
  return this->FindDichotomic(value, this->XCoordinates);
}

unsigned int svtkHyperTreeGrid::FindDichotomicY(double value) const
{
  assert("pre: exist_coordinates_explict" && this->WithCoordinates);
  return this->FindDichotomic(value, this->YCoordinates);
}

unsigned int svtkHyperTreeGrid::FindDichotomicZ(double value) const
{
  assert("pre: exist_coordinates_explict" && this->WithCoordinates);
  return this->FindDichotomic(value, this->ZCoordinates);
}

svtkHyperTreeGridNonOrientedGeometryCursor* svtkHyperTreeGrid::FindNonOrientedGeometryCursor(
  double x[3])
{
#ifdef TRACE
  std::cerr << "FindNonOrientedGeometryCursor: " << x[0] << "; " << x[1] << "; " << x[2]
            << std::endl;
#endif
  unsigned int i = this->FindDichotomicX(x[0]);
  if (i == UINT_MAX)
  {
    return nullptr;
  }
#ifdef TRACE
  std::cerr << "Position i# " << i << std::endl;
#endif
  unsigned int j = this->FindDichotomicY(x[1]);
  if (j == UINT_MAX)
  {
    return nullptr;
  }
#ifdef TRACE
  std::cerr << "Position j# " << j << std::endl;
#endif
  unsigned int k = this->FindDichotomicZ(x[2]);
  if (k == UINT_MAX)
  {
    return nullptr;
  }
#ifdef TRACE
  std::cerr << "Position k# " << k << std::endl;
#endif

  svtkIdType index;
  this->GetIndexFromLevelZeroCoordinates(index, i, j, k);
#ifdef TRACE
  std::cerr << "Tree index# " << index << std::endl;
#endif

  svtkHyperTreeGridNonOrientedGeometryCursor* cursor =
    svtkHyperTreeGridNonOrientedGeometryCursor::New();
  cursor->Initialize(this, index, false);

  switch (this->BranchFactor)
  {
    case 2:
    {
      while (!cursor->IsLeaf())
      {
        double p[3];
        cursor->GetPoint(p);
        unsigned int ichild = 0;
        if (x[0] <= p[0])
        {
        }
        else
        {
          ichild = 1;
        }
        if (x[1] <= p[1])
        {
        }
        else
        {
          ichild = 2 + ichild;
        }
        if (x[2] <= p[2])
        {
        }
        else
        {
          ichild = 4 + ichild;
        }
        cursor->ToChild(ichild);
      }
      break;
    }
    case 3:
    {
      /*
           while(not cursor->IsLeaf())
           {
             const double* origin = cursor->GetOrigin();
             const double* scale = cursor->GetSize();
             double *limit = new double(this->BranchFactor);


             unsigned int ichild = 0;
             if (x[ 0 ] <= p[ 0 ]) {
             } else {
               ichild = 1;
             }
             if (x[ 1 ] <= p[ 1 ]) {
             } else {
               ichild = 3 + ichild;
             }
             if (x[ 2 ] <= p[ 2 ]) {
             } else {
               ichild = 9 + ichild;
             }
             cursor->ToChild(ichild);
           }
     */
      assert("pre: not_implemented_raf_3" && false);
      break;
    }
  }

  return cursor;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::InitializeNonOrientedVonNeumannSuperCursor(
  svtkHyperTreeGridNonOrientedVonNeumannSuperCursor* cursor, svtkIdType index, bool create)
{
  cursor->Initialize(this, index, create);
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedVonNeumannSuperCursor*
svtkHyperTreeGrid::NewNonOrientedVonNeumannSuperCursor(svtkIdType index, bool create)
{
  svtkHyperTreeGridNonOrientedVonNeumannSuperCursor* cursor =
    svtkHyperTreeGridNonOrientedVonNeumannSuperCursor::New();
  cursor->Initialize(this, index, create);
  return cursor;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::InitializeNonOrientedVonNeumannSuperCursorLight(
  svtkHyperTreeGridNonOrientedVonNeumannSuperCursorLight* cursor, svtkIdType index, bool create)
{
  cursor->Initialize(this, index, create);
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedVonNeumannSuperCursorLight*
svtkHyperTreeGrid::NewNonOrientedVonNeumannSuperCursorLight(svtkIdType index, bool create)
{
  svtkHyperTreeGridNonOrientedVonNeumannSuperCursorLight* cursor =
    svtkHyperTreeGridNonOrientedVonNeumannSuperCursorLight::New();
  cursor->Initialize(this, index, create);
  return cursor;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::InitializeNonOrientedMooreSuperCursor(
  svtkHyperTreeGridNonOrientedMooreSuperCursor* cursor, svtkIdType index, bool create)
{
  cursor->Initialize(this, index, create);
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedMooreSuperCursor* svtkHyperTreeGrid::NewNonOrientedMooreSuperCursor(
  svtkIdType index, bool create)
{
  svtkHyperTreeGridNonOrientedMooreSuperCursor* cursor =
    svtkHyperTreeGridNonOrientedMooreSuperCursor::New();
  cursor->Initialize(this, index, create);
  return cursor;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::InitializeNonOrientedMooreSuperCursorLight(
  svtkHyperTreeGridNonOrientedMooreSuperCursorLight* cursor, svtkIdType index, bool create)
{
  cursor->Initialize(this, index, create);
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedMooreSuperCursorLight*
svtkHyperTreeGrid::NewNonOrientedMooreSuperCursorLight(svtkIdType index, bool create)
{
  svtkHyperTreeGridNonOrientedMooreSuperCursorLight* cursor =
    svtkHyperTreeGridNonOrientedMooreSuperCursorLight::New();
  cursor->Initialize(this, index, create);
  return cursor;
}

//-----------------------------------------------------------------------------
svtkHyperTree* svtkHyperTreeGrid::GetTree(svtkIdType index, bool create)
{
  // Wrap convenience macro for outside use
  svtkHyperTree* tree = GetHyperTreeFromThisMacro(index);

  // Create a new cursor if only required to do so
  if (create && !tree)
  {
    tree = svtkHyperTree::CreateInstance(this->BranchFactor, this->Dimension);
    tree->SetTreeIndex(index);
    this->HyperTrees[index] = tree;
    tree->Delete();

    // JB pour initialiser le scales au niveau de HT
    // Esperons qu'aucun HT n'est cree hors de l'appel a cette methode
    // Ce service ne devrait pas exister ou etre visible car c'est au niveau d'un HT ou d'un
    // cursor que cet appel est fait
    if (!tree->HasScales())
    {
      double origin[3];
      double scale[3];
      this->GetLevelZeroOriginAndSizeFromIndex(tree->GetTreeIndex(), origin, scale);
      tree->SetScales(std::make_shared<svtkHyperTreeGridScales>(this->BranchFactor, scale));
    }
  }

  return tree;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::SetTree(svtkIdType index, svtkHyperTree* tree)
{
  // Assign given tree at given index of hyper tree grid
  tree->SetTreeIndex(index);
  this->HyperTrees[index] = tree;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::ShallowCopy(svtkDataObject* src)
{
  svtkHyperTreeGrid* htg = svtkHyperTreeGrid::SafeDownCast(src);
  assert("src_same_type" && htg);

  // Copy member variables
  this->CopyStructure(htg);

  this->PointData->ShallowCopy(htg->GetPointData());

  // Call superclass
  this->Superclass::ShallowCopy(src);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::DeepCopy(svtkDataObject* src)
{
  assert("pre: src_exists" && src != nullptr);
  svtkHyperTreeGrid* htg = svtkHyperTreeGrid::SafeDownCast(src);
  assert("pre: same_type" && htg != nullptr);

  // Copy grid parameters
  this->ModeSqueeze = htg->ModeSqueeze;
  this->FreezeState = htg->FreezeState;
  this->Dimension = htg->Dimension;
  this->Orientation = htg->Orientation;
  this->BranchFactor = htg->BranchFactor;
  this->NumberOfChildren = htg->NumberOfChildren;
  this->DepthLimiter = htg->DepthLimiter;
  this->TransposedRootIndexing = htg->TransposedRootIndexing;
  memcpy(this->Axis, htg->GetAxes(), 2 * sizeof(unsigned int));

  this->HasInterface = htg->HasInterface;
  this->SetInterfaceNormalsName(htg->InterfaceNormalsName);
  this->SetInterfaceInterceptsName(htg->InterfaceInterceptsName);

  if (htg->Mask)
  {
    svtkNew<svtkBitArray> mask;
    this->SetMask(mask);
    this->Mask->DeepCopy(htg->Mask);
  }

  if (htg->PureMask)
  {
    if (!this->PureMask)
    {
      this->PureMask = svtkBitArray::New();
    }
    this->PureMask->DeepCopy(htg->PureMask);
    this->InitPureMask = htg->InitPureMask;
  }

  this->PointData->DeepCopy(htg->GetPointData());

  // Rectilinear part
  memcpy(this->Dimensions, htg->GetDimensions(), 3 * sizeof(unsigned int));
  memcpy(this->Extent, htg->GetExtent(), 6 * sizeof(int));
  memcpy(this->CellDims, htg->GetCellDims(), 3 * sizeof(unsigned int));
  this->DataDescription = htg->DataDescription;

  this->WithCoordinates = htg->WithCoordinates;

  if (this->WithCoordinates)
  {
    svtkDoubleArray* s;
    s = svtkDoubleArray::New();
    s->DeepCopy(htg->XCoordinates);
    this->SetXCoordinates(s);
    s->Delete();
    s = svtkDoubleArray::New();
    s->DeepCopy(htg->YCoordinates);
    this->SetYCoordinates(s);
    s->Delete();
    s = svtkDoubleArray::New();
    s->DeepCopy(htg->ZCoordinates);
    this->SetZCoordinates(s);
    s->Delete();
  }

  // Call superclass
  this->Superclass::DeepCopy(src);
  this->HyperTrees.clear();

  for (auto it = htg->HyperTrees.begin(); it != htg->HyperTrees.end(); ++it)
  {
    svtkHyperTree* tree = svtkHyperTree::CreateInstance(this->BranchFactor, this->Dimension);
    assert("pre: same_type" && tree != nullptr);
    tree->CopyStructure(it->second);
    this->HyperTrees[it->first] = tree;
    tree->Delete();
  }
}

//----------------------------------------------------------------------------
bool svtkHyperTreeGrid::RecursivelyInitializePureMask(
  svtkHyperTreeGridNonOrientedCursor* cursor, svtkDataArray* normale)
{
  // Retrieve mask value at cursor
  svtkIdType id = cursor->GetGlobalNodeIndex();
  bool mask = this->HasMask() && this->Mask->GetValue(id);

  if (!mask && normale)
  {
    double values[3];
    normale->GetTuple(id, values);
    // FR Retrieve cell interface value at cursor (is interface if one value is non null)
    bool isInterface = (values[0] != 0 || values[1] != 0 || values[2] != 0);
    // FR Cell with interface is considered as "not pure"
    mask = isInterface;
  }

  //  Dot recurse if node is masked or is a leaf
  if (!mask && !cursor->IsLeaf())
  {
    // Iterate over all chidren
    unsigned int numChildren = this->GetNumberOfChildren();
    bool pure = false;
    for (unsigned int child = 0; child < numChildren; ++child)
    {
      cursor->ToChild(child);
      // FR Obligatoire en profondeur afin d'associer une valeur a chaque maille
      pure |= this->RecursivelyInitializePureMask(cursor, normale);
      cursor->ToParent();
    }
    // Set and return pure material mask with recursively computed value
    this->PureMask->SetTuple1(id, pure);
    return pure;
  }

  // Set and return pure material mask with recursively computed value
  this->PureMask->SetTuple1(id, mask);
  return mask;
}

//----------------------------------------------------------------------------
svtkBitArray* svtkHyperTreeGrid::GetPureMask()
{
  // Check whether a pure material mask was initialized
  if (!this->InitPureMask)
  {
    if (!this->Mask || !this->Mask->GetNumberOfTuples())
    {
      // Keep track of the fact that a pure material mask now exists
      this->InitPureMask = true;
      return nullptr;
    }
    // If not, then create one
    if (this->PureMask == nullptr)
    {
      this->PureMask = svtkBitArray::New();
    }
    this->PureMask->SetNumberOfTuples(this->Mask ? this->Mask->GetNumberOfTuples() : 0);

    // Iterate over hyper tree grid
    svtkIdType index;
    svtkHyperTreeGridIterator it;
    it.Initialize(this);

    svtkDataArray* normale = nullptr;
    if (this->HasInterface)
    {
      // Interface defined
      normale = this->GetFieldData()->GetArray(this->InterfaceNormalsName);
    }

    svtkNew<svtkHyperTreeGridNonOrientedCursor> cursor;
    while (it.GetNextTree(index))
    {
      // Create cursor instance over current hyper tree
      this->InitializeNonOrientedCursor(cursor, index);
      // Recursively initialize pure material mask
      this->RecursivelyInitializePureMask(cursor, normale);
    }

    // Keep track of the fact that a pure material mask now exists
    this->InitPureMask = true;
  }

  // Return existing or created pure material mask
  return this->PureMask;
}

//----------------------------------------------------------------------------
unsigned long svtkHyperTreeGrid::GetActualMemorySizeBytes()
{
  size_t size = 0; // in bytes

  size += this->svtkDataObject::GetActualMemorySize() << 10;

  // Iterate over all trees in grid
  svtkHyperTreeGridIterator it;
  it.Initialize(this);
  while (svtkHyperTree* tree = it.GetNextTree())
  {
    size += tree->GetActualMemorySizeBytes();
  }

  // Approximate map memory size
  size += this->HyperTrees.size() * sizeof(svtkIdType) * 3;

  size += sizeof(bool);

  if (this->XCoordinates)
  {
    size += this->XCoordinates->GetActualMemorySize() << 10;
  }

  if (this->YCoordinates)
  {
    size += this->YCoordinates->GetActualMemorySize() << 10;
  }

  if (this->ZCoordinates)
  {
    size += this->ZCoordinates->GetActualMemorySize() << 10;
  }

  if (this->Mask)
  {
    size += this->Mask->GetActualMemorySize() << 10;
  }

  // JB Faut il compter le cout des grandeurs dans la representation ???
  // JB Il ne me semble pas que cela soit fait ainsi dans les autres representations
  size += this->PointData->GetActualMemorySize() << 10;

  return static_cast<unsigned long>(size);
}

//----------------------------------------------------------------------------
unsigned long svtkHyperTreeGrid::GetActualMemorySize()
{
  // in kilibytes
  return (this->GetActualMemorySizeBytes() >> 10);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::GetIndexFromLevelZeroCoordinates(
  svtkIdType& treeindex, unsigned int i, unsigned int j, unsigned int k) const
{
  // Distinguish between two cases depending on indexing order
  if (this->TransposedRootIndexing)
  {
    treeindex = static_cast<svtkIdType>(k) +
      static_cast<svtkIdType>(this->CellDims[2]) *
        (static_cast<svtkIdType>(j) +
          static_cast<svtkIdType>(i) * static_cast<svtkIdType>(this->CellDims[1]));
  }
  else
  {
    treeindex = static_cast<svtkIdType>(i) +
      static_cast<svtkIdType>(this->CellDims[0]) *
        (static_cast<svtkIdType>(j) +
          static_cast<svtkIdType>(k) * static_cast<svtkIdType>(this->CellDims[1]));
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGrid::GetShiftedLevelZeroIndex(
  svtkIdType treeindex, unsigned int i, unsigned int j, unsigned int k) const
{
  svtkIdType dtreeindex = 0;
  this->GetIndexFromLevelZeroCoordinates(dtreeindex, i, j, k);
  return treeindex + dtreeindex;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::GetLevelZeroCoordinatesFromIndex(
  svtkIdType treeindex, unsigned int& i, unsigned int& j, unsigned int& k) const
{
  // Distinguish between two cases depending on indexing order
  if (this->TransposedRootIndexing)
  {
    unsigned long nbKxJ =
      static_cast<unsigned long>(this->CellDims[2]) * static_cast<unsigned long>(this->CellDims[1]);
    i = static_cast<unsigned int>(treeindex / nbKxJ);
    svtkIdType reste = treeindex - i * nbKxJ;
    j = static_cast<unsigned int>(reste / this->CellDims[2]);
    k = static_cast<unsigned int>(reste - j * this->CellDims[2]);
  }
  else
  {
    unsigned long nbIxJ = this->CellDims[0] * this->CellDims[1];
    k = static_cast<unsigned int>(treeindex / nbIxJ);
    svtkIdType reste = treeindex - k * nbIxJ;
    j = static_cast<unsigned int>(reste / this->CellDims[0]);
    i = static_cast<unsigned int>(reste - j * this->CellDims[0]);
  }

  assert(i < this->CellDims[0]);
  assert(j < this->CellDims[1]);
  assert(k < this->CellDims[2]);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::GetLevelZeroOriginAndSizeFromIndex(
  svtkIdType treeindex, double* Origin, double* Size)
{
  assert("pre: exist_coordinates_explict" && this->WithCoordinates);

  // Compute origin and size of the cursor
  unsigned int i, j, k;
  this->GetLevelZeroCoordinatesFromIndex(treeindex, i, j, k);

  svtkDataArray* xCoords = this->XCoordinates;
  svtkDataArray* yCoords = this->YCoordinates;
  svtkDataArray* zCoords = this->ZCoordinates;
  Origin[0] = xCoords->GetTuple1(i);
  Origin[1] = yCoords->GetTuple1(j);
  Origin[2] = zCoords->GetTuple1(k);

  if (this->Dimensions[0] == 1)
  {
    Size[0] = 0.;
  }
  else
  {
    Size[0] = xCoords->GetTuple1(i + 1) - Origin[0];
  }
  if (this->Dimensions[1] == 1)
  {
    Size[1] = 0.;
  }
  else
  {
    Size[1] = yCoords->GetTuple1(j + 1) - Origin[1];
  }
  if (this->Dimensions[2] == 1)
  {
    Size[2] = 0.;
  }
  else
  {
    Size[2] = zCoords->GetTuple1(k + 1) - Origin[2];
  }
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::GetLevelZeroOriginFromIndex(svtkIdType treeindex, double* Origin)
{
  assert("pre: exist_coordinates_explict" && this->WithCoordinates);

  // Compute origin and size of the cursor
  unsigned int i, j, k;
  this->GetLevelZeroCoordinatesFromIndex(treeindex, i, j, k);

  svtkDataArray* xCoords = this->XCoordinates;
  svtkDataArray* yCoords = this->YCoordinates;
  svtkDataArray* zCoords = this->ZCoordinates;
  Origin[0] = xCoords->GetTuple1(i);
  Origin[1] = yCoords->GetTuple1(j);
  Origin[2] = zCoords->GetTuple1(k);
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGrid::GetGlobalNodeIndexMax()
{
  // Iterate over all hyper trees
  svtkIdType max = 0;
  svtkHyperTree* crtTree = nullptr;
  svtkHyperTreeGridIterator it;
  this->InitializeTreeIterator(it);
  while ((crtTree = it.GetNextTree()))
  {
    max = std::max(max, crtTree->GetGlobalNodeIndexMax());
  } // it
  return max;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::InitializeLocalIndexNode()
{
  // Iterate over all hyper trees
  svtkIdType local = 0;
  svtkHyperTree* crtTree = nullptr;
  svtkHyperTreeGridIterator it;
  this->InitializeTreeIterator(it);
  while ((crtTree = it.GetNextTree()))
  {
    crtTree->SetGlobalIndexStart(local);
    local += crtTree->GetNumberOfVertices();
  } // it
}

//=============================================================================
// Hyper tree grid iterator
// Implemented here because it needs access to the internal classes.
//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::svtkHyperTreeGridIterator::Initialize(svtkHyperTreeGrid* grid)
{
  assert(grid != nullptr);
  this->Grid = grid;
  this->Iterator = grid->HyperTrees.begin();
}

//-----------------------------------------------------------------------------
svtkHyperTree* svtkHyperTreeGrid::svtkHyperTreeGridIterator::GetNextTree(svtkIdType& index)
{
  if (this->Iterator == this->Grid->HyperTrees.end())
  {
    return nullptr;
  }
  svtkHyperTree* t = this->Iterator->second.GetPointer();
  index = this->Iterator->first;
  ++this->Iterator;
  return t;
}

//-----------------------------------------------------------------------------
svtkHyperTree* svtkHyperTreeGrid::svtkHyperTreeGridIterator::GetNextTree()
{
  svtkIdType index;
  return GetNextTree(index);
}

//=============================================================================
// Hard-coded child mask bitcodes
static const unsigned int HyperTreeGridMask_1_2[2] = { 0x80000000, 0x20000000 };

static const unsigned int HyperTreeGridMask_1_3[3] = { 0x80000000, 0x40000000, 0x20000000 };

static const unsigned int HyperTreeGridMask_2_2[4] = { 0xd0000000, 0x64000000, 0x13000000,
  0x05800000 };

static const unsigned int HyperTreeGridMask_2_3[9] = { 0xd0000000, 0x40000000, 0x64000000,
  0x10000000, 0x08000000, 0x04000000, 0x13000000, 0x01000000, 0x05800000 };

static const unsigned int HyperTreeGridMask_3_2[8] = { 0xd8680000, 0x6c320000, 0x1b098000,
  0x0d82c000, 0x00683600, 0x00321b00, 0x000986c0, 0x0002c360 };

static const unsigned int HyperTreeGridMask_3_3[27] = { 0xd8680000, 0x48200000, 0x6c320000,
  0x18080000, 0x08000000, 0x0c020000, 0x1b098000, 0x09008000, 0x0d82c000, 0x00680000, 0x00200000,
  0x00320000, 0x00080000, 0x00040000, 0x00020000, 0x00098000, 0x00008000, 0x0002c000, 0x00683600,
  0x00201200, 0x00321b00, 0x00080600, 0x00000200, 0x00020300, 0x000986c0, 0x00008240, 0x0002c360 };

static const unsigned int* HyperTreeGridMask[3][2] = {
  { HyperTreeGridMask_1_2, HyperTreeGridMask_1_3 },
  { HyperTreeGridMask_2_2, HyperTreeGridMask_2_3 }, { HyperTreeGridMask_3_2, HyperTreeGridMask_3_3 }
};

//-----------------------------------------------------------------------------
unsigned int svtkHyperTreeGrid::GetChildMask(unsigned int child)
{
  int i = this->GetDimension() - 1;
  int j = this->GetBranchFactor() - 2;
  return HyperTreeGridMask[i][j][child];
}

//-----------------------------------------------------------------------------
double* svtkHyperTreeGrid::GetBounds()
{
  assert("pre: exist_coordinates_explict" && this->WithCoordinates);

  // Recompute each call
  // Retrieve coordinate arrays
  svtkDataArray* coords[3] = { this->XCoordinates, this->YCoordinates, this->ZCoordinates };
  for (unsigned int i = 0; i < 3; ++i)
  {
    if (!coords[i] || !coords[i]->GetNumberOfTuples())
    {
      return nullptr;
    }
  }

  // Get bounds from coordinate arrays
  for (unsigned int i = 0; i < 3; ++i)
  {
    unsigned int di = 2 * i;
    unsigned int dip = di + 1;
    this->Bounds[di] = coords[i]->GetComponent(0, 0);
    this->Bounds[dip] = coords[i]->GetComponent(coords[i]->GetNumberOfTuples() - 1, 0);

    // Ensure that the bounds are increasing
    if (this->Bounds[di] > this->Bounds[dip])
    {
      std::swap(this->Bounds[di], this->Bounds[dip]);
    }
  }

  return this->Bounds;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::GetBounds(double* obds)
{
  double* bds = this->GetBounds();
  memcpy(obds, bds, 6 * sizeof(double));
}

//-----------------------------------------------------------------------------
double* svtkHyperTreeGrid::GetCenter()
{
  double* bds = this->GetBounds();
  this->Center[0] = bds[0] + (bds[1] - bds[0]) / 2.0;
  this->Center[1] = bds[2] + (bds[3] - bds[2]) / 2.0;
  this->Center[2] = bds[4] + (bds[5] - bds[4]) / 2.0;
  return this->Center;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGrid::GetCenter(double* octr)
{
  double* ctr = this->GetCenter();
  memcpy(octr, ctr, 3 * sizeof(double));
}

//-----------------------------------------------------------------------------
svtkPointData* svtkHyperTreeGrid::GetPointData()
{
  return this->PointData.GetPointer();
}

//-----------------------------------------------------------------------------
svtkUnsignedCharArray* svtkHyperTreeGrid::GetTreeGhostArray()
{
  if (!this->TreeGhostArrayCached)
  {
    this->TreeGhostArray = svtkArrayDownCast<svtkUnsignedCharArray>(
      this->GetPointData()->GetArray(svtkDataSetAttributes::GhostArrayName()));
    this->TreeGhostArrayCached = true;
  }
  assert(this->TreeGhostArray ==
    svtkArrayDownCast<svtkUnsignedCharArray>(
      this->GetPointData()->GetArray(svtkDataSetAttributes::GhostArrayName())));
  return this->TreeGhostArray;
}

//----------------------------------------------------------------------------
svtkUnsignedCharArray* svtkHyperTreeGrid::AllocateTreeGhostArray()
{
  if (!this->GetTreeGhostArray())
  {
    svtkNew<svtkUnsignedCharArray> ghosts;
    ghosts->SetName(svtkDataSetAttributes::GhostArrayName());
    ghosts->SetNumberOfComponents(1);
    ghosts->SetNumberOfTuples(this->GetMaxNumberOfTrees());
    ghosts->Fill(0);
    this->GetPointData()->AddArray(ghosts);
    ghosts->Delete();
    this->TreeGhostArray = ghosts;
    this->TreeGhostArrayCached = true;
  }
  return this->TreeGhostArray;
}

//----------------------------------------------------------------------------
svtkUnsignedCharArray* svtkHyperTreeGrid::GetGhostCells()
{
  return svtkUnsignedCharArray::SafeDownCast(
    this->PointData->GetArray(svtkDataSetAttributes::GhostArrayName()));
}

//----------------------------------------------------------------------------
bool svtkHyperTreeGrid::HasAnyGhostCells() const
{
  return this->PointData->GetArray(svtkDataSetAttributes::GhostArrayName()) != nullptr;
}
#pragma GCC diagnostic pop
