/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUniformGridAMR.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkOverlappingAMR.h"
#include "svtkAMRInformation.h"
#include "svtkCellData.h"
#include "svtkDataSetAttributes.h"
#include "svtkInformationIdTypeKey.h"
#include "svtkObjectFactory.h"
#include "svtkUniformGrid.h"
#include "svtkUniformGridAMRDataIterator.h"
#include "svtkUnsignedCharArray.h"
#include <vector>

svtkStandardNewMacro(svtkOverlappingAMR);

svtkInformationKeyMacro(svtkOverlappingAMR, NUMBER_OF_BLANKED_POINTS, IdType);

//----------------------------------------------------------------------------
svtkOverlappingAMR::svtkOverlappingAMR() = default;

//----------------------------------------------------------------------------
svtkOverlappingAMR::~svtkOverlappingAMR() = default;

//----------------------------------------------------------------------------
void svtkOverlappingAMR::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  if (this->AMRInfo)
  {
    this->AMRInfo->PrintSelf(os, indent);
  }
}

//----------------------------------------------------------------------------
svtkCompositeDataIterator* svtkOverlappingAMR::NewIterator()
{
  svtkUniformGridAMRDataIterator* iter = svtkUniformGridAMRDataIterator::New();
  iter->SetDataSet(this);
  return iter;
}

//----------------------------------------------------------------------------
void svtkOverlappingAMR::SetRefinementRatio(unsigned int level, int ratio)
{
  this->AMRInfo->SetRefinementRatio(level, ratio);
}

//----------------------------------------------------------------------------
int svtkOverlappingAMR::GetRefinementRatio(unsigned int level)
{
  if (!AMRInfo->HasRefinementRatio())
  {
    AMRInfo->GenerateRefinementRatio();
  }
  return this->AMRInfo->GetRefinementRatio(level);
}

//----------------------------------------------------------------------------
int svtkOverlappingAMR::GetRefinementRatio(svtkCompositeDataIterator* iter)
{
  svtkUniformGridAMRDataIterator* amrIter = svtkUniformGridAMRDataIterator::SafeDownCast(iter);

  unsigned int level = amrIter->GetCurrentLevel();
  return this->AMRInfo->GetRefinementRatio(level);
}

//----------------------------------------------------------------------------
void svtkOverlappingAMR::GenerateParentChildInformation()
{
  this->AMRInfo->GenerateParentChildInformation();
}

//----------------------------------------------------------------------------
bool svtkOverlappingAMR::HasChildrenInformation()
{
  return AMRInfo->HasChildrenInformation();
}

//----------------------------------------------------------------------------
unsigned int* svtkOverlappingAMR::GetParents(
  unsigned int level, unsigned int index, unsigned int& num)
{
  return this->AMRInfo->GetParents(level, index, num);
}

//------------------------------------------------------------------------------
unsigned int* svtkOverlappingAMR::GetChildren(
  unsigned int level, unsigned int index, unsigned int& num)
{
  return this->AMRInfo->GetChildren(level, index, num);
}

//------------------------------------------------------------------------------
void svtkOverlappingAMR::PrintParentChildInfo(unsigned int level, unsigned int index)
{
  this->AMRInfo->PrintParentChildInfo(level, index);
}

//------------------------------------------------------------------------------
void svtkOverlappingAMR::SetAMRBox(unsigned int level, unsigned int id, const svtkAMRBox& box)
{
  this->AMRInfo->SetAMRBox(level, id, box);
}

//------------------------------------------------------------------------------
const svtkAMRBox& svtkOverlappingAMR::GetAMRBox(unsigned int level, unsigned int id)
{
  const svtkAMRBox& box = this->AMRInfo->GetAMRBox(level, id);
  if (box.IsInvalid())
  {
    svtkErrorMacro("Invalid AMR box");
  }
  return box;
}

//------------------------------------------------------------------------------
void svtkOverlappingAMR::SetSpacing(unsigned int level, const double spacing[3])
{
  this->AMRInfo->SetSpacing(level, spacing);
}

//------------------------------------------------------------------------------
void svtkOverlappingAMR::GetSpacing(unsigned int level, double spacing[3])
{
  return this->AMRInfo->GetSpacing(level, spacing);
}

//----------------------------------------------------------------------------
void svtkOverlappingAMR::GetBounds(unsigned int level, unsigned int id, double bb[6])
{
  this->AMRInfo->GetBounds(level, id, bb);
}

//----------------------------------------------------------------------------
void svtkOverlappingAMR::GetOrigin(unsigned int level, unsigned int id, double origin[3])
{
  double bb[6];
  this->GetBounds(level, id, bb);
  origin[0] = bb[0];
  origin[1] = bb[2];
  origin[2] = bb[4];
}

//----------------------------------------------------------------------------
void svtkOverlappingAMR::SetOrigin(const double origin[3])
{
  return this->AMRInfo->SetOrigin(origin);
}

//----------------------------------------------------------------------------
double* svtkOverlappingAMR::GetOrigin()
{
  return this->AMRInfo ? this->AMRInfo->GetOrigin() : nullptr;
}

//----------------------------------------------------------------------------
void svtkOverlappingAMR::SetAMRBlockSourceIndex(unsigned int level, unsigned int id, int sourceId)
{
  unsigned int index = this->AMRInfo->GetIndex(level, id);
  this->AMRInfo->SetAMRBlockSourceIndex(index, sourceId);
}

//----------------------------------------------------------------------------
int svtkOverlappingAMR::GetAMRBlockSourceIndex(unsigned int level, unsigned int id)
{
  unsigned int index = this->AMRInfo->GetIndex(level, id);
  return this->AMRInfo->GetAMRBlockSourceIndex(index);
}

//----------------------------------------------------------------------------
void svtkOverlappingAMR::Audit()
{
  this->AMRInfo->Audit();

  int emptyDimension(-1);
  switch (this->GetGridDescription())
  {
    case SVTK_YZ_PLANE:
      emptyDimension = 0;
      break;
    case SVTK_XZ_PLANE:
      emptyDimension = 1;
      break;
    case SVTK_XY_PLANE:
      emptyDimension = 2;
      break;
  }

  svtkSmartPointer<svtkUniformGridAMRDataIterator> iter;
  iter.TakeReference(svtkUniformGridAMRDataIterator::SafeDownCast(this->NewIterator()));
  iter->SetSkipEmptyNodes(1);
  for (iter->GoToFirstItem(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
  {
    svtkUniformGrid* grid = svtkUniformGrid::SafeDownCast(iter->GetCurrentDataObject());
    int hasGhost = grid->HasAnyGhostCells();

    unsigned int level = iter->GetCurrentLevel();
    unsigned int id = iter->GetCurrentIndex();
    const svtkAMRBox& box = this->AMRInfo->GetAMRBox(level, id);
    int dims[3];
    box.GetNumberOfNodes(dims);

    double spacing[3];
    this->GetSpacing(level, spacing);

    double origin[3];
    this->GetOrigin(level, id, origin);

    for (int d = 0; d < 3; d++)
    {
      if (d == emptyDimension)
      {
        if (grid->GetSpacing()[d] != spacing[d])
        {
          svtkErrorMacro(
            "The grid spacing does not match AMRInfo at (" << level << ", " << id << ")");
        }
        if (!hasGhost && grid->GetOrigin()[d] != origin[d])
        {
          svtkErrorMacro(
            "The grid origin does not match AMRInfo at (" << level << ", " << id << ")");
        }
        if (!hasGhost && grid->GetDimensions()[d] != dims[d])
        {
          svtkErrorMacro(
            "The grid dimensions does not match AMRInfo at (" << level << ", " << id << ")");
        }
      }
    }
  }
}

bool svtkOverlappingAMR::FindGrid(double q[3], unsigned int& level, unsigned int& gridId)
{
  return this->AMRInfo->FindGrid(q, level, gridId);
}
