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
#include "svtkUniformGridAMR.h"
#include "svtkAMRDataInternals.h"
#include "svtkAMRInformation.h"
#include "svtkInformation.h"
#include "svtkInformationKey.h"
#include "svtkInformationVector.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkType.h"
#include "svtkUniformGrid.h"
#include "svtkUniformGridAMRDataIterator.h"

svtkStandardNewMacro(svtkUniformGridAMR);

//----------------------------------------------------------------------------
svtkUniformGridAMR::svtkUniformGridAMR()
{
  this->Bounds[0] = SVTK_DOUBLE_MAX;
  this->Bounds[1] = SVTK_DOUBLE_MIN;
  this->Bounds[2] = SVTK_DOUBLE_MAX;
  this->Bounds[3] = SVTK_DOUBLE_MIN;
  this->Bounds[4] = SVTK_DOUBLE_MAX;
  this->Bounds[5] = SVTK_DOUBLE_MIN;
  this->AMRInfo = nullptr;
  this->AMRData = svtkAMRDataInternals::New();
}

//----------------------------------------------------------------------------
svtkUniformGridAMR::~svtkUniformGridAMR()
{
  if (this->AMRInfo)
  {
    this->AMRInfo->Delete();
  }
  this->AMRData->Delete();
}

//----------------------------------------------------------------------------
void svtkUniformGridAMR::SetAMRInfo(svtkAMRInformation* amrInfo)
{
  if (amrInfo == this->AMRInfo)
  {
    return;
  }
  if (this->AMRInfo)
  {
    this->AMRInfo->Delete();
  }
  this->AMRInfo = amrInfo;
  if (this->AMRInfo)
  {
    this->AMRInfo->Register(this);
  }
  this->Modified();
}

//----------------------------------------------------------------------------
svtkUniformGrid* svtkUniformGridAMR::GetDataSet(unsigned int level, unsigned int idx)
{
  return this->AMRData->GetDataSet(this->GetCompositeIndex(level, idx));
}

//----------------------------------------------------------------------------
svtkCompositeDataIterator* svtkUniformGridAMR::NewIterator()
{
  svtkUniformGridAMRDataIterator* iter = svtkUniformGridAMRDataIterator::New();
  iter->SetDataSet(this);
  return iter;
}

//----------------------------------------------------------------------------
void svtkUniformGridAMR::Initialize()
{
  this->Initialize(0, nullptr);
}

//----------------------------------------------------------------------------
void svtkUniformGridAMR::Initialize(int numLevels, const int* blocksPerLevel)
{
  this->Bounds[0] = SVTK_DOUBLE_MAX;
  this->Bounds[1] = SVTK_DOUBLE_MIN;
  this->Bounds[2] = SVTK_DOUBLE_MAX;
  this->Bounds[3] = SVTK_DOUBLE_MIN;
  this->Bounds[4] = SVTK_DOUBLE_MAX;
  this->Bounds[5] = SVTK_DOUBLE_MIN;

  svtkSmartPointer<svtkAMRInformation> amrInfo = svtkSmartPointer<svtkAMRInformation>::New();
  this->SetAMRInfo(amrInfo);
  this->AMRInfo->Initialize(numLevels, blocksPerLevel);
  this->AMRData->Initialize();
}

//----------------------------------------------------------------------------
unsigned int svtkUniformGridAMR::GetNumberOfLevels()
{
  unsigned int nlev = 0;
  if (this->AMRInfo)
  {
    nlev = this->AMRInfo->GetNumberOfLevels();
  }
  return nlev;
}

//----------------------------------------------------------------------------
unsigned int svtkUniformGridAMR::GetTotalNumberOfBlocks()
{
  unsigned int nblocks = 0;
  if (this->AMRInfo)
  {
    nblocks = this->AMRInfo->GetTotalNumberOfBlocks();
  }
  return nblocks;
}

//----------------------------------------------------------------------------
unsigned int svtkUniformGridAMR::GetNumberOfDataSets(const unsigned int level)
{
  unsigned int ndata = 0;
  if (this->AMRInfo)
  {
    ndata = this->AMRInfo->GetNumberOfDataSets(level);
  }
  return ndata;
}

//----------------------------------------------------------------------------
void svtkUniformGridAMR::SetDataSet(unsigned int level, unsigned int idx, svtkUniformGrid* grid)
{
  if (!grid)
  {
    return; // nullptr grid, nothing to do
  }
  if (level >= this->GetNumberOfLevels() || idx >= this->GetNumberOfDataSets(level))
  {
    svtkErrorMacro("Invalid data set index: " << level << " " << idx);
    return;
  }

  if (this->AMRInfo->GetGridDescription() < 0)
  {
    this->AMRInfo->SetGridDescription(grid->GetGridDescription());
  }
  else if (grid->GetGridDescription() != this->AMRInfo->GetGridDescription())
  {
    svtkErrorMacro("Inconsistent types of svtkUniformGrid");
    return;
  }
  int index = this->AMRInfo->GetIndex(level, idx);
  this->AMRData->Insert(index, grid);

  // update bounds
  double bb[6];
  grid->GetBounds(bb);
  // update bounds
  for (int i = 0; i < 3; ++i)
  {
    if (bb[i * 2] < this->Bounds[i * 2])
    {
      this->Bounds[i * 2] = bb[i * 2];
    }
    if (bb[i * 2 + 1] > this->Bounds[i * 2 + 1])
    {
      this->Bounds[i * 2 + 1] = bb[i * 2 + 1];
    }
  } // END for each dimension
}

//----------------------------------------------------------------------------
void svtkUniformGridAMR::SetDataSet(svtkCompositeDataIterator* compositeIter, svtkDataObject* dataObj)
{
  svtkUniformGridAMRDataIterator* itr = svtkUniformGridAMRDataIterator::SafeDownCast(compositeIter);
  svtkUniformGrid* grid = svtkUniformGrid::SafeDownCast(dataObj);
  int level = itr->GetCurrentLevel();
  int id = itr->GetCurrentIndex();
  this->SetDataSet(level, id, grid);
};

//----------------------------------------------------------------------------
void svtkUniformGridAMR::SetGridDescription(int gridDescription)
{
  if (this->AMRInfo)
  {
    this->AMRInfo->SetGridDescription(gridDescription);
  }
}

//----------------------------------------------------------------------------
int svtkUniformGridAMR::GetGridDescription()
{
  int desc = 0;
  if (this->AMRInfo)
  {
    desc = this->AMRInfo->GetGridDescription();
  }
  return desc;
}

//------------------------------------------------------------------------------
svtkDataObject* svtkUniformGridAMR::GetDataSet(svtkCompositeDataIterator* compositeIter)
{
  svtkUniformGridAMRDataIterator* itr = svtkUniformGridAMRDataIterator::SafeDownCast(compositeIter);
  if (!itr)
  {
    return nullptr;
  }
  int level = itr->GetCurrentLevel();
  int id = itr->GetCurrentIndex();
  return this->GetDataSet(level, id);
}

//----------------------------------------------------------------------------
int svtkUniformGridAMR::GetCompositeIndex(const unsigned int level, const unsigned int index)
{

  if (level >= this->GetNumberOfLevels() || index >= this->GetNumberOfDataSets(level))
  {
    svtkErrorMacro("Invalid level-index pair: " << level << ", " << index);
    return 0;
  }
  return this->AMRInfo->GetIndex(level, index);
}
//----------------------------------------------------------------------------
void svtkUniformGridAMR::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
void svtkUniformGridAMR::GetLevelAndIndex(
  unsigned int flatIdx, unsigned int& level, unsigned int& idx)
{
  this->AMRInfo->ComputeIndexPair(flatIdx, level, idx);
}

//----------------------------------------------------------------------------
svtkUniformGridAMR* svtkUniformGridAMR::GetData(svtkInformation* info)
{
  return info ? svtkUniformGridAMR::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkUniformGridAMR* svtkUniformGridAMR::GetData(svtkInformationVector* v, int i)
{
  return svtkUniformGridAMR::GetData(v->GetInformationObject(i));
}

//------------------------------------------------------------------------------
void svtkUniformGridAMR::ShallowCopy(svtkDataObject* src)
{
  if (src == this)
  {
    return;
  }

  this->Superclass::ShallowCopy(src);

  if (svtkUniformGridAMR* hbds = svtkUniformGridAMR::SafeDownCast(src))
  {
    this->SetAMRInfo(hbds->GetAMRInfo());
    this->AMRData->ShallowCopy(hbds->GetAMRData());
    memcpy(this->Bounds, hbds->Bounds, sizeof(double) * 6);
  }

  this->Modified();
}

//------------------------------------------------------------------------------
void svtkUniformGridAMR::DeepCopy(svtkDataObject* src)
{
  if (src == this)
  {
    return;
  }

  this->Superclass::DeepCopy(src);

  if (svtkUniformGridAMR* hbds = svtkUniformGridAMR::SafeDownCast(src))
  {
    this->SetAMRInfo(nullptr);
    this->AMRInfo = svtkAMRInformation::New();
    this->AMRInfo->DeepCopy(hbds->GetAMRInfo());
    memcpy(this->Bounds, hbds->Bounds, sizeof(double) * 6);
  }

  this->Modified();
}

//------------------------------------------------------------------------------
void svtkUniformGridAMR::CopyStructure(svtkCompositeDataSet* src)
{
  if (src == this)
  {
    return;
  }

  if (svtkUniformGridAMR* hbds = svtkUniformGridAMR::SafeDownCast(src))
  {
    this->SetAMRInfo(hbds->GetAMRInfo());
  }

  this->Modified();
}

//----------------------------------------------------------------------------
const double* svtkUniformGridAMR::GetBounds()
{
  return !this->AMRData->Empty() ? this->Bounds : this->AMRInfo->GetBounds();
}

//----------------------------------------------------------------------------
void svtkUniformGridAMR::GetBounds(double bounds[6])
{
  const double* bb = this->GetBounds();
  for (int i = 0; i < 6; ++i)
  {
    bounds[i] = bb[i];
  }
}

//----------------------------------------------------------------------------
void svtkUniformGridAMR::GetMin(double min[3])
{
  const double* bb = this->GetBounds();
  min[0] = bb[0];
  min[1] = bb[2];
  min[2] = bb[4];
}

//----------------------------------------------------------------------------
void svtkUniformGridAMR::GetMax(double max[3])
{
  const double* bb = this->GetBounds();
  max[0] = bb[1];
  max[1] = bb[3];
  max[2] = bb[5];
}
