/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCompositeDataSet.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkCompositeDataSet.h"

#include "svtkBoundingBox.h"
#include "svtkCompositeDataIterator.h"
#include "svtkDataSet.h"
#include "svtkInformation.h"
#include "svtkInformationIntegerKey.h"
#include "svtkInformationStringKey.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"
#include "svtkSmartPointer.h"

svtkInformationKeyMacro(svtkCompositeDataSet, NAME, String);
svtkInformationKeyMacro(svtkCompositeDataSet, CURRENT_PROCESS_CAN_LOAD_BLOCK, Integer);

//----------------------------------------------------------------------------
svtkCompositeDataSet::svtkCompositeDataSet() = default;

//----------------------------------------------------------------------------
svtkCompositeDataSet::~svtkCompositeDataSet() = default;

//----------------------------------------------------------------------------
svtkCompositeDataSet* svtkCompositeDataSet::GetData(svtkInformation* info)
{
  return info ? svtkCompositeDataSet::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkCompositeDataSet* svtkCompositeDataSet::GetData(svtkInformationVector* v, int i)
{
  return svtkCompositeDataSet::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
void svtkCompositeDataSet::ShallowCopy(svtkDataObject* src)
{
  if (src == this)
  {
    return;
  }

  this->Superclass::ShallowCopy(src);
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkCompositeDataSet::DeepCopy(svtkDataObject* src)
{
  if (src == this)
  {
    return;
  }

  this->Superclass::DeepCopy(src);
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkCompositeDataSet::Initialize()
{
  this->Superclass::Initialize();
}

//----------------------------------------------------------------------------
unsigned long svtkCompositeDataSet::GetActualMemorySize()
{
  unsigned long memSize = 0;
  svtkCompositeDataIterator* iter = this->NewIterator();
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
  {
    svtkDataObject* dobj = iter->GetCurrentDataObject();
    memSize += dobj->GetActualMemorySize();
  }
  iter->Delete();
  return memSize;
}

//----------------------------------------------------------------------------
svtkIdType svtkCompositeDataSet::GetNumberOfPoints()
{
  return this->GetNumberOfElements(svtkDataSet::POINT);
}

//----------------------------------------------------------------------------
svtkIdType svtkCompositeDataSet::GetNumberOfCells()
{
  return this->GetNumberOfElements(svtkDataSet::CELL);
}

//----------------------------------------------------------------------------
svtkIdType svtkCompositeDataSet::GetNumberOfElements(int type)
{
  svtkSmartPointer<svtkCompositeDataIterator> iter;
  iter.TakeReference(this->NewIterator());
  iter->SkipEmptyNodesOn();
  svtkIdType numElements = 0;
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
  {
    numElements += iter->GetCurrentDataObject()->GetNumberOfElements(type);
  }

  return numElements;
}

//----------------------------------------------------------------------------
void svtkCompositeDataSet::GetBounds(double bounds[6])
{
  double bds[6];
  svtkBoundingBox bbox;
  svtkCompositeDataIterator* iter = this->NewIterator();

  for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
  {
    svtkDataSet* ds = svtkDataSet::SafeDownCast(iter->GetCurrentDataObject());
    if (ds)
    {
      ds->GetBounds(bds);
      bbox.AddBounds(bds);
    }
  }

  bbox.GetBounds(bounds);
  iter->Delete();
}

//----------------------------------------------------------------------------
void svtkCompositeDataSet::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
