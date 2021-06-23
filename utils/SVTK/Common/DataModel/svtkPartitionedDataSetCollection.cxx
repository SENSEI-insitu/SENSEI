/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPartitionedDataSetCollection.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkPartitionedDataSetCollection.h"

#include "svtkDataSet.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"
#include "svtkPartitionedDataSet.h"

svtkStandardNewMacro(svtkPartitionedDataSetCollection);
//----------------------------------------------------------------------------
svtkPartitionedDataSetCollection::svtkPartitionedDataSetCollection() {}

//----------------------------------------------------------------------------
svtkPartitionedDataSetCollection::~svtkPartitionedDataSetCollection() {}

//----------------------------------------------------------------------------
svtkPartitionedDataSetCollection* svtkPartitionedDataSetCollection::GetData(svtkInformation* info)
{
  return info ? svtkPartitionedDataSetCollection::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkPartitionedDataSetCollection* svtkPartitionedDataSetCollection::GetData(
  svtkInformationVector* v, int i)
{
  return svtkPartitionedDataSetCollection::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
void svtkPartitionedDataSetCollection::SetNumberOfPartitionedDataSets(unsigned int numDataSets)
{
  this->Superclass::SetNumberOfChildren(numDataSets);
}

//----------------------------------------------------------------------------
unsigned int svtkPartitionedDataSetCollection::GetNumberOfPartitionedDataSets()
{
  return this->Superclass::GetNumberOfChildren();
}

//----------------------------------------------------------------------------
svtkPartitionedDataSet* svtkPartitionedDataSetCollection::GetPartitionedDataSet(unsigned int idx)
{
  return svtkPartitionedDataSet::SafeDownCast(this->Superclass::GetChild(idx));
}

//----------------------------------------------------------------------------
void svtkPartitionedDataSetCollection::SetPartitionedDataSet(
  unsigned int idx, svtkPartitionedDataSet* dataset)
{
  this->Superclass::SetChild(idx, dataset);
}

//----------------------------------------------------------------------------
void svtkPartitionedDataSetCollection::RemovePartitionedDataSet(unsigned int idx)
{
  this->Superclass::RemoveChild(idx);
}

//----------------------------------------------------------------------------
void svtkPartitionedDataSetCollection::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
