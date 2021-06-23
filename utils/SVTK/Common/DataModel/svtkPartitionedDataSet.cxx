/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPartitionedDataSet.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkPartitionedDataSet.h"

#include "svtkDataSet.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkPartitionedDataSet);
//----------------------------------------------------------------------------
svtkPartitionedDataSet::svtkPartitionedDataSet() {}

//----------------------------------------------------------------------------
svtkPartitionedDataSet::~svtkPartitionedDataSet() {}

//----------------------------------------------------------------------------
svtkPartitionedDataSet* svtkPartitionedDataSet::GetData(svtkInformation* info)
{
  return info ? svtkPartitionedDataSet::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkPartitionedDataSet* svtkPartitionedDataSet::GetData(svtkInformationVector* v, int i)
{
  return svtkPartitionedDataSet::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
void svtkPartitionedDataSet::SetNumberOfPartitions(unsigned int numPartitions)
{
  this->Superclass::SetNumberOfChildren(numPartitions);
}

//----------------------------------------------------------------------------
unsigned int svtkPartitionedDataSet::GetNumberOfPartitions()
{
  return this->Superclass::GetNumberOfChildren();
}

//----------------------------------------------------------------------------
svtkDataSet* svtkPartitionedDataSet::GetPartition(unsigned int idx)
{
  return svtkDataSet::SafeDownCast(this->GetPartitionAsDataObject(idx));
}

//----------------------------------------------------------------------------
svtkDataObject* svtkPartitionedDataSet::GetPartitionAsDataObject(unsigned int idx)
{
  return this->Superclass::GetChild(idx);
}

//----------------------------------------------------------------------------
void svtkPartitionedDataSet::SetPartition(unsigned int idx, svtkDataObject* partition)
{
  if (partition && partition->IsA("svtkCompositeDataSet"))
  {
    svtkErrorMacro("Partition cannot be a svtkCompositeDataSet.");
    return;
  }

  this->Superclass::SetChild(idx, partition);
}

//----------------------------------------------------------------------------
void svtkPartitionedDataSet::RemoveNullPartitions()
{
  unsigned int next = 0;
  for (unsigned int cc = 0; cc < this->GetNumberOfPartitions(); ++cc)
  {
    auto ds = this->GetPartition(cc);
    if (ds)
    {
      if (next < cc)
      {
        this->SetPartition(next, ds);
        if (this->HasChildMetaData(cc))
        {
          this->SetChildMetaData(next, this->GetChildMetaData(cc));
        }
        this->SetPartition(cc, nullptr);
        this->SetChildMetaData(cc, nullptr);
      }
      next++;
    }
  }
  this->SetNumberOfPartitions(next);
}

//----------------------------------------------------------------------------
void svtkPartitionedDataSet::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
