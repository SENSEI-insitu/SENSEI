/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMultiBlockDataSet.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkMultiBlockDataSet.h"

#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkMultiBlockDataSet);
//----------------------------------------------------------------------------
svtkMultiBlockDataSet::svtkMultiBlockDataSet() = default;

//----------------------------------------------------------------------------
svtkMultiBlockDataSet::~svtkMultiBlockDataSet() = default;

//----------------------------------------------------------------------------
svtkMultiBlockDataSet* svtkMultiBlockDataSet::GetData(svtkInformation* info)
{
  return info ? svtkMultiBlockDataSet::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkMultiBlockDataSet* svtkMultiBlockDataSet::GetData(svtkInformationVector* v, int i)
{
  return svtkMultiBlockDataSet::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
void svtkMultiBlockDataSet::SetNumberOfBlocks(unsigned int numBlocks)
{
  this->Superclass::SetNumberOfChildren(numBlocks);
}

//----------------------------------------------------------------------------
unsigned int svtkMultiBlockDataSet::GetNumberOfBlocks()
{
  return this->Superclass::GetNumberOfChildren();
}

//----------------------------------------------------------------------------
svtkDataObject* svtkMultiBlockDataSet::GetBlock(unsigned int blockno)
{
  return this->Superclass::GetChild(blockno);
}

//----------------------------------------------------------------------------
void svtkMultiBlockDataSet::SetBlock(unsigned int blockno, svtkDataObject* block)
{
  if (block && block->IsA("svtkCompositeDataSet") && !block->IsA("svtkMultiBlockDataSet") &&
    !block->IsA("svtkMultiPieceDataSet") && !block->IsA("svtkPartitionedDataSet"))
  {
    svtkErrorMacro(<< block->GetClassName() << " cannot be added as a block.");
    return;
  }
  this->Superclass::SetChild(blockno, block);
}

//----------------------------------------------------------------------------
void svtkMultiBlockDataSet::RemoveBlock(unsigned int blockno)
{
  this->Superclass::RemoveChild(blockno);
}

//----------------------------------------------------------------------------
void svtkMultiBlockDataSet::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
