/*=========================================================================
  Program:   Visualization Toolkit
  Module:    svtkHierarchicalBoxDataSet.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkHierarchicalBoxDataSet.h"
#include "svtkHierarchicalBoxDataIterator.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkHierarchicalBoxDataSet);

//-----------------------------------------------------------------------------
svtkHierarchicalBoxDataSet::svtkHierarchicalBoxDataSet() = default;

//-----------------------------------------------------------------------------
svtkHierarchicalBoxDataSet::~svtkHierarchicalBoxDataSet() = default;

//-----------------------------------------------------------------------------
void svtkHierarchicalBoxDataSet::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
svtkCompositeDataIterator* svtkHierarchicalBoxDataSet::NewIterator()
{
  svtkCompositeDataIterator* iter = svtkHierarchicalBoxDataIterator::New();
  iter->SetDataSet(this);
  return iter;
}

//-----------------------------------------------------------------------------
svtkHierarchicalBoxDataSet* svtkHierarchicalBoxDataSet::GetData(svtkInformation* info)
{
  return info ? svtkHierarchicalBoxDataSet::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkHierarchicalBoxDataSet* svtkHierarchicalBoxDataSet::GetData(svtkInformationVector* v, int i)
{
  return svtkHierarchicalBoxDataSet::GetData(v->GetInformationObject(i));
}
