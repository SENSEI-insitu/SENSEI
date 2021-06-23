/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCompositeDataIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkCompositeDataIterator.h"
#include "svtkCompositeDataSet.h"
#include "svtkObjectFactory.h"

//----------------------------------------------------------------------------
svtkCompositeDataIterator::svtkCompositeDataIterator()
{
  this->Reverse = 0;
  this->SkipEmptyNodes = 1;
  this->DataSet = nullptr;
}

//----------------------------------------------------------------------------
svtkCompositeDataIterator::~svtkCompositeDataIterator()
{
  this->SetDataSet(nullptr);
}

//----------------------------------------------------------------------------
void svtkCompositeDataIterator::SetDataSet(svtkCompositeDataSet* ds)
{
  svtkSetObjectBodyMacro(DataSet, svtkCompositeDataSet, ds);
  if (ds)
  {
    this->GoToFirstItem();
  }
}

//----------------------------------------------------------------------------
void svtkCompositeDataIterator::InitTraversal()
{
  this->SetReverse(0);
  this->GoToFirstItem();
}

//----------------------------------------------------------------------------
void svtkCompositeDataIterator::InitReverseTraversal()
{
  this->SetReverse(1);
  this->GoToFirstItem();
}

//----------------------------------------------------------------------------
void svtkCompositeDataIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Reverse: " << (this->Reverse ? "On" : "Off") << endl;
  os << indent << "SkipEmptyNodes: " << (this->SkipEmptyNodes ? "On" : "Off") << endl;
}
