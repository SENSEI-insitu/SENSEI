/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTreeIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkTreeIterator.h"

#include "svtkTree.h"

svtkTreeIterator::svtkTreeIterator()
{
  this->Tree = nullptr;
  this->StartVertex = -1;
  this->NextId = -1;
}

svtkTreeIterator::~svtkTreeIterator()
{
  if (this->Tree)
  {
    this->Tree->Delete();
    this->Tree = nullptr;
  }
}

void svtkTreeIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Tree: " << this->Tree << endl;
  os << indent << "StartVertex: " << this->StartVertex << endl;
  os << indent << "NextId: " << this->NextId << endl;
}

void svtkTreeIterator::SetTree(svtkTree* tree)
{
  svtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting Tree to " << tree);
  if (this->Tree != tree)
  {
    svtkTree* temp = this->Tree;
    this->Tree = tree;
    if (this->Tree != nullptr)
    {
      this->Tree->Register(this);
    }
    if (temp != nullptr)
    {
      temp->UnRegister(this);
    }
    this->StartVertex = -1;
    this->Initialize();
    this->Modified();
  }
}

void svtkTreeIterator::SetStartVertex(svtkIdType vertex)
{
  if (this->StartVertex != vertex)
  {
    this->StartVertex = vertex;
    this->Initialize();
    this->Modified();
  }
}

svtkIdType svtkTreeIterator::Next()
{
  svtkIdType last = this->NextId;
  if (last != -1)
  {
    this->NextId = this->NextInternal();
  }
  return last;
}

bool svtkTreeIterator::HasNext()
{
  return this->NextId != -1;
}

void svtkTreeIterator::Restart()
{
  this->Initialize();
}
