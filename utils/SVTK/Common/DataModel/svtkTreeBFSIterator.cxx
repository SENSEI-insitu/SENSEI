/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTreeBFSIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkTreeBFSIterator.h"

#include "svtkIntArray.h"
#include "svtkObjectFactory.h"
#include "svtkTree.h"

#include <queue>
using std::queue;

class svtkTreeBFSIteratorInternals
{
public:
  queue<svtkIdType> Queue;
};

svtkStandardNewMacro(svtkTreeBFSIterator);

svtkTreeBFSIterator::svtkTreeBFSIterator()
{
  this->Internals = new svtkTreeBFSIteratorInternals();
  this->Color = svtkIntArray::New();
}

svtkTreeBFSIterator::~svtkTreeBFSIterator()
{
  delete this->Internals;
  this->Internals = nullptr;

  if (this->Color)
  {
    this->Color->Delete();
    this->Color = nullptr;
  }
}

void svtkTreeBFSIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

void svtkTreeBFSIterator::Initialize()
{
  if (this->Tree == nullptr)
  {
    return;
  }
  // Set all colors to white
  this->Color->Resize(this->Tree->GetNumberOfVertices());
  for (svtkIdType i = 0; i < this->Tree->GetNumberOfVertices(); i++)
  {
    this->Color->SetValue(i, this->WHITE);
  }
  if (this->StartVertex < 0)
  {
    this->StartVertex = this->Tree->GetRoot();
  }
  while (!this->Internals->Queue.empty())
  {
    this->Internals->Queue.pop();
  }

  // Find the first item
  if (this->Tree->GetNumberOfVertices() > 0)
  {
    this->NextId = this->NextInternal();
  }
  else
  {
    this->NextId = -1;
  }
}

svtkIdType svtkTreeBFSIterator::NextInternal()
{
  if (this->Color->GetValue(this->StartVertex) == this->WHITE)
  {
    this->Color->SetValue(this->StartVertex, this->GRAY);
    this->Internals->Queue.push(this->StartVertex);
  }

  while (!this->Internals->Queue.empty())
  {
    svtkIdType currentId = this->Internals->Queue.front();
    this->Internals->Queue.pop();

    for (svtkIdType childNum = 0; childNum < this->Tree->GetNumberOfChildren(currentId); childNum++)
    {
      svtkIdType childId = this->Tree->GetChild(currentId, childNum);
      if (this->Color->GetValue(childId) == this->WHITE)
      {
        // Found a white vertex; make it gray, add it to the queue
        this->Color->SetValue(childId, this->GRAY);
        this->Internals->Queue.push(childId);
      }
    }

    this->Color->SetValue(currentId, this->BLACK);
    return currentId;
  }
  return -1;
}
