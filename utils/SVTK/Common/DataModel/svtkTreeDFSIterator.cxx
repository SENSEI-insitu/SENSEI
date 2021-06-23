/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTreeDFSIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------*/

#include "svtkTreeDFSIterator.h"

#include "svtkIntArray.h"
#include "svtkObjectFactory.h"
#include "svtkTree.h"

#include <stack>
using std::stack;

struct svtkTreeDFSIteratorPosition
{
  svtkTreeDFSIteratorPosition(svtkIdType vertex, svtkIdType index)
    : Vertex(vertex)
    , Index(index)
  {
  }
  svtkIdType Vertex;
  svtkIdType Index; // How far along we are in the vertex's edge array
};

class svtkTreeDFSIteratorInternals
{
public:
  stack<svtkTreeDFSIteratorPosition> Stack;
};

svtkStandardNewMacro(svtkTreeDFSIterator);

svtkTreeDFSIterator::svtkTreeDFSIterator()
{
  this->Internals = new svtkTreeDFSIteratorInternals();
  this->Color = svtkIntArray::New();
  this->Mode = 0;
  this->CurRoot = -1;
}

svtkTreeDFSIterator::~svtkTreeDFSIterator()
{
  delete this->Internals;
  this->Internals = nullptr;

  if (this->Color)
  {
    this->Color->Delete();
    this->Color = nullptr;
  }
}

void svtkTreeDFSIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Mode: " << this->Mode << endl;
  os << indent << "CurRoot: " << this->CurRoot << endl;
}

void svtkTreeDFSIterator::Initialize()
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
  this->CurRoot = this->StartVertex;
  while (!this->Internals->Stack.empty())
  {
    this->Internals->Stack.pop();
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

void svtkTreeDFSIterator::SetMode(int mode)
{
  if (this->Mode != mode)
  {
    this->Mode = mode;
    this->Initialize();
    this->Modified();
  }
}

svtkIdType svtkTreeDFSIterator::NextInternal()
{
  while (this->Color->GetValue(this->StartVertex) != this->BLACK)
  {
    while (!this->Internals->Stack.empty())
    {
      // Pop the current position off the stack
      svtkTreeDFSIteratorPosition pos = this->Internals->Stack.top();
      this->Internals->Stack.pop();
      // cout << "popped " << pos.Vertex << "," << pos.Index << " off the stack" << endl;

      svtkIdType nchildren = this->Tree->GetNumberOfChildren(pos.Vertex);
      while (pos.Index < nchildren &&
        this->Color->GetValue(this->Tree->GetChild(pos.Vertex, pos.Index)) != this->WHITE)
      {
        pos.Index++;
      }
      if (pos.Index == nchildren)
      {
        // cout << "DFS coloring " << pos.Vertex << " black" << endl;
        // Done with this vertex; make it black and leave it off the stack
        this->Color->SetValue(pos.Vertex, this->BLACK);
        if (this->Mode == this->FINISH)
        {
          // cout << "DFS finished " << pos.Vertex << endl;
          return pos.Vertex;
        }
        // Done with the start vertex, so we are totally done!
        if (pos.Vertex == this->StartVertex)
        {
          return -1;
        }
      }
      else
      {
        // Not done with this vertex; put it back on the stack
        this->Internals->Stack.push(pos);

        // Found a white vertex; make it gray, add it to the stack
        svtkIdType found = this->Tree->GetChild(pos.Vertex, pos.Index);
        // cout << "DFS coloring " << found << " gray (adjacency)" << endl;
        this->Color->SetValue(found, this->GRAY);
        this->Internals->Stack.push(svtkTreeDFSIteratorPosition(found, 0));
        if (this->Mode == this->DISCOVER)
        {
          // cout << "DFS adjacent discovery " << found << endl;
          return found;
        }
      }
    }

    // Done with this component, so find a white vertex and start a new seedgeh
    if (this->Color->GetValue(this->StartVertex) != this->BLACK)
    {
      while (true)
      {
        if (this->Color->GetValue(this->CurRoot) == this->WHITE)
        {
          // Found a new component; make it gray, put it on the stack
          // cerr << "DFS coloring " << this->CurRoot << " gray (new component)" << endl;
          this->Internals->Stack.push(svtkTreeDFSIteratorPosition(this->CurRoot, 0));
          this->Color->SetValue(this->CurRoot, this->GRAY);
          if (this->Mode == this->DISCOVER)
          {
            // cerr << "DFS new component discovery " << this->CurRoot << endl;
            return this->CurRoot;
          }
          break;
        }
        else if (this->Color->GetValue(this->CurRoot) == this->GRAY)
        {
          svtkErrorMacro(
            "There should be no gray vertices in the graph when starting a new component.");
        }
        this->CurRoot = (this->CurRoot + 1) % this->Tree->GetNumberOfVertices();
      }
    }
  }
  // cout << "DFS no more!" << endl;
  return -1;
}
