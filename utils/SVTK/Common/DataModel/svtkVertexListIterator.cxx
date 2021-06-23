/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVertexListIterator.cxx

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

#include "svtkVertexListIterator.h"

#include "svtkDataObject.h"
#include "svtkDistributedGraphHelper.h"
#include "svtkGraph.h"
#include "svtkInformation.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkVertexListIterator);
//----------------------------------------------------------------------------
svtkVertexListIterator::svtkVertexListIterator()
{
  this->Current = 0;
  this->End = 0;
  this->Graph = nullptr;
}

//----------------------------------------------------------------------------
svtkVertexListIterator::~svtkVertexListIterator()
{
  if (this->Graph)
  {
    this->Graph->Delete();
  }
}

//----------------------------------------------------------------------------
void svtkVertexListIterator::SetGraph(svtkGraph* graph)
{
  svtkSetObjectBodyMacro(Graph, svtkGraph, graph);
  if (this->Graph)
  {
    this->Current = 0;
    this->End = this->Graph->GetNumberOfVertices();

    // For a distributed graph, shift the iteration space to cover
    // local vertices
    svtkDistributedGraphHelper* helper = this->Graph->GetDistributedGraphHelper();
    if (helper)
    {
      int myRank = this->Graph->GetInformation()->Get(svtkDataObject::DATA_PIECE_NUMBER());
      this->Current = helper->MakeDistributedId(myRank, this->Current);
      this->End = helper->MakeDistributedId(myRank, this->End);
    }
  }
}

//----------------------------------------------------------------------------
void svtkVertexListIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Graph: " << (this->Graph ? "" : "(null)") << endl;
  if (this->Graph)
  {
    this->Graph->PrintSelf(os, indent.GetNextIndent());
  }
}
