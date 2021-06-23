/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGraphInternals.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*----------------------------------------------------------------------------
 Copyright (c) Sandia Corporation
 See Copyright.txt or http://www.paraview.org/HTML/Copyright.html for details.
----------------------------------------------------------------------------*/
#include "svtkGraphInternals.h"

#include "svtkDistributedGraphHelper.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkGraphInternals);

//----------------------------------------------------------------------------
svtkGraphInternals::svtkGraphInternals()
{
  this->NumberOfEdges = 0;
  this->LastRemoteEdgeId = -1;
  this->UsingPedigreeIds = false;
}

//----------------------------------------------------------------------------
svtkGraphInternals::~svtkGraphInternals() = default;

//----------------------------------------------------------------------------
void svtkGraphInternals::RemoveEdgeFromOutList(svtkIdType e, std::vector<svtkOutEdgeType>& outEdges)
{
  size_t outSize = outEdges.size();
  size_t i = 0;
  for (; i < outSize; ++i)
  {
    if (outEdges[i].Id == e)
    {
      break;
    }
  }
  if (i == outSize)
  {
    svtkErrorMacro("Could not find edge in source edge list.");
    return;
  }
  outEdges[i] = outEdges[outSize - 1];
  outEdges.pop_back();
}

//----------------------------------------------------------------------------
void svtkGraphInternals::RemoveEdgeFromInList(svtkIdType e, std::vector<svtkInEdgeType>& inEdges)
{
  size_t inSize = inEdges.size();
  size_t i = 0;
  for (; i < inSize; ++i)
  {
    if (inEdges[i].Id == e)
    {
      break;
    }
  }
  if (i == inSize)
  {
    svtkErrorMacro("Could not find edge in source edge list.");
    return;
  }
  inEdges[i] = inEdges[inSize - 1];
  inEdges.pop_back();
}

//----------------------------------------------------------------------------
void svtkGraphInternals::ReplaceEdgeFromOutList(
  svtkIdType from, svtkIdType to, std::vector<svtkOutEdgeType>& outEdges)
{
  size_t outSize = outEdges.size();
  for (size_t i = 0; i < outSize; ++i)
  {
    if (outEdges[i].Id == from)
    {
      outEdges[i].Id = to;
    }
  }
}

//----------------------------------------------------------------------------
void svtkGraphInternals::ReplaceEdgeFromInList(
  svtkIdType from, svtkIdType to, std::vector<svtkInEdgeType>& inEdges)
{
  size_t inSize = inEdges.size();
  for (size_t i = 0; i < inSize; ++i)
  {
    if (inEdges[i].Id == from)
    {
      inEdges[i].Id = to;
    }
  }
}
