/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGraphInternals.h

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
/**
 * @class   svtkGraphInternals
 * @brief   Internal representation of svtkGraph
 *
 *
 * This is the internal representation of svtkGraph, used only in rare cases
 * where one must modify that representation.
 */

#ifndef svtkGraphInternals_h
#define svtkGraphInternals_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkGraph.h"

#include <map>    // STL Header
#include <vector> // STL Header

//----------------------------------------------------------------------------
// class svtkVertexAdjacencyList
//----------------------------------------------------------------------------

class svtkVertexAdjacencyList
{
public:
  std::vector<svtkInEdgeType> InEdges;
  std::vector<svtkOutEdgeType> OutEdges;
};

//----------------------------------------------------------------------------
// class svtkGraphInternals
//----------------------------------------------------------------------------
class SVTKCOMMONDATAMODEL_EXPORT svtkGraphInternals : public svtkObject
{
public:
  static svtkGraphInternals* New();

  svtkTypeMacro(svtkGraphInternals, svtkObject);
  std::vector<svtkVertexAdjacencyList> Adjacency;

  svtkIdType NumberOfEdges;

  svtkIdType LastRemoteEdgeId;
  svtkIdType LastRemoteEdgeSource;
  svtkIdType LastRemoteEdgeTarget;

  // Whether we have used pedigree IDs to refer to the vertices of the
  // graph, e.g., to add edges or vertices. In a distributed graph,
  // the pedigree-id interface is mutually exclusive with the
  // no-argument AddVertex() function in svtkMutableUndirectedGraph and
  // svtkMutableDirectedGraph.
  bool UsingPedigreeIds;

  /**
   * Convenience method for removing an edge from an out edge list.
   */
  void RemoveEdgeFromOutList(svtkIdType e, std::vector<svtkOutEdgeType>& outEdges);

  /**
   * Convenience method for removing an edge from an in edge list.
   */
  void RemoveEdgeFromInList(svtkIdType e, std::vector<svtkInEdgeType>& inEdges);

  /**
   * Convenience method for renaming an edge in an out edge list.
   */
  void ReplaceEdgeFromOutList(svtkIdType from, svtkIdType to, std::vector<svtkOutEdgeType>& outEdges);

  /**
   * Convenience method for renaming an edge in an in edge list.
   */
  void ReplaceEdgeFromInList(svtkIdType from, svtkIdType to, std::vector<svtkInEdgeType>& inEdges);

protected:
  svtkGraphInternals();
  ~svtkGraphInternals() override;

private:
  svtkGraphInternals(const svtkGraphInternals&) = delete;
  void operator=(const svtkGraphInternals&) = delete;
};

#endif // svtkGraphInternals_h

// SVTK-HeaderTest-Exclude: svtkGraphInternals.h
