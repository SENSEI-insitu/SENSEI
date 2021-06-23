/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVertexListIterator.h

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
/**
 * @class   svtkVertexListIterator
 * @brief   Iterates all vertices in a graph.
 *
 *
 * svtkVertexListIterator iterates through all vertices in a graph.
 * Create an instance of this and call graph->GetVertices(it) to initialize
 * this iterator. You may alternately call SetGraph() to initialize the
 * iterator.
 *
 * @sa
 * svtkGraph
 */

#ifndef svtkVertexListIterator_h
#define svtkVertexListIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

#include "svtkGraph.h" // For edge type definitions

class svtkGraphEdge;

class SVTKCOMMONDATAMODEL_EXPORT svtkVertexListIterator : public svtkObject
{
public:
  static svtkVertexListIterator* New();
  svtkTypeMacro(svtkVertexListIterator, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Setup the iterator with a graph.
   */
  virtual void SetGraph(svtkGraph* graph);

  //@{
  /**
   * Get the graph associated with this iterator.
   */
  svtkGetObjectMacro(Graph, svtkGraph);
  //@}

  //@{
  /**
   * Returns the next edge in the graph.
   */
  svtkIdType Next()
  {
    svtkIdType v = this->Current;
    ++this->Current;
    return v;
  }
  //@}

  /**
   * Whether this iterator has more edges.
   */
  bool HasNext() { return this->Current != this->End; }

protected:
  svtkVertexListIterator();
  ~svtkVertexListIterator() override;

  svtkGraph* Graph;
  svtkIdType Current;
  svtkIdType End;

private:
  svtkVertexListIterator(const svtkVertexListIterator&) = delete;
  void operator=(const svtkVertexListIterator&) = delete;
};

#endif
