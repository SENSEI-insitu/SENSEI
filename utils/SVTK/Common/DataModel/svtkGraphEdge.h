/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGraphEdge.h

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
 * @class   svtkGraphEdge
 * @brief   Representation of a single graph edge.
 *
 *
 * A heavy-weight (svtkObject subclass) graph edge object that may be used
 * instead of the svtkEdgeType struct, for use with wrappers.
 * The edge contains the source and target vertex ids, and the edge id.
 *
 * @sa
 * svtkGraph
 */

#ifndef svtkGraphEdge_h
#define svtkGraphEdge_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkGraphEdge : public svtkObject
{
public:
  static svtkGraphEdge* New();
  svtkTypeMacro(svtkGraphEdge, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * The source of the edge.
   */
  svtkSetMacro(Source, svtkIdType);
  svtkGetMacro(Source, svtkIdType);
  //@}

  //@{
  /**
   * The target of the edge.
   */
  svtkSetMacro(Target, svtkIdType);
  svtkGetMacro(Target, svtkIdType);
  //@}

  //@{
  /**
   * The id of the edge.
   */
  svtkSetMacro(Id, svtkIdType);
  svtkGetMacro(Id, svtkIdType);
  //@}

protected:
  svtkGraphEdge();
  ~svtkGraphEdge() override;

  svtkIdType Source;
  svtkIdType Target;
  svtkIdType Id;

private:
  svtkGraphEdge(const svtkGraphEdge&) = delete;
  void operator=(const svtkGraphEdge&) = delete;
};

#endif
