/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPolygonBuilder.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPolygonBuilder
 *
 *
 *  The polygon output is the boundary of the union of the triangles.
 *  It is assumed that the input triangles form a simple polygon. It is
 *  currently used to compute polygons for slicing.
 *
 */

#ifndef svtkPolygonBuilder_h
#define svtkPolygonBuilder_h

#include "svtkCommonMiscModule.h" // For export macro
#include "svtkIdList.h"
#include "svtkObject.h"
#include "svtkType.h" //for basic types
#include <cstddef>   //for size_t
#include <map>       //for private data members
#include <utility>   //for private data members
#include <vector>    // for private data members

class svtkIdListCollection;

class SVTKCOMMONMISC_EXPORT svtkPolygonBuilder
{
public:
  svtkPolygonBuilder();

  /**
   * Insert a triangle as a triplet of point IDs.
   */
  void InsertTriangle(const svtkIdType* abc);

  /**
   * Populate polys with lists of polygons, defined as sequential external
   * vertices. It is the responsibility of the user to delete these generated
   * lists in order to avoid memory leaks.
   */
  void GetPolygons(svtkIdListCollection* polys);

  /**
   * Prepare the builder for a new set of inputs.
   */
  void Reset();

private:
  typedef std::pair<svtkIdType, svtkIdType> Edge;
  typedef std::map<Edge, size_t> EdgeHistogram;
  typedef std::multimap<svtkIdType, svtkIdType> EdgeMap;
  typedef std::vector<svtkIdType> Triangle;
  typedef std::vector<Triangle> Triangles;
  typedef std::map<svtkIdType, Triangles> TriangleMap;

  TriangleMap Tris;

  EdgeHistogram EdgeCounter;
  EdgeMap Edges;
};

#endif
// SVTK-HeaderTest-Exclude: svtkPolygonBuilder.h
