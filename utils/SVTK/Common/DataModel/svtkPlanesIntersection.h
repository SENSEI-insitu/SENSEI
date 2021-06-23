/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPlanesIntersection.h

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
 * @class   svtkPlanesIntersection
 * @brief   A svtkPlanesIntersection object is a
 *    svtkPlanes object that can compute whether the arbitrary convex region
 *    bounded by it's planes intersects an axis-aligned box.
 *
 *
 *    A subclass of svtkPlanes, this class determines whether it
 *    intersects an axis aligned box.   This is motivated by the
 *    need to intersect the axis aligned region of a spacial
 *    decomposition of volume data with various other regions.
 *    It uses the algorithm from Graphics Gems IV, page 81.
 *
 * @par Caveat:
 *    An instance of svtkPlanes can be redefined by changing the planes,
 *    but this subclass then will not know if the region vertices are
 *    up to date.  (Region vertices can be specified in SetRegionVertices
 *    or computed by the subclass.)  So Delete and recreate if you want
 *    to change the set of planes.
 *
 */

#ifndef svtkPlanesIntersection_h
#define svtkPlanesIntersection_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkPlanes.h"

class svtkPoints;
class svtkPointsProjectedHull;
class svtkCell;

class SVTKCOMMONDATAMODEL_EXPORT svtkPlanesIntersection : public svtkPlanes
{
  svtkTypeMacro(svtkPlanesIntersection, svtkPlanes);

public:
  void PrintSelf(ostream& os, svtkIndent indent) override;

  static svtkPlanesIntersection* New();

  /**
   * It helps if you know the vertices of the convex region.
   * If you don't, we will calculate them.  Region vertices
   * are 3-tuples.
   */

  void SetRegionVertices(svtkPoints* pts);
  void SetRegionVertices(double* v, int nvertices);
  int GetNumberOfRegionVertices();
  // Retained for backward compatibility
  int GetNumRegionVertices() { return this->GetNumberOfRegionVertices(); }
  int GetRegionVertices(double* v, int nvertices);

  /**
   * Return 1 if the axis aligned box defined by R intersects
   * the region defined by the planes, or 0 otherwise.
   */

  int IntersectsRegion(svtkPoints* R);

  /**
   * A convenience function provided by this class, returns
   * 1 if the polygon defined in pts intersects the bounding
   * box defined in bounds, 0 otherwise.

   * The points must define a planar polygon.
   */

  static int PolygonIntersectsBBox(double bounds[6], svtkPoints* pts);

  /**
   * Another convenience function provided by this class, returns
   * the svtkPlanesIntersection object representing a 3D
   * cell.  The point IDs for each face must be given in
   * counter-clockwise order from the outside of the cell.
   */

  static svtkPlanesIntersection* Convert3DCell(svtkCell* cell);

protected:
  static void ComputeNormal(double* p1, double* p2, double* p3, double normal[3]);
  static double EvaluatePlaneEquation(double* x, double* p);
  static void PlaneEquation(double* n, double* x, double* p);
  static int GoodNormal(double* n);
  static int Invert3x3(double M[3][3]);

  svtkPlanesIntersection();
  ~svtkPlanesIntersection() override;

private:
  int IntersectsBoundingBox(svtkPoints* R);
  int EnclosesBoundingBox(svtkPoints* R);
  int EvaluateFacePlane(int plane, svtkPoints* R);
  int IntersectsProjection(svtkPoints* R, int direction);

  void SetPlaneEquations();
  void ComputeRegionVertices();

  void planesMatrix(int p1, int p2, int p3, double M[3][3]) const;
  int duplicate(double testv[3]) const;
  void planesRHS(int p1, int p2, int p3, double r[3]) const;
  int outsideRegion(double v[3]);

  // plane equations
  double* Planes;

  // vertices of convex regions enclosed by the planes, also
  //    the ccw hull of that region projected in 3 orthog. directions
  svtkPointsProjectedHull* RegionPts;

  svtkPlanesIntersection(const svtkPlanesIntersection&) = delete;
  void operator=(const svtkPlanesIntersection&) = delete;
};
#endif
