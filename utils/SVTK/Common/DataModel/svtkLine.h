/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLine.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkLine
 * @brief   cell represents a 1D line
 *
 * svtkLine is a concrete implementation of svtkCell to represent a 1D line.
 */

#ifndef svtkLine_h
#define svtkLine_h

#include "svtkCell.h"
#include "svtkCommonDataModelModule.h" // For export macro
class svtkIncrementalPointLocator;

class SVTKCOMMONDATAMODEL_EXPORT svtkLine : public svtkCell
{
public:
  static svtkLine* New();
  svtkTypeMacro(svtkLine, svtkCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * See the svtkCell API for descriptions of these methods.
   */
  int GetCellType() override { return SVTK_LINE; }
  int GetCellDimension() override { return 1; }
  int GetNumberOfEdges() override { return 0; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int) override { return nullptr; }
  svtkCell* GetFace(int) override { return nullptr; }
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;
  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;
  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  double* GetParametricCoords() override;
  //@}

  /**
   * Clip this line using scalar value provided. Like contouring, except
   * that it cuts the line to produce other lines.
   */
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* lines, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;

  /**
   * Return the center of the triangle in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  /**
   * Line-line intersection. Intersection has to occur within [0,1] parametric
   * coordinates and with specified tolerance.
   */
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;

  /**
   * Performs intersection of the projection of two finite 3D lines onto a 2D
   * plane. An intersection is found if the projection of the two lines onto
   * the plane perpendicular to the cross product of the two lines intersect.
   * The parameters (u,v) are the parametric coordinates of the lines at the
   * position of closest approach.
   */
  static int Intersection(const double p1[3], const double p2[3], const double x1[3],
    const double x2[3], double& u, double& v);

  /**
   * Performs intersection of two finite 3D lines. An intersection is found if
   * the projection of the two lines onto the plane perpendicular to the cross
   * product of the two lines intersect, and if the distance between the
   * closest points of approach are within a relative tolerance. The parameters
   * (u,v) are the parametric coordinates of the lines at the position of
   * closest approach.

   * NOTE: "Unlike Intersection(), which determines whether the projections of
   * two lines onto a plane intersect, Intersection3D() determines whether the
   * lines themselves in 3D space intersect, within a tolerance.
   */
  static int Intersection3D(
    double p1[3], double p2[3], double x1[3], double x2[3], double& u, double& v);

  /**
   * Compute the distance of a point x to a finite line (p1,p2). The method
   * computes the parametric coordinate t and the point location on the
   * line. Note that t is unconstrained (i.e., it may lie outside the range
   * [0,1]) but the closest point will lie within the finite line [p1,p2], if
   * it is defined. Also, the method returns the distance squared between x and
   * the line (p1,p2).
   */
  static double DistanceToLine(const double x[3], const double p1[3], const double p2[3], double& t,
    double closestPoint[3] = nullptr);

  /**
   * Determine the distance of the current vertex to the edge defined by
   * the vertices provided.  Returns distance squared. Note: line is assumed
   * infinite in extent.
   */
  static double DistanceToLine(const double x[3], const double p1[3], const double p2[3]);

  /**
   * Computes the shortest distance squared between two infinite lines, each
   * defined by a pair of points (l0,l1) and (m0,m1).
   * Upon return, the closest points on the two line segments will be stored
   * in closestPt1 and closestPt2. Their parametric coords
   * (-inf <= t0, t1 <= inf) will be stored in t0 and t1. The return value is
   * the shortest distance squared between the two line-segments.
   */
  static double DistanceBetweenLines(double l0[3], double l1[3], double m0[3], double m1[3],
    double closestPt1[3], double closestPt2[3], double& t1, double& t2);

  /**
   * Computes the shortest distance squared between two finite line segments
   * defined by their end points (l0,l1) and (m0,m1).
   * Upon return, the closest points on the two line segments will be stored
   * in closestPt1 and closestPt2. Their parametric coords (0 <= t0, t1 <= 1)
   * will be stored in t0 and t1. The return value is the shortest distance
   * squared between the two line-segments.
   */
  static double DistanceBetweenLineSegments(double l0[3], double l1[3], double m0[3], double m1[3],
    double closestPt1[3], double closestPt2[3], double& t1, double& t2);

  static void InterpolationFunctions(const double pcoords[3], double weights[2]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[2]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[2]) override
  {
    svtkLine::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[2]) override
  {
    svtkLine::InterpolationDerivs(pcoords, derivs);
  }
  //@}

protected:
  svtkLine();
  ~svtkLine() override {}

private:
  svtkLine(const svtkLine&) = delete;
  void operator=(const svtkLine&) = delete;
};

//----------------------------------------------------------------------------
inline int svtkLine::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = 0.5;
  pcoords[1] = pcoords[2] = 0.0;
  return 0;
}

#endif
