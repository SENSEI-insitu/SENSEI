/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPolygon.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPolygon
 * @brief   a cell that represents an n-sided polygon
 *
 * svtkPolygon is a concrete implementation of svtkCell to represent a 2D
 * n-sided polygon. The polygons cannot have any internal holes, and cannot
 * self-intersect. Define the polygon with n-points ordered in the counter-
 * clockwise direction; do not repeat the last point.
 */

#ifndef svtkPolygon_h
#define svtkPolygon_h

#include "svtkCell.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkDoubleArray;
class svtkIdTypeArray;
class svtkLine;
class svtkPoints;
class svtkQuad;
class svtkTriangle;
class svtkIncrementalPointLocator;

class SVTKCOMMONDATAMODEL_EXPORT svtkPolygon : public svtkCell
{
public:
  static svtkPolygon* New();
  svtkTypeMacro(svtkPolygon, svtkCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * See the svtkCell API for descriptions of these methods.
   */
  int GetCellType() override { return SVTK_POLYGON; }
  int GetCellDimension() override { return 2; }
  int GetNumberOfEdges() override { return this->GetNumberOfPoints(); }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int) override { return nullptr; }
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;
  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* tris, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;
  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  int IsPrimaryCell() override { return 0; }
  //@}

  /**
   * Compute the area of a polygon. This is a convenience function
   * which simply calls static double ComputeArea(svtkPoints *p,
   * svtkIdType numPts, svtkIdType *pts, double normal[3]);
   * with the appropriate parameters from the instantiated svtkPolygon.
   */
  double ComputeArea();

  /**
   * Compute the interpolation functions/derivatives.
   * (aka shape functions/derivatives)
   * Two interpolation algorithms are available: 1/r^2 and Mean Value
   * Coordinate. The former is used by default. To use the second algorithm,
   * set UseMVCInterpolation to be true.
   * The function assumes the input point lies on the polygon plane without
   * checking that.
   */
  void InterpolateFunctions(const double x[3], double* sf) override;

  //@{
  /**
   * Computes the unit normal to the polygon. If pts=nullptr, point indexing is
   * assumed to be {0, 1, ..., numPts-1}.
   */
  static void ComputeNormal(svtkPoints* p, int numPts, const svtkIdType* pts, double n[3]);
  static void ComputeNormal(svtkPoints* p, double n[3]);
  static void ComputeNormal(svtkIdTypeArray* ids, svtkPoints* pts, double n[3]);
  //@}

  /**
   * Compute the polygon normal from an array of points. This version assumes
   * that the polygon is convex, and looks for the first valid normal.
   */
  static void ComputeNormal(int numPts, double* pts, double n[3]);

  /**
   * Determine whether or not a polygon is convex. This is a convenience
   * function that simply calls static bool IsConvex(int numPts,
   * svtkIdType *pts, svtkPoints *p) with the appropriate parameters from the
   * instantiated svtkPolygon.
   */
  bool IsConvex();

  //@{
  /**
   * Determine whether or not a polygon is convex. If pts=nullptr, point indexing
   * is assumed to be {0, 1, ..., numPts-1}.
   */
  static bool IsConvex(svtkPoints* p, int numPts, svtkIdType* pts);
  static bool IsConvex(svtkIdTypeArray* ids, svtkPoints* p);
  static bool IsConvex(svtkPoints* p);
  //@}

  //@{
  /**
   * Compute the centroid of a set of points. Returns false if the computation
   * is invalid (this occurs when numPts=0 or when ids is empty).
   */
  static bool ComputeCentroid(svtkPoints* p, int numPts, const svtkIdType* pts, double centroid[3]);
  static bool ComputeCentroid(svtkIdTypeArray* ids, svtkPoints* pts, double centroid[3]);
  //@}

  /**
   * Compute the area of a polygon in 3D. The area is returned, as well as
   * the normal (a side effect of using this method). If you desire to
   * compute the area of a triangle, use svtkTriangleArea which is faster.
   * If pts==nullptr, point indexing is supposed to be {0, 1, ..., numPts-1}.
   * If you already have a svtkPolygon instantiated, a convenience function,
   * ComputeArea() is provided.
   */
  static double ComputeArea(svtkPoints* p, svtkIdType numPts, const svtkIdType* pts, double normal[3]);

  /**
   * Create a local s-t coordinate system for a polygon. The point p0 is
   * the origin of the local system, p10 is s-axis vector, and p20 is the
   * t-axis vector. (These are expressed in the modeling coordinate system and
   * are vectors of dimension [3].) The values l20 and l20 are the lengths of
   * the vectors p10 and p20, and n is the polygon normal.
   */
  int ParameterizePolygon(
    double p0[3], double p10[3], double& l10, double p20[3], double& l20, double n[3]);

  /**
   * Determine whether point is inside polygon. Function uses ray-casting
   * to determine if point is inside polygon. Works for arbitrary polygon shape
   * (e.g., non-convex). Returns 0 if point is not in polygon; 1 if it is.
   * Can also return -1 to indicate degenerate polygon.
   */
  static int PointInPolygon(double x[3], int numPts, double* pts, double bounds[6], double n[3]);

  /**
   * Triangulate this polygon. The user must provide the svtkIdList outTris.
   * On output, the outTris list contains the ids of the points defining
   * the triangulation. The ids are ordered into groups of three: each
   * three-group defines one triangle.
   */
  int Triangulate(svtkIdList* outTris);

  /**
   * Same as Triangulate(svtkIdList *outTris)
   * but with a first pass to split the polygon into non-degenerate polygons.
   */
  int NonDegenerateTriangulate(svtkIdList* outTris);

  /**
   * Triangulate polygon and enforce that the ratio of the smallest triangle
   * area to the polygon area is greater than a user-defined tolerance. The user
   * must provide the svtkIdList outTris. On output, the outTris list contains
   * the ids of the points defining the triangulation. The ids are ordered into
   * groups of three: each three-group defines one triangle.
   */
  int BoundedTriangulate(svtkIdList* outTris, double tol);

  /**
   * Compute the distance of a point to a polygon. The closest point on
   * the polygon is also returned. The bounds should be provided to
   * accelerate the computation.
   */
  static double DistanceToPolygon(
    double x[3], int numPts, double* pts, double bounds[6], double closest[3]);

  /**
   * Method intersects two polygons. You must supply the number of points and
   * point coordinates (npts, *pts) and the bounding box (bounds) of the two
   * polygons. Also supply a tolerance squared for controlling
   * error. The method returns 1 if there is an intersection, and 0 if
   * not. A single point of intersection x[3] is also returned if there
   * is an intersection.
   */
  static int IntersectPolygonWithPolygon(int npts, double* pts, double bounds[6], int npts2,
    double* pts2, double bounds2[6], double tol, double x[3]);

  /**
   * Intersect two convex 2D polygons to produce a line segment as output.
   * The return status of the methods indicated no intersection (returns 0);
   * a single point of intersection (returns 1); or a line segment (i.e., two
   * points of intersection, returns 2). The points of intersection are
   * returned in the arrays p0 and p1.  If less than two points of
   * intersection are generated then p1 and/or p0 may be
   * indeterminiate. Finally, if the two convex polygons are parallel, then
   * "0" is returned (i.e., no intersection) even if the triangles lie on one
   * another.
   */
  static int IntersectConvex2DCells(
    svtkCell* cell1, svtkCell* cell2, double tol, double p0[3], double p1[3]);

  //@{
  /**
   * Set/Get the flag indicating whether to use Mean Value Coordinate for the
   * interpolation. If true, InterpolateFunctions() uses the Mean Value
   * Coordinate to compute weights. Otherwise, the conventional 1/r^2 method
   * is used. The UseMVCInterpolation parameter is set to false by default.
   */
  svtkGetMacro(UseMVCInterpolation, bool);
  svtkSetMacro(UseMVCInterpolation, bool);
  //@}

protected:
  svtkPolygon();
  ~svtkPolygon() override;

  // Compute the interpolation functions using Mean Value Coordinate.
  void InterpolateFunctionsUsingMVC(const double x[3], double* weights);

  // variables used by instances of this class
  double Tolerance;            // Intersection tolerance
  int SuccessfulTriangulation; // Stops recursive tri. if necessary
  double Normal[3];            // polygon normal
  svtkIdList* Tris;
  svtkTriangle* Triangle;
  svtkQuad* Quad;
  svtkDoubleArray* TriScalars;
  svtkLine* Line;

  // Parameter indicating whether to use Mean Value Coordinate algorithm
  // for interpolation. The parameter is false by default.
  bool UseMVCInterpolation;

  // Helper methods for triangulation------------------------------
  /**
   * A fast triangulation method. Uses recursive divide and
   * conquer based on plane splitting to reduce loop into triangles.
   * The cell (e.g., triangle) is presumed properly initialized (i.e.,
   * Points and PointIds).
   */
  int EarCutTriangulation();

  /**
   * A fast triangulation method. Uses recursive divide and
   * conquer based on plane splitting to reduce loop into triangles.
   * The cell (e.g., triangle) is presumed properly initialized (i.e.,
   * Points and PointIds). Unlike EarCutTriangulation(), vertices are visited
   * sequentially without preference to angle.
   */
  int UnbiasedEarCutTriangulation(int seed);

private:
  svtkPolygon(const svtkPolygon&) = delete;
  void operator=(const svtkPolygon&) = delete;
};

#endif
