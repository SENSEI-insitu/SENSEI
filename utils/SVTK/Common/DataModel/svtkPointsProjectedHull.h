/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPointsProjectedHull.h

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
 * @class   svtkPointsProjectedHull
 * @brief   the convex hull of the orthogonal
 *    projection of the svtkPoints in the 3 coordinate directions
 *
 *    a subclass of svtkPoints, it maintains the counter clockwise
 *    convex hull of the points (projected orthogonally in the
 *    three coordinate directions) and has a method to
 *    test for intersection of that hull with an axis aligned
 *    rectangle.  This is used for intersection tests of 3D volumes.
 */

#ifndef svtkPointsProjectedHull_h
#define svtkPointsProjectedHull_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkPoints.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkPointsProjectedHull : public svtkPoints
{
  svtkTypeMacro(svtkPointsProjectedHull, svtkPoints);

public:
  void PrintSelf(ostream& os, svtkIndent indent) override;

  static svtkPointsProjectedHull* New();

  /**
   * determine whether the resulting rectangle intersects the
   * convex hull of the projection of the points along that axis.
   */

  int RectangleIntersectionX(svtkPoints* R);

  /**
   * the convex hull of the projection of the points along the
   * positive X-axis.
   */

  int RectangleIntersectionX(float ymin, float ymax, float zmin, float zmax);
  int RectangleIntersectionX(double ymin, double ymax, double zmin, double zmax);

  /**
   * of the parallel projection along the Y axis of the points
   */

  int RectangleIntersectionY(svtkPoints* R);

  /**
   * the convex hull of the projection of the points along the
   * positive Y-axis.
   */

  int RectangleIntersectionY(float zmin, float zmax, float xmin, float xmax);
  int RectangleIntersectionY(double zmin, double zmax, double xmin, double xmax);

  /**
   * of the parallel projection along the Z axis of the points
   */

  int RectangleIntersectionZ(svtkPoints* R);

  /**
   * the convex hull of the projection of the points along the
   * positive Z-axis.
   */

  int RectangleIntersectionZ(float xmin, float xmax, float ymin, float ymax);
  int RectangleIntersectionZ(double xmin, double xmax, double ymin, double ymax);

  /**
   * Returns the coordinates (y,z) of the points in the convex hull
   * of the projection of the points down the positive x-axis.  pts has
   * storage for len*2 values.
   */

  int GetCCWHullX(float* pts, int len);
  int GetCCWHullX(double* pts, int len);

  /**
   * Returns the coordinates (z, x) of the points in the convex hull
   * of the projection of the points down the positive y-axis.  pts has
   * storage for len*2 values.
   */

  int GetCCWHullY(float* pts, int len);
  int GetCCWHullY(double* pts, int len);

  /**
   * Returns the coordinates (x, y) of the points in the convex hull
   * of the projection of the points down the positive z-axis.  pts has
   * storage for len*2 values.
   */

  int GetCCWHullZ(float* pts, int len);
  int GetCCWHullZ(double* pts, int len);

  /**
   * Returns the number of points in the convex hull of the projection
   * of the points down the positive x-axis
   */

  int GetSizeCCWHullX();

  /**
   * Returns the number of points in the convex hull of the projection
   * of the points down the positive y-axis
   */

  int GetSizeCCWHullY();

  /**
   * Returns the number of points in the convex hull of the projection
   * of the points down the positive z-axis
   */

  int GetSizeCCWHullZ();

  void Initialize() override;
  void Reset() override { this->Initialize(); }

  /**
   * Forces recalculation of convex hulls, use this if
   * you delete/add points
   */

  void Update();

protected:
  svtkPointsProjectedHull();
  ~svtkPointsProjectedHull() override;

private:
  int RectangleIntersection(double hmin, double hmax, double vmin, double vmax, int direction);
  int GrahamScanAlgorithm(int direction);
  void GetPoints();
  int RectangleBoundingBoxIntersection(
    double hmin, double hmax, double vmin, double vmax, int direction);
  int RectangleOutside(double hmin, double hmax, double vmin, double vmax, int direction);

  int RectangleOutside1DPolygon(double hmin, double hmax, double vmin, double vmax, int dir);

  void InitFlags();
  void ClearAllocations();

  static int RemoveExtras(double* pts, int n);
  static double Distance(double* p1, double* p2);
  static svtkIdType PositionInHull(double* base, double* top, double* pt);
  static int OutsideLine(
    double hmin, double hmax, double vmin, double vmax, double* p0, double* p1, double* insidePt);
  static int OutsideHorizontalLine(
    double vmin, double vmax, double* p0, double* p1, double* insidePt);
  static int OutsideVerticalLine(
    double hmin, double hmax, double* p0, double* p1, double* insidePt);

  double* Pts;
  svtkIdType Npts;
  svtkTimeStamp PtsTime;

  double* CCWHull[3];
  float HullBBox[3][4];
  int HullSize[3];
  svtkTimeStamp HullTime[3];

  svtkPointsProjectedHull(const svtkPointsProjectedHull&) = delete;
  void operator=(const svtkPointsProjectedHull&) = delete;
};
#endif
