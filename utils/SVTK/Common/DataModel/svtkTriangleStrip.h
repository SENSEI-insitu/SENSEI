/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTriangleStrip.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkTriangleStrip
 * @brief   a cell that represents a triangle strip
 *
 * svtkTriangleStrip is a concrete implementation of svtkCell to represent a 2D
 * triangle strip. A triangle strip is a compact representation of triangles
 * connected edge to edge in strip fashion. The connectivity of a triangle
 * strip is three points defining an initial triangle, then for each
 * additional triangle, a single point that, combined with the previous two
 * points, defines the next triangle.
 */

#ifndef svtkTriangleStrip_h
#define svtkTriangleStrip_h

#include "svtkCell.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkLine;
class svtkTriangle;
class svtkIncrementalPointLocator;

class SVTKCOMMONDATAMODEL_EXPORT svtkTriangleStrip : public svtkCell
{
public:
  static svtkTriangleStrip* New();
  svtkTypeMacro(svtkTriangleStrip, svtkCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * See the svtkCell API for descriptions of these methods.
   */
  int GetCellType() override { return SVTK_TRIANGLE_STRIP; }
  int GetCellDimension() override { return 2; }
  int GetNumberOfEdges() override { return this->GetNumberOfPoints(); }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int svtkNotUsed(faceId)) override { return nullptr; }
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;
  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;
  //@}

  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  int IsPrimaryCell() override { return 0; }

  /**
   * Return the center of the point cloud in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  /**
   * Given a triangle strip, decompose it into a list of (triangle)
   * polygons. The polygons are appended to the end of the list of triangles.
   */
  static void DecomposeStrip(int npts, const svtkIdType* pts, svtkCellArray* tris);

protected:
  svtkTriangleStrip();
  ~svtkTriangleStrip() override;

  svtkLine* Line;
  svtkTriangle* Triangle;

private:
  svtkTriangleStrip(const svtkTriangleStrip&) = delete;
  void operator=(const svtkTriangleStrip&) = delete;
};

#endif
