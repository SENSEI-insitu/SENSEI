/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVertex.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkVertex
 * @brief   a cell that represents a 3D point
 *
 * svtkVertex is a concrete implementation of svtkCell to represent a 3D point.
 */

#ifndef svtkVertex_h
#define svtkVertex_h

#include "svtkCell.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkIncrementalPointLocator;

class SVTKCOMMONDATAMODEL_EXPORT svtkVertex : public svtkCell
{
public:
  static svtkVertex* New();
  svtkTypeMacro(svtkVertex, svtkCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Make a new svtkVertex object with the same information as this object.
   */

  //@{
  /**
   * See the svtkCell API for descriptions of these methods.
   */
  int GetCellType() override { return SVTK_VERTEX; }
  int GetCellDimension() override { return 0; }
  int GetNumberOfEdges() override { return 0; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int) override { return nullptr; }
  svtkCell* GetFace(int) override { return nullptr; }
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* pts, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
    svtkCellData* outCd, int insideOut) override;
  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  double* GetParametricCoords() override;
  //@}

  /**
   * Given parametric coordinates of a point, return the closest cell
   * boundary, and whether the point is inside or outside of the cell. The
   * cell boundary is defined by a list of points (pts) that specify a vertex
   * (1D cell).  If the return value of the method is != 0, then the point is
   * inside the cell.
   */
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;

  /**
   * Generate contouring primitives. The scalar list cellScalars are
   * scalar values at each cell point. The point locator is essentially a
   * points list that merges points as they are inserted (i.e., prevents
   * duplicates).
   */
  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts1, svtkCellArray* lines, svtkCellArray* verts2, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;

  /**
   * Return the center of the triangle in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  /**
   * Intersect with a ray. Return parametric coordinates (both line and cell)
   * and global intersection coordinates, given ray definition and tolerance.
   * The method returns non-zero value if intersection occurs.
   */
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;

  /**
   * Triangulate the vertex. This method fills pts and ptIds with information
   * from the only point in the vertex.
   */
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;

  /**
   * Get the derivative of the vertex. Returns (0.0, 0.0, 0.0) for all
   * dimensions.
   */
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;

  static void InterpolationFunctions(const double pcoords[3], double weights[1]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[3]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[1]) override
  {
    svtkVertex::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[3]) override
  {
    svtkVertex::InterpolationDerivs(pcoords, derivs);
  }
  //@}

protected:
  svtkVertex();
  ~svtkVertex() override {}

private:
  svtkVertex(const svtkVertex&) = delete;
  void operator=(const svtkVertex&) = delete;
};

//----------------------------------------------------------------------------
inline int svtkVertex::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = pcoords[2] = 0.0;
  return 0;
}

#endif
