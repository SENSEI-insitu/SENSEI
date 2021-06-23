/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadraticTriangle.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkQuadraticTriangle
 * @brief   cell represents a parabolic, isoparametric triangle
 *
 * svtkQuadraticTriangle is a concrete implementation of svtkNonLinearCell to
 * represent a two-dimensional, 6-node, isoparametric parabolic triangle. The
 * interpolation is the standard finite element, quadratic isoparametric
 * shape function. The cell includes three mid-edge nodes besides the three
 * triangle vertices. The ordering of the three points defining the cell is
 * point ids (0-2,3-5) where id #3 is the midedge node between points
 * (0,1); id #4 is the midedge node between points (1,2); and id #5 is the
 * midedge node between points (2,0).
 *
 * @sa
 * svtkQuadraticEdge svtkQuadraticTetra svtkQuadraticPyramid
 * svtkQuadraticQuad svtkQuadraticHexahedron svtkQuadraticWedge
 */

#ifndef svtkQuadraticTriangle_h
#define svtkQuadraticTriangle_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNonLinearCell.h"

class svtkQuadraticEdge;
class svtkTriangle;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkQuadraticTriangle : public svtkNonLinearCell
{
public:
  static svtkQuadraticTriangle* New();
  svtkTypeMacro(svtkQuadraticTriangle, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Implement the svtkCell API. See the svtkCell API for descriptions
   * of these methods.
   */
  int GetCellType() override { return SVTK_QUADRATIC_TRIANGLE; }
  int GetCellDimension() override { return 2; }
  int GetNumberOfEdges() override { return 3; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int) override { return nullptr; }
  //@}

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

  /**
   * Clip this quadratic triangle using scalar value provided. Like
   * contouring, except that it cuts the triangle to produce linear
   * triangles.
   */
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;

  /**
   * Line-edge intersection. Intersection has to occur within [0,1] parametric
   * coordinates and with specified tolerance.
   */
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;

  /**
   * Return the center of the quadratic triangle in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  /**
   * Return the distance of the parametric coordinate provided to the
   * cell. If inside the cell, a distance of zero is returned.
   */
  double GetParametricDistance(const double pcoords[3]) override;

  static void InterpolationFunctions(const double pcoords[3], double weights[6]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[12]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[6]) override
  {
    svtkQuadraticTriangle::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[12]) override
  {
    svtkQuadraticTriangle::InterpolationDerivs(pcoords, derivs);
  }
  //@}

protected:
  svtkQuadraticTriangle();
  ~svtkQuadraticTriangle() override;

  svtkQuadraticEdge* Edge;
  svtkTriangle* Face;
  svtkDoubleArray* Scalars; // used to avoid New/Delete in contouring/clipping

private:
  svtkQuadraticTriangle(const svtkQuadraticTriangle&) = delete;
  void operator=(const svtkQuadraticTriangle&) = delete;
};
//----------------------------------------------------------------------------
inline int svtkQuadraticTriangle::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = 1. / 3;
  pcoords[2] = 0.0;
  return 0;
}

#endif
