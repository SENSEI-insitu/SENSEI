/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadraticEdge.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkQuadraticEdge
 * @brief   cell represents a parabolic, isoparametric edge
 *
 * svtkQuadraticEdge is a concrete implementation of svtkNonLinearCell to
 * represent a one-dimensional, 3-nodes, isoparametric parabolic line. The
 * interpolation is the standard finite element, quadratic isoparametric
 * shape function. The cell includes a mid-edge node. The ordering of the
 * three points defining the cell is point ids (0,1,2) where id #2 is the
 * midedge node.
 *
 * @sa
 * svtkQuadraticTriangle svtkQuadraticTetra svtkQuadraticWedge
 * svtkQuadraticQuad svtkQuadraticHexahedron svtkQuadraticPyramid
 */

#ifndef svtkQuadraticEdge_h
#define svtkQuadraticEdge_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNonLinearCell.h"

class svtkLine;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkQuadraticEdge : public svtkNonLinearCell
{
public:
  static svtkQuadraticEdge* New();
  svtkTypeMacro(svtkQuadraticEdge, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Implement the svtkCell API. See the svtkCell API for descriptions
   * of these methods.
   */
  int GetCellType() override { return SVTK_QUADRATIC_EDGE; }
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

  /**
   * Clip this edge using scalar value provided. Like contouring, except
   * that it cuts the edge to produce linear line segments.
   */
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* lines, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;

  /**
   * Line-edge intersection. Intersection has to occur within [0,1] parametric
   * coordinates and with specified tolerance.
   */
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;

  /**
   * Return the center of the quadratic tetra in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  static void InterpolationFunctions(const double pcoords[3], double weights[3]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[3]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[3]) override
  {
    svtkQuadraticEdge::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[3]) override
  {
    svtkQuadraticEdge::InterpolationDerivs(pcoords, derivs);
  }
  //@}

protected:
  svtkQuadraticEdge();
  ~svtkQuadraticEdge() override;

  svtkLine* Line;
  svtkDoubleArray* Scalars; // used to avoid New/Delete in contouring/clipping

private:
  svtkQuadraticEdge(const svtkQuadraticEdge&) = delete;
  void operator=(const svtkQuadraticEdge&) = delete;
};
//----------------------------------------------------------------------------
inline int svtkQuadraticEdge::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = 0.5;
  pcoords[1] = pcoords[2] = 0.;
  return 0;
}

#endif
