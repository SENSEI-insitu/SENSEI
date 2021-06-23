/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadraticWedge.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkQuadraticWedge
 * @brief   cell represents a parabolic, 15-node isoparametric wedge
 *
 * svtkQuadraticWedge is a concrete implementation of svtkNonLinearCell to
 * represent a three-dimensional, 15-node isoparametric parabolic
 * wedge. The interpolation is the standard finite element, quadratic
 * isoparametric shape function. The cell includes a mid-edge node. The
 * ordering of the fifteen points defining the cell is point ids (0-5,6-14)
 * where point ids 0-5 are the six corner vertices of the wedge, defined
 * analogously to the six points in svtkWedge (points (0,1,2) form the base of
 * the wedge which, using the right hand rule, forms a triangle whose normal
 * points away from the triangular face (3,4,5)); followed by nine midedge
 * nodes (6-14). Note that these midedge nodes correspond lie on the edges
 * defined by (0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (0,3), (1,4), (2,5).
 *
 * @sa
 * svtkQuadraticEdge svtkQuadraticTriangle svtkQuadraticTetra
 * svtkQuadraticHexahedron svtkQuadraticQuad svtkQuadraticPyramid
 */

#ifndef svtkQuadraticWedge_h
#define svtkQuadraticWedge_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNonLinearCell.h"

class svtkQuadraticEdge;
class svtkQuadraticQuad;
class svtkQuadraticTriangle;
class svtkWedge;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkQuadraticWedge : public svtkNonLinearCell
{
public:
  static svtkQuadraticWedge* New();
  svtkTypeMacro(svtkQuadraticWedge, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Implement the svtkCell API. See the svtkCell API for descriptions
   * of these methods.
   */
  int GetCellType() override { return SVTK_QUADRATIC_WEDGE; }
  int GetCellDimension() override { return 3; }
  int GetNumberOfEdges() override { return 9; }
  int GetNumberOfFaces() override { return 5; }
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int faceId) override;
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
   * Clip this quadratic hexahedron using scalar value provided. Like
   * contouring, except that it cuts the hex to produce linear
   * tetrahedron.
   */
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* tetras, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;

  /**
   * Line-edge intersection. Intersection has to occur within [0,1] parametric
   * coordinates and with specified tolerance.
   */
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;

  /**
   * Return the center of the quadratic wedge in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  static void InterpolationFunctions(const double pcoords[3], double weights[15]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[45]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[15]) override
  {
    svtkQuadraticWedge::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[45]) override
  {
    svtkQuadraticWedge::InterpolationDerivs(pcoords, derivs);
  }
  //@}
  //@{
  /**
   * Return the ids of the vertices defining edge/face (`edgeId`/`faceId').
   * Ids are related to the cell, not to the dataset.
   *
   * @note The return type changed. It used to be int*, it is now const svtkIdType*.
   * This is so ids are unified between svtkCell and svtkPoints.
   */
  static const svtkIdType* GetEdgeArray(svtkIdType edgeId);
  static const svtkIdType* GetFaceArray(svtkIdType faceId);
  //@}

  /**
   * Given parametric coordinates compute inverse Jacobian transformation
   * matrix. Returns 9 elements of 3x3 inverse Jacobian plus interpolation
   * function derivatives.
   */
  void JacobianInverse(const double pcoords[3], double** inverse, double derivs[45]);

protected:
  svtkQuadraticWedge();
  ~svtkQuadraticWedge() override;

  svtkQuadraticEdge* Edge;
  svtkQuadraticTriangle* TriangleFace;
  svtkQuadraticQuad* Face;
  svtkWedge* Wedge;
  svtkPointData* PointData;
  svtkCellData* CellData;
  svtkDoubleArray* CellScalars;
  svtkDoubleArray* Scalars; // used to avoid New/Delete in contouring/clipping

  void Subdivide(
    svtkPointData* inPd, svtkCellData* inCd, svtkIdType cellId, svtkDataArray* cellScalars);

private:
  svtkQuadraticWedge(const svtkQuadraticWedge&) = delete;
  void operator=(const svtkQuadraticWedge&) = delete;
};
//----------------------------------------------------------------------------
// Return the center of the quadratic wedge in parametric coordinates.
inline int svtkQuadraticWedge::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = 1. / 3;
  pcoords[2] = 0.5;
  return 0;
}

#endif
