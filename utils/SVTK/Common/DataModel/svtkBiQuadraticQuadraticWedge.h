/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBiQuadraticQuadraticWedge.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkBiQuadraticQuadraticWedge
 * @brief   cell represents a parabolic, 18-node isoparametric wedge
 *
 * svtkBiQuadraticQuadraticWedge is a concrete implementation of svtkNonLinearCell to
 * represent a three-dimensional, 18-node isoparametric biquadratic
 * wedge. The interpolation is the standard finite element,
 * biquadratic-quadratic isoparametric shape function plus the linear functions.
 * The cell includes a mid-edge node. The
 * ordering of the 18 points defining the cell is point ids (0-5,6-15, 16-18)
 * where point ids 0-5 are the six corner vertices of the wedge; followed by
 * nine midedge nodes (6-15) and 3 center-face nodes. Note that these midedge
 * nodes correspond lie
 * on the edges defined by (0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (0,3),
 * (1,4), (2,5), and the center-face nodes are laying in quads 16-(0,1,4,3),
 * 17-(1,2,5,4) and (2,0,3,5).
 *
 * @sa
 * svtkQuadraticEdge svtkQuadraticTriangle svtkQuadraticTetra
 * svtkQuadraticHexahedron svtkQuadraticQuad svtkQuadraticPyramid
 *
 * @par Thanks:
 * Thanks to Soeren Gebbert who developed this class and
 * integrated it into SVTK 5.0.
 */

#ifndef svtkBiQuadraticQuadraticWedge_h
#define svtkBiQuadraticQuadraticWedge_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNonLinearCell.h"

class svtkQuadraticEdge;
class svtkBiQuadraticQuad;
class svtkQuadraticTriangle;
class svtkWedge;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkBiQuadraticQuadraticWedge : public svtkNonLinearCell
{
public:
  static svtkBiQuadraticQuadraticWedge* New();
  svtkTypeMacro(svtkBiQuadraticQuadraticWedge, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Implement the svtkCell API. See the svtkCell API for descriptions
   * of these methods.
   */
  int GetCellType() override { return SVTK_BIQUADRATIC_QUADRATIC_WEDGE; }
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
  int EvaluatePosition(const double x[3], double* closestPoint, int& subId, double pcoords[3],
    double& dist2, double* weights) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  double* GetParametricCoords() override;

  /**
   * Clip this quadratic Wedge using scalar value provided. Like
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
    svtkBiQuadraticQuadraticWedge::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[45]) override
  {
    svtkBiQuadraticQuadraticWedge::InterpolationDerivs(pcoords, derivs);
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
  svtkBiQuadraticQuadraticWedge();
  ~svtkBiQuadraticQuadraticWedge() override;

  svtkQuadraticEdge* Edge;
  svtkQuadraticTriangle* TriangleFace;
  svtkBiQuadraticQuad* Face;
  svtkWedge* Wedge;
  svtkDoubleArray* Scalars; // used to avoid New/Delete in contouring/clipping

private:
  svtkBiQuadraticQuadraticWedge(const svtkBiQuadraticQuadraticWedge&) = delete;
  void operator=(const svtkBiQuadraticQuadraticWedge&) = delete;
};
//----------------------------------------------------------------------------
// Return the center of the quadratic wedge in parametric coordinates.
inline int svtkBiQuadraticQuadraticWedge::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = 1. / 3;
  pcoords[2] = 0.5;
  return 0;
}

#endif
