/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadraticLinearQuad.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkQuadraticLinearQuad
 * @brief   cell represents a quadratic-linear, 6-node isoparametric quad
 *
 * svtkQuadraticQuad is a concrete implementation of svtkNonLinearCell to
 * represent a two-dimensional, 6-node isoparametric quadratic-linear quadrilateral
 * element. The interpolation is the standard finite element, quadratic-linear
 * isoparametric shape function. The cell includes a mid-edge node for two
 * of the four edges. The ordering of the six points defining
 * the cell are point ids (0-3,4-5) where ids 0-3 define the four corner
 * vertices of the quad; ids 4-7 define the midedge nodes (0,1) and (2,3) .
 *
 * @sa
 * svtkQuadraticEdge svtkQuadraticTriangle svtkQuadraticTetra svtkQuadraticQuad
 * svtkQuadraticHexahedron svtkQuadraticWedge svtkQuadraticPyramid
 *
 * @par Thanks:
 * Thanks to Soeren Gebbert  who developed this class and
 * integrated it into SVTK 5.0.
 */

#ifndef svtkQuadraticLinearQuad_h
#define svtkQuadraticLinearQuad_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNonLinearCell.h"

class svtkQuadraticEdge;
class svtkLine;
class svtkQuad;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkQuadraticLinearQuad : public svtkNonLinearCell
{
public:
  static svtkQuadraticLinearQuad* New();
  svtkTypeMacro(svtkQuadraticLinearQuad, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Implement the svtkCell API. See the svtkCell API for descriptions
   * of these methods.
   */
  int GetCellType() override { return SVTK_QUADRATIC_LINEAR_QUAD; }
  int GetCellDimension() override { return 2; }
  int GetNumberOfEdges() override { return 4; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int) override;
  svtkCell* GetFace(int) override { return nullptr; }
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
   * Clip this quadratic linear quad using scalar value provided. Like
   * contouring, except that it cuts the quad to produce linear triangles.
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
   * Return the center of the pyramid in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  static void InterpolationFunctions(const double pcoords[3], double weights[6]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[12]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[6]) override
  {
    svtkQuadraticLinearQuad::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[12]) override
  {
    svtkQuadraticLinearQuad::InterpolationDerivs(pcoords, derivs);
  }
  //@}
  /**
   * Return the ids of the vertices defining edge (`edgeId`).
   * Ids are related to the cell, not to the dataset.
   *
   * @note The return type changed. It used to be int*, it is now const svtkIdType*.
   * This is so ids are unified between svtkCell and svtkPoints.
   *
   * @note The return type changed. It used to be int*, it is now const svtkIdType*.
   * This is so ids are unified between svtkCell and svtkPoints.
   */
  static int* GetEdgeArray(svtkIdType edgeId);

protected:
  svtkQuadraticLinearQuad();
  ~svtkQuadraticLinearQuad() override;

  svtkQuadraticEdge* Edge;
  svtkLine* LinEdge;
  svtkQuad* Quad;
  svtkDoubleArray* Scalars;

private:
  svtkQuadraticLinearQuad(const svtkQuadraticLinearQuad&) = delete;
  void operator=(const svtkQuadraticLinearQuad&) = delete;
};
//----------------------------------------------------------------------------
inline int svtkQuadraticLinearQuad::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = 0.5;
  pcoords[2] = 0.;
  return 0;
}

#endif
