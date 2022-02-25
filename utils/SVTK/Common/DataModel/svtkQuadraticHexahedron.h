/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadraticHexahedron.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkQuadraticHexahedron
 * @brief   cell represents a parabolic, 20-node isoparametric hexahedron
 *
 * svtkQuadraticHexahedron is a concrete implementation of svtkNonLinearCell to
 * represent a three-dimensional, 20-node isoparametric parabolic
 * hexahedron. The interpolation is the standard finite element, quadratic
 * isoparametric shape function. The cell includes a mid-edge node. The
 * ordering of the twenty points defining the cell is point ids (0-7,8-19)
 * where point ids 0-7 are the eight corner vertices of the cube; followed by
 * twelve midedge nodes (8-19). Note that these midedge nodes correspond lie
 * on the edges defined by (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7),
 * (7,4), (0,4), (1,5), (2,6), (3,7).
 *
 * @sa
 * svtkQuadraticEdge svtkQuadraticTriangle svtkQuadraticTetra
 * svtkQuadraticQuad svtkQuadraticPyramid svtkQuadraticWedge
 */

#ifndef svtkQuadraticHexahedron_h
#define svtkQuadraticHexahedron_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNonLinearCell.h"

class svtkQuadraticEdge;
class svtkQuadraticQuad;
class svtkHexahedron;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkQuadraticHexahedron : public svtkNonLinearCell
{
public:
  static svtkQuadraticHexahedron* New();
  svtkTypeMacro(svtkQuadraticHexahedron, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Implement the svtkCell API. See the svtkCell API for descriptions
   * of these methods.
   */
  int GetCellType() override { return SVTK_QUADRATIC_HEXAHEDRON; }
  int GetCellDimension() override { return 3; }
  int GetNumberOfEdges() override { return 12; }
  int GetNumberOfFaces() override { return 6; }
  svtkCell* GetEdge(int) override;
  svtkCell* GetFace(int) override;
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

  static void InterpolationFunctions(const double pcoords[3], double weights[20]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[60]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[20]) override
  {
    svtkQuadraticHexahedron::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[60]) override
  {
    svtkQuadraticHexahedron::InterpolationDerivs(pcoords, derivs);
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
  void JacobianInverse(const double pcoords[3], double** inverse, double derivs[60]);

protected:
  svtkQuadraticHexahedron();
  ~svtkQuadraticHexahedron() override;

  svtkQuadraticEdge* Edge;
  svtkQuadraticQuad* Face;
  svtkHexahedron* Hex;
  svtkPointData* PointData;
  svtkCellData* CellData;
  svtkDoubleArray* CellScalars;
  svtkDoubleArray* Scalars;

  void Subdivide(
    svtkPointData* inPd, svtkCellData* inCd, svtkIdType cellId, svtkDataArray* cellScalars);

private:
  svtkQuadraticHexahedron(const svtkQuadraticHexahedron&) = delete;
  void operator=(const svtkQuadraticHexahedron&) = delete;
};

#endif
