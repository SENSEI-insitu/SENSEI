/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBiQuadraticQuadraticHexahedron.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkBiQuadraticQuadraticHexahedron
 * @brief   cell represents a biquadratic,
 * 24-node isoparametric hexahedron
 *
 * svtkBiQuadraticQuadraticHexahedron is a concrete implementation of svtkNonLinearCell to
 * represent a three-dimensional, 24-node isoparametric biquadratic
 * hexahedron. The interpolation is the standard finite element,
 * biquadratic-quadratic
 * isoparametric shape function. The cell includes mid-edge and center-face nodes. The
 * ordering of the 24 points defining the cell is point ids (0-7,8-19, 20-23)
 * where point ids 0-7 are the eight corner vertices of the cube; followed by
 * twelve midedge nodes (8-19), nodes 20-23 are the center-face nodes. Note that
 * these midedge nodes correspond lie
 * on the edges defined by (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7),
 * (7,4), (0,4), (1,5), (2,6), (3,7). The center face nodes laying in quad
 * 22-(0,1,5,4), 21-(1,2,6,5), 23-(2,3,7,6) and 22-(3,0,4,7)
 *
 * \verbatim
 *
 * top
 *  7--14--6
 *  |      |
 * 15      13
 *  |      |
 *  4--12--5
 *
 *  middle
 * 19--23--18
 *  |      |
 * 20      21
 *  |      |
 * 16--22--17
 *
 * bottom
 *  3--10--2
 *  |      |
 * 11      9
 *  |      |
 *  0-- 8--1
 *
 * \endverbatim
 *
 *
 * @sa
 * svtkQuadraticEdge svtkQuadraticTriangle svtkQuadraticTetra
 * svtkQuadraticQuad svtkQuadraticPyramid svtkQuadraticWedge
 *
 * @par Thanks:
 * Thanks to Soeren Gebbert  who developed this class and
 * integrated it into SVTK 5.0.
 */

#ifndef svtkBiQuadraticQuadraticHexahedron_h
#define svtkBiQuadraticQuadraticHexahedron_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNonLinearCell.h"

class svtkQuadraticEdge;
class svtkQuadraticQuad;
class svtkBiQuadraticQuad;
class svtkHexahedron;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkBiQuadraticQuadraticHexahedron : public svtkNonLinearCell
{
public:
  static svtkBiQuadraticQuadraticHexahedron* New();
  svtkTypeMacro(svtkBiQuadraticQuadraticHexahedron, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Implement the svtkCell API. See the svtkCell API for descriptions
   * of these methods.
   */
  int GetCellType() override { return SVTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON; }
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
   * Clip this biquadratic hexahedron using scalar value provided. Like
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

  static void InterpolationFunctions(const double pcoords[3], double weights[24]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[72]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[24]) override
  {
    svtkBiQuadraticQuadraticHexahedron::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[72]) override
  {
    svtkBiQuadraticQuadraticHexahedron::InterpolationDerivs(pcoords, derivs);
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
  void JacobianInverse(const double pcoords[3], double** inverse, double derivs[72]);

protected:
  svtkBiQuadraticQuadraticHexahedron();
  ~svtkBiQuadraticQuadraticHexahedron() override;

  svtkQuadraticEdge* Edge;
  svtkQuadraticQuad* Face;
  svtkBiQuadraticQuad* BiQuadFace;
  svtkHexahedron* Hex;
  svtkPointData* PointData;
  svtkCellData* CellData;
  svtkDoubleArray* CellScalars;
  svtkDoubleArray* Scalars;

  void Subdivide(
    svtkPointData* inPd, svtkCellData* inCd, svtkIdType cellId, svtkDataArray* cellScalars);

private:
  svtkBiQuadraticQuadraticHexahedron(const svtkBiQuadraticQuadraticHexahedron&) = delete;
  void operator=(const svtkBiQuadraticQuadraticHexahedron&) = delete;
};

#endif
