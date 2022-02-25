/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTriQuadraticHexahedron.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkTriQuadraticHexahedron
 * @brief   cell represents a parabolic, 27-node isoparametric hexahedron
 *
 * svtkTriQuadraticHexahedron is a concrete implementation of svtkNonLinearCell to
 * represent a three-dimensional, 27-node isoparametric triquadratic
 * hexahedron. The interpolation is the standard finite element, triquadratic
 * isoparametric shape function. The cell includes 8 edge nodes, 12 mid-edge nodes,
 * 6 mid-face nodes and one mid-volume node. The ordering of the 27 points defining the
 * cell is point ids (0-7,8-19, 20-25, 26)
 * where point ids 0-7 are the eight corner vertices of the cube; followed by
 * twelve midedge nodes (8-19); followed by 6 mid-face nodes (20-25) and the last node (26)
 * is the mid-volume node. Note that these midedge nodes correspond lie
 * on the edges defined by (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7),
 * (7,4), (0,4), (1,5), (2,6), (3,7). The mid-surface nodes lies on the faces
 * defined by (first edge nodes id's, than mid-edge nodes id's):
 * (0,1,5,4;8,17,12,16), (1,2,6,5;9,18,13,17), (2,3,7,6,10,19,14,18),
 * (3,0,4,7;11,16,15,19), (0,1,2,3;8,9,10,11), (4,5,6,7;12,13,14,15).
 * The last point lies in the center of the cell (0,1,2,3,4,5,6,7).
 *
 * \verbatim
 *
 * top
 *  7--14--6
 *  |      |
 * 15  25  13
 *  |      |
 *  4--12--5
 *
 *  middle
 * 19--23--18
 *  |      |
 * 20  26  21
 *  |      |
 * 16--22--17
 *
 * bottom
 *  3--10--2
 *  |      |
 * 11  24  9
 *  |      |
 *  0-- 8--1
 *
 * \endverbatim
 *
 *
 * @sa
 * svtkQuadraticEdge svtkQuadraticTriangle svtkQuadraticTetra
 * svtkQuadraticQuad svtkQuadraticPyramid svtkQuadraticWedge
 * svtkBiQuadraticQuad
 *
 * @par Thanks:
 * Thanks to Soeren Gebbert who developed this class and
 * integrated it into SVTK 5.0.
 */

#ifndef svtkTriQuadraticHexahedron_h
#define svtkTriQuadraticHexahedron_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNonLinearCell.h"

class svtkQuadraticEdge;
class svtkBiQuadraticQuad;
class svtkHexahedron;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkTriQuadraticHexahedron : public svtkNonLinearCell
{
public:
  static svtkTriQuadraticHexahedron* New();
  svtkTypeMacro(svtkTriQuadraticHexahedron, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Implement the svtkCell API. See the svtkCell API for descriptions
   * of these methods.
   */
  int GetCellType() override { return SVTK_TRIQUADRATIC_HEXAHEDRON; }
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
  int EvaluatePosition(const double x[3], double* closestPoint, int& subId, double pcoords[3],
    double& dist2, double* weights) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  double* GetParametricCoords() override;

  /**
   * Clip this triquadratic hexahedron using scalar value provided. Like
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

  static void InterpolationFunctions(const double pcoords[3], double weights[27]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[81]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[27]) override
  {
    svtkTriQuadraticHexahedron::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[81]) override
  {
    svtkTriQuadraticHexahedron::InterpolationDerivs(pcoords, derivs);
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
  void JacobianInverse(const double pcoords[3], double** inverse, double derivs[81]);

protected:
  svtkTriQuadraticHexahedron();
  ~svtkTriQuadraticHexahedron() override;

  svtkQuadraticEdge* Edge;
  svtkBiQuadraticQuad* Face;
  svtkHexahedron* Hex;
  svtkDoubleArray* Scalars;

private:
  svtkTriQuadraticHexahedron(const svtkTriQuadraticHexahedron&) = delete;
  void operator=(const svtkTriQuadraticHexahedron&) = delete;
};

#endif
