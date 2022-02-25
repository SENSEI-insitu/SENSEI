/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadraticTetra.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkQuadraticTetra
 * @brief   cell represents a parabolic, 10-node isoparametric tetrahedron
 *
 * svtkQuadraticTetra is a concrete implementation of svtkNonLinearCell to
 * represent a three-dimensional, 10-node, isoparametric parabolic
 * tetrahedron. The interpolation is the standard finite element, quadratic
 * isoparametric shape function. The cell includes a mid-edge node on each of
 * the size edges of the tetrahedron. The ordering of the ten points defining
 * the cell is point ids (0-3,4-9) where ids 0-3 are the four tetra
 * vertices; and point ids 4-9 are the midedge nodes between (0,1), (1,2),
 * (2,0), (0,3), (1,3), and (2,3).
 *
 * Note that this class uses an internal linear tessellation for some internal operations
 * (e.g., clipping and contouring). This means that some artifacts may appear trying to
 * represent a non-linear interpolation function with linear tets.
 *
 * @sa
 * svtkQuadraticEdge svtkQuadraticTriangle svtkQuadraticWedge
 * svtkQuadraticQuad svtkQuadraticHexahedron svtkQuadraticPyramid
 */

#ifndef svtkQuadraticTetra_h
#define svtkQuadraticTetra_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNonLinearCell.h"

class svtkQuadraticEdge;
class svtkQuadraticTriangle;
class svtkTetra;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkQuadraticTetra : public svtkNonLinearCell
{
public:
  static svtkQuadraticTetra* New();
  svtkTypeMacro(svtkQuadraticTetra, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Implement the svtkCell API. See the svtkCell API for descriptions
   * of these methods.
   */
  int GetCellType() override { return SVTK_QUADRATIC_TETRA; }
  int GetCellDimension() override { return 3; }
  int GetNumberOfEdges() override { return 6; }
  int GetNumberOfFaces() override { return 4; }
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
   * Clip this edge using scalar value provided. Like contouring, except
   * that it cuts the tetra to produce new tetras.
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
   * Return the center of the quadratic tetra in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  /**
   * Return the distance of the parametric coordinate provided to the
   * cell. If inside the cell, a distance of zero is returned.
   */
  double GetParametricDistance(const double pcoords[3]) override;

  static void InterpolationFunctions(const double pcoords[3], double weights[10]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[30]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[10]) override
  {
    svtkQuadraticTetra::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[30]) override
  {
    svtkQuadraticTetra::InterpolationDerivs(pcoords, derivs);
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
  void JacobianInverse(const double pcoords[3], double** inverse, double derivs[30]);

protected:
  svtkQuadraticTetra();
  ~svtkQuadraticTetra() override;

  svtkQuadraticEdge* Edge;
  svtkQuadraticTriangle* Face;
  svtkTetra* Tetra;
  svtkDoubleArray* Scalars; // used to avoid New/Delete in contouring/clipping

private:
  svtkQuadraticTetra(const svtkQuadraticTetra&) = delete;
  void operator=(const svtkQuadraticTetra&) = delete;
};

#endif
