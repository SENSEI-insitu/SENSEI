/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadraticPyramid.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkQuadraticPyramid
 * @brief   cell represents a parabolic, 13-node isoparametric pyramid
 *
 * svtkQuadraticPyramid is a concrete implementation of svtkNonLinearCell to
 * represent a three-dimensional, 13-node isoparametric parabolic
 * pyramid. The interpolation is the standard finite element, quadratic
 * isoparametric shape function. The cell includes a mid-edge node. The
 * ordering of the thirteen points defining the cell is point ids (0-4,5-12)
 * where point ids 0-4 are the five corner vertices of the pyramid; followed
 * by eight midedge nodes (5-12). Note that these midedge nodes lie
 * on the edges defined by (0,1), (1,2), (2,3), (3,0), (0,4), (1,4), (2,4),
 * (3,4), respectively. The parametric location of vertex #4 is [0, 0, 1].
 *
 * @sa
 * svtkQuadraticEdge svtkQuadraticTriangle svtkQuadraticTetra
 * svtkQuadraticHexahedron svtkQuadraticQuad svtkQuadraticWedge
 *
 * @par Thanks:
 * The shape functions and derivatives could be implemented thanks to
 * the report Pyramid Solid Elements Linear and Quadratic Iso-P Models
 * From Center For Aerospace Structures
 */

#ifndef svtkQuadraticPyramid_h
#define svtkQuadraticPyramid_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNonLinearCell.h"

class svtkQuadraticEdge;
class svtkQuadraticQuad;
class svtkQuadraticTriangle;
class svtkTetra;
class svtkPyramid;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkQuadraticPyramid : public svtkNonLinearCell
{
public:
  static svtkQuadraticPyramid* New();
  svtkTypeMacro(svtkQuadraticPyramid, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Implement the svtkCell API. See the svtkCell API for descriptions
   * of these methods.
   */
  int GetCellType() override { return SVTK_QUADRATIC_PYRAMID; }
  int GetCellDimension() override { return 3; }
  int GetNumberOfEdges() override { return 8; }
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
   * Clip this quadratic triangle using scalar value provided. Like
   * contouring, except that it cuts the triangle to produce linear
   * triangles.
   */
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* tets, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;

  /**
   * Line-edge intersection. Intersection has to occur within [0,1] parametric
   * coordinates and with specified tolerance.
   */
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;

  /**
   * Return the center of the quadratic pyramid in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  static void InterpolationFunctions(const double pcoords[3], double weights[13]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[39]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[13]) override
  {
    svtkQuadraticPyramid::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[39]) override
  {
    svtkQuadraticPyramid::InterpolationDerivs(pcoords, derivs);
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
  void JacobianInverse(const double pcoords[3], double** inverse, double derivs[39]);

protected:
  svtkQuadraticPyramid();
  ~svtkQuadraticPyramid() override;

  svtkQuadraticEdge* Edge;
  svtkQuadraticTriangle* TriangleFace;
  svtkQuadraticQuad* Face;
  svtkTetra* Tetra;
  svtkPyramid* Pyramid;
  svtkPointData* PointData;
  svtkCellData* CellData;
  svtkDoubleArray* CellScalars;
  svtkDoubleArray* Scalars; // used to avoid New/Delete in contouring/clipping

  //@{
  /**
   * This method adds in a point at the center of the quadrilateral face
   * and then interpolates values to that point. In order to do this it
   * also resizes certain member variable arrays. For safety should call
   * ResizeArrays() after the results of Subdivide() are not needed anymore.
   **/
  void Subdivide(
    svtkPointData* inPd, svtkCellData* inCd, svtkIdType cellId, svtkDataArray* cellScalars);
  //@}
  //@{
  /**
   * Resize the superclasses' member arrays to newSize where newSize should either be
   * 13 or 14. Call with 13 to reset the reallocation done in the Subdivide()
   * method or call with 14 to add one extra tuple for the generated point in
   * Subdivice. For efficiency it only resizes the superclasses' arrays.
   **/
  void ResizeArrays(svtkIdType newSize);
  //@}

private:
  svtkQuadraticPyramid(const svtkQuadraticPyramid&) = delete;
  void operator=(const svtkQuadraticPyramid&) = delete;
};
//----------------------------------------------------------------------------
// Return the center of the quadratic pyramid in parametric coordinates.
//
inline int svtkQuadraticPyramid::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = 6.0 / 13.0;
  pcoords[2] = 3.0 / 13.0;
  return 0;
}

#endif
