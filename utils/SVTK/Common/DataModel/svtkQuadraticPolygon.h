/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadraticPolygon.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkQuadraticPolygon
 * @brief   a cell that represents a parabolic n-sided polygon
 *
 * svtkQuadraticPolygon is a concrete implementation of svtkNonLinearCell to
 * represent a 2D n-sided (2*n nodes) parabolic polygon. The polygon cannot
 * have any internal holes, and cannot self-intersect. The cell includes a
 * mid-edge node for each of the n edges of the cell. The ordering of the
 * 2*n points defining the cell are point ids (0..n-1 and n..2*n-1) where ids
 * 0..n-1 define the corner vertices of the polygon; ids n..2*n-1 define the
 * midedge nodes. Define the polygon with points ordered in the counter-
 * clockwise direction; do not repeat the last point.
 *
 * @sa
 * svtkQuadraticEdge svtkQuadraticTriangle svtkQuadraticTetra
 * svtkQuadraticHexahedron svtkQuadraticWedge svtkQuadraticPyramid
 */

#ifndef svtkQuadraticPolygon_h
#define svtkQuadraticPolygon_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNonLinearCell.h"

class svtkQuadraticEdge;
class svtkPolygon;
class svtkIdTypeArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkQuadraticPolygon : public svtkNonLinearCell
{
public:
  static svtkQuadraticPolygon* New();
  svtkTypeMacro(svtkQuadraticPolygon, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Implement the svtkCell API. See the svtkCell API for descriptions
   * of these methods.
   */
  int GetCellType() override { return SVTK_QUADRATIC_POLYGON; }
  int GetCellDimension() override { return 2; }
  int GetNumberOfEdges() override { return this->GetNumberOfPoints() / 2; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int) override;
  svtkCell* GetFace(int) override { return nullptr; }
  int IsPrimaryCell() override { return 0; }

  //@{
  /**
   * These methods are based on the svtkPolygon ones :
   * the svtkQuadraticPolygon (with n edges and 2*n points)
   * is transform into a svtkPolygon (with 2*n edges and 2*n points)
   * and the svtkPolygon methods are called.
   */
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;
  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;
  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;
  void InterpolateFunctions(const double x[3], double* weights) override;
  static void ComputeCentroid(svtkIdTypeArray* ids, svtkPoints* pts, double centroid[3]);
  int ParameterizePolygon(
    double p0[3], double p10[3], double& l10, double p20[3], double& l20, double n[3]);
  static int PointInPolygon(double x[3], int numPts, double* pts, double bounds[6], double n[3]);
  int Triangulate(svtkIdList* outTris);
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  int NonDegenerateTriangulate(svtkIdList* outTris);
  static double DistanceToPolygon(
    double x[3], int numPts, double* pts, double bounds[6], double closest[3]);
  static int IntersectPolygonWithPolygon(int npts, double* pts, double bounds[6], int npts2,
    double* pts2, double bounds2[6], double tol, double x[3]);
  static int IntersectConvex2DCells(
    svtkCell* cell1, svtkCell* cell2, double tol, double p0[3], double p1[3]);
  //@}

  // Not implemented
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;

  //@{
  /**
   * Set/Get the flag indicating whether to use Mean Value Coordinate for the
   * interpolation. If true, InterpolateFunctions() uses the Mean Value
   * Coordinate to compute weights. Otherwise, the conventional 1/r^2 method
   * is used. The UseMVCInterpolation parameter is set to true by default.
   */
  svtkGetMacro(UseMVCInterpolation, bool);
  svtkSetMacro(UseMVCInterpolation, bool);
  //@}

protected:
  svtkQuadraticPolygon();
  ~svtkQuadraticPolygon() override;

  // variables used by instances of this class
  svtkPolygon* Polygon;
  svtkQuadraticEdge* Edge;

  // Parameter indicating whether to use Mean Value Coordinate algorithm
  // for interpolation. The parameter is true by default.
  bool UseMVCInterpolation;

  //@{
  /**
   * Methods to transform a svtkQuadraticPolygon variable into a svtkPolygon
   * variable.
   */
  static void GetPermutationFromPolygon(svtkIdType nb, svtkIdList* permutation);
  static void PermuteToPolygon(svtkIdType nbPoints, double* inPoints, double* outPoints);
  static void PermuteToPolygon(svtkCell* inCell, svtkCell* outCell);
  static void PermuteToPolygon(svtkPoints* inPoints, svtkPoints* outPoints);
  static void PermuteToPolygon(svtkIdTypeArray* inIds, svtkIdTypeArray* outIds);
  static void PermuteToPolygon(svtkDataArray* inDataArray, svtkDataArray* outDataArray);
  void InitializePolygon();
  //@}

  //@{
  /**
   * Methods to transform a svtkPolygon variable into a svtkQuadraticPolygon
   * variable.
   */
  static void GetPermutationToPolygon(svtkIdType nb, svtkIdList* permutation);
  static void PermuteFromPolygon(svtkIdType nb, double* values);
  static void ConvertFromPolygon(svtkIdList* ids);
  //@}

private:
  svtkQuadraticPolygon(const svtkQuadraticPolygon&) = delete;
  void operator=(const svtkQuadraticPolygon&) = delete;
};

#endif
