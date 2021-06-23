/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericCell.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkGenericCell
 * @brief   provides thread-safe access to cells
 *
 * svtkGenericCell is a class that provides access to concrete types of cells.
 * It's main purpose is to allow thread-safe access to cells, supporting
 * the svtkDataSet::GetCell(svtkGenericCell *) method. svtkGenericCell acts
 * like any type of cell, it just dereferences an internal representation.
 * The SetCellType() methods use \#define constants; these are defined in
 * the file svtkCellType.h.
 *
 * @sa
 * svtkCell svtkDataSet
 */

#ifndef svtkGenericCell_h
#define svtkGenericCell_h

#include "svtkCell.h"
#include "svtkCommonDataModelModule.h" // For export macro

class SVTKCOMMONDATAMODEL_EXPORT svtkGenericCell : public svtkCell
{
public:
  /**
   * Create handle to any type of cell; by default a svtkEmptyCell.
   */
  static svtkGenericCell* New();

  svtkTypeMacro(svtkGenericCell, svtkCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Set the points object to use for this cell. This updates the internal cell
   * storage as well as the public member variable Points.
   */
  void SetPoints(svtkPoints* points);

  /**
   * Set the point ids to use for this cell. This updates the internal cell
   * storage as well as the public member variable PointIds.
   */
  void SetPointIds(svtkIdList* pointIds);

  //@{
  /**
   * See the svtkCell API for descriptions of these methods.
   */
  void ShallowCopy(svtkCell* c) override;
  void DeepCopy(svtkCell* c) override;
  int GetCellType() override;
  int GetCellDimension() override;
  int IsLinear() override;
  int RequiresInitialization() override;
  void Initialize() override;
  int RequiresExplicitFaceRepresentation() override;
  void SetFaces(svtkIdType* faces) override;
  svtkIdType* GetFaces() override;
  int GetNumberOfEdges() override;
  int GetNumberOfFaces() override;
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int faceId) override;
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;
  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* connectivity, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  int GetParametricCenter(double pcoords[3]) override;
  double* GetParametricCoords() override;
  int IsPrimaryCell() override;
  //@}

  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double* weights) override;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override;
  //@}

  /**
   * This method is used to support the svtkDataSet::GetCell(svtkGenericCell *)
   * method. It allows svtkGenericCell to act like any cell type by
   * dereferencing an internal instance of a concrete cell type. When
   * you set the cell type, you are resetting a pointer to an internal
   * cell which is then used for computation.
   */
  void SetCellType(int cellType);
  void SetCellTypeToEmptyCell() { this->SetCellType(SVTK_EMPTY_CELL); }
  void SetCellTypeToVertex() { this->SetCellType(SVTK_VERTEX); }
  void SetCellTypeToPolyVertex() { this->SetCellType(SVTK_POLY_VERTEX); }
  void SetCellTypeToLine() { this->SetCellType(SVTK_LINE); }
  void SetCellTypeToPolyLine() { this->SetCellType(SVTK_POLY_LINE); }
  void SetCellTypeToTriangle() { this->SetCellType(SVTK_TRIANGLE); }
  void SetCellTypeToTriangleStrip() { this->SetCellType(SVTK_TRIANGLE_STRIP); }
  void SetCellTypeToPolygon() { this->SetCellType(SVTK_POLYGON); }
  void SetCellTypeToPixel() { this->SetCellType(SVTK_PIXEL); }
  void SetCellTypeToQuad() { this->SetCellType(SVTK_QUAD); }
  void SetCellTypeToTetra() { this->SetCellType(SVTK_TETRA); }
  void SetCellTypeToVoxel() { this->SetCellType(SVTK_VOXEL); }
  void SetCellTypeToHexahedron() { this->SetCellType(SVTK_HEXAHEDRON); }
  void SetCellTypeToWedge() { this->SetCellType(SVTK_WEDGE); }
  void SetCellTypeToPyramid() { this->SetCellType(SVTK_PYRAMID); }
  void SetCellTypeToPentagonalPrism() { this->SetCellType(SVTK_PENTAGONAL_PRISM); }
  void SetCellTypeToHexagonalPrism() { this->SetCellType(SVTK_HEXAGONAL_PRISM); }
  void SetCellTypeToPolyhedron() { this->SetCellType(SVTK_POLYHEDRON); }
  void SetCellTypeToConvexPointSet() { this->SetCellType(SVTK_CONVEX_POINT_SET); }
  void SetCellTypeToQuadraticEdge() { this->SetCellType(SVTK_QUADRATIC_EDGE); }
  void SetCellTypeToCubicLine() { this->SetCellType(SVTK_CUBIC_LINE); }
  void SetCellTypeToQuadraticTriangle() { this->SetCellType(SVTK_QUADRATIC_TRIANGLE); }
  void SetCellTypeToBiQuadraticTriangle() { this->SetCellType(SVTK_BIQUADRATIC_TRIANGLE); }
  void SetCellTypeToQuadraticQuad() { this->SetCellType(SVTK_QUADRATIC_QUAD); }
  void SetCellTypeToQuadraticPolygon() { this->SetCellType(SVTK_QUADRATIC_POLYGON); }
  void SetCellTypeToQuadraticTetra() { this->SetCellType(SVTK_QUADRATIC_TETRA); }
  void SetCellTypeToQuadraticHexahedron() { this->SetCellType(SVTK_QUADRATIC_HEXAHEDRON); }
  void SetCellTypeToQuadraticWedge() { this->SetCellType(SVTK_QUADRATIC_WEDGE); }
  void SetCellTypeToQuadraticPyramid() { this->SetCellType(SVTK_QUADRATIC_PYRAMID); }
  void SetCellTypeToQuadraticLinearQuad() { this->SetCellType(SVTK_QUADRATIC_LINEAR_QUAD); }
  void SetCellTypeToBiQuadraticQuad() { this->SetCellType(SVTK_BIQUADRATIC_QUAD); }
  void SetCellTypeToQuadraticLinearWedge() { this->SetCellType(SVTK_QUADRATIC_LINEAR_WEDGE); }
  void SetCellTypeToBiQuadraticQuadraticWedge()
  {
    this->SetCellType(SVTK_BIQUADRATIC_QUADRATIC_WEDGE);
  }
  void SetCellTypeToTriQuadraticHexahedron() { this->SetCellType(SVTK_TRIQUADRATIC_HEXAHEDRON); }
  void SetCellTypeToBiQuadraticQuadraticHexahedron()
  {
    this->SetCellType(SVTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON);
  }
  void SetCellTypeToLagrangeTriangle() { this->SetCellType(SVTK_LAGRANGE_TRIANGLE); }
  void SetCellTypeToLagrangeTetra() { this->SetCellType(SVTK_LAGRANGE_TETRAHEDRON); }
  void SetCellTypeToLagrangeCurve() { this->SetCellType(SVTK_LAGRANGE_CURVE); }
  void SetCellTypeToLagrangeQuadrilateral() { this->SetCellType(SVTK_LAGRANGE_QUADRILATERAL); }
  void SetCellTypeToLagrangeHexahedron() { this->SetCellType(SVTK_LAGRANGE_HEXAHEDRON); }
  void SetCellTypeToLagrangeWedge() { this->SetCellType(SVTK_LAGRANGE_WEDGE); }

  void SetCellTypeToBezierTriangle() { this->SetCellType(SVTK_BEZIER_TRIANGLE); }
  void SetCellTypeToBezierTetra() { this->SetCellType(SVTK_BEZIER_TETRAHEDRON); }
  void SetCellTypeToBezierCurve() { this->SetCellType(SVTK_BEZIER_CURVE); }
  void SetCellTypeToBezierQuadrilateral() { this->SetCellType(SVTK_BEZIER_QUADRILATERAL); }
  void SetCellTypeToBezierHexahedron() { this->SetCellType(SVTK_BEZIER_HEXAHEDRON); }
  void SetCellTypeToBezierWedge() { this->SetCellType(SVTK_BEZIER_WEDGE); }
  /**
   * Instantiate a new svtkCell based on it's cell type value
   */
  static svtkCell* InstantiateCell(int cellType);

  svtkCell* GetRepresentativeCell() { return this->Cell; }

protected:
  svtkGenericCell();
  ~svtkGenericCell() override;

  svtkCell* Cell;
  svtkCell* CellStore[SVTK_NUMBER_OF_CELL_TYPES];

private:
  svtkGenericCell(const svtkGenericCell&) = delete;
  void operator=(const svtkGenericCell&) = delete;
};

#endif
