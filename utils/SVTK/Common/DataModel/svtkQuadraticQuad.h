/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadraticQuad.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkQuadraticQuad
 * @brief   cell represents a parabolic, 8-node isoparametric quad
 *
 * svtkQuadraticQuad is a concrete implementation of svtkNonLinearCell to
 * represent a two-dimensional, 8-node isoparametric parabolic quadrilateral
 * element. The interpolation is the standard finite element, quadratic
 * isoparametric shape function. The cell includes a mid-edge node for each
 * of the four edges of the cell. The ordering of the eight points defining
 * the cell are point ids (0-3,4-7) where ids 0-3 define the four corner
 * vertices of the quad; ids 4-7 define the midedge nodes (0,1), (1,2),
 * (2,3), (3,0).
 *
 * @sa
 * svtkQuadraticEdge svtkQuadraticTriangle svtkQuadraticTetra
 * svtkQuadraticHexahedron svtkQuadraticWedge svtkQuadraticPyramid
 */

#ifndef svtkQuadraticQuad_h
#define svtkQuadraticQuad_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNonLinearCell.h"

class svtkQuadraticEdge;
class svtkQuad;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkQuadraticQuad : public svtkNonLinearCell
{
public:
  static svtkQuadraticQuad* New();
  svtkTypeMacro(svtkQuadraticQuad, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Implement the svtkCell API. See the svtkCell API for descriptions
   * of these methods.
   */
  int GetCellType() override { return SVTK_QUADRATIC_QUAD; }
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
  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  double* GetParametricCoords() override;

  /**
   * Clip this quadratic quad using scalar value provided. Like contouring,
   * except that it cuts the quad to produce linear triangles.
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

  static void InterpolationFunctions(const double pcoords[3], double weights[8]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[16]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[8]) override
  {
    svtkQuadraticQuad::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[16]) override
  {
    svtkQuadraticQuad::InterpolationDerivs(pcoords, derivs);
  }
  //@}

protected:
  svtkQuadraticQuad();
  ~svtkQuadraticQuad() override;

  svtkQuadraticEdge* Edge;
  svtkQuad* Quad;
  svtkPointData* PointData;
  svtkDoubleArray* Scalars;

  // In order to achieve some functionality we introduce a fake center point
  // which require to have some extra functionalities compare to other non-linar
  // cells
  svtkCellData* CellData;
  svtkDoubleArray* CellScalars;
  void Subdivide(double* weights);
  void InterpolateAttributes(
    svtkPointData* inPd, svtkCellData* inCd, svtkIdType cellId, svtkDataArray* cellScalars);

private:
  svtkQuadraticQuad(const svtkQuadraticQuad&) = delete;
  void operator=(const svtkQuadraticQuad&) = delete;
};
//----------------------------------------------------------------------------
inline int svtkQuadraticQuad::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = 0.5;
  pcoords[2] = 0.;
  return 0;
}

#endif
