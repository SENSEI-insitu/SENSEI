/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBiQuadraticQuad.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkBiQuadraticQuad
 * @brief   cell represents a parabolic, 9-node
 * isoparametric quad
 *
 * svtkQuadraticQuad is a concrete implementation of svtkNonLinearCell to
 * represent a two-dimensional, 9-node isoparametric parabolic quadrilateral
 * element with a Centerpoint. The interpolation is the standard finite
 * element, quadratic isoparametric shape function. The cell includes a
 * mid-edge node for each of the four edges of the cell and a center node at
 * the surface. The ordering of the eight points defining the cell are point
 * ids (0-3,4-8) where ids 0-3 define the four corner vertices of the quad;
 * ids 4-7 define the midedge nodes (0,1), (1,2), (2,3), (3,0) and 8 define
 * the face center node.
 *
 * @sa
 * svtkQuadraticEdge svtkQuadraticTriangle svtkQuadraticTetra
 * svtkQuadraticHexahedron svtkQuadraticWedge svtkQuadraticPyramid
 * svtkQuadraticQuad
 *
 * @par Thanks:
 * Thanks to Soeren Gebbert who developed this class and
 * integrated it into SVTK 5.0.
 */

#ifndef svtkBiQuadraticQuad_h
#define svtkBiQuadraticQuad_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNonLinearCell.h"

class svtkQuadraticEdge;
class svtkQuad;
class svtkTriangle;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkBiQuadraticQuad : public svtkNonLinearCell
{
public:
  static svtkBiQuadraticQuad* New();
  svtkTypeMacro(svtkBiQuadraticQuad, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Implement the svtkCell API. See the svtkCell API for descriptions
   * of these methods.
   */
  int GetCellType() override { return SVTK_BIQUADRATIC_QUAD; }
  int GetCellDimension() override { return 2; }
  int GetNumberOfEdges() override { return 4; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int) override;
  svtkCell* GetFace(int) override { return nullptr; }

  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;
  int EvaluatePosition(const double x[3], double* closestPoint, int& subId, double pcoords[3],
    double& dist2, double* weights) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  double* GetParametricCoords() override;

  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;

  /**
   * Clip this biquadratic quad using scalar value provided. Like contouring,
   * except that it cuts the twi quads to produce linear triangles.
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

  void InterpolateFunctions(const double pcoords[3], double weights[9]) override
  {
    svtkBiQuadraticQuad::InterpolationFunctionsPrivate(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[18]) override
  {
    svtkBiQuadraticQuad::InterpolationDerivsPrivate(pcoords, derivs);
  }
  //@}

protected:
  svtkBiQuadraticQuad();
  ~svtkBiQuadraticQuad() override;

  svtkQuadraticEdge* Edge;
  svtkQuad* Quad;
  svtkTriangle* Triangle;
  svtkDoubleArray* Scalars;

private:
  svtkBiQuadraticQuad(const svtkBiQuadraticQuad&) = delete;
  void operator=(const svtkBiQuadraticQuad&) = delete;

  static void InterpolationFunctionsPrivate(const double pcoords[3], double weights[9]);
  static void InterpolationDerivsPrivate(const double pcoords[3], double derivs[18]);
};
//----------------------------------------------------------------------------
inline int svtkBiQuadraticQuad::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = 0.5;
  pcoords[2] = 0.;
  return 0;
}

#endif
