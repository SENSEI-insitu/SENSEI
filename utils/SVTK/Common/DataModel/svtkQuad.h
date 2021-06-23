/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuad.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkQuad
 * @brief   a cell that represents a 2D quadrilateral
 *
 * svtkQuad is a concrete implementation of svtkCell to represent a 2D
 * quadrilateral. svtkQuad is defined by the four points (0,1,2,3) in
 * counterclockwise order. svtkQuad uses the standard isoparametric
 * interpolation functions for a linear quadrilateral.
 */

#ifndef svtkQuad_h
#define svtkQuad_h

#include "svtkCell.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkLine;
class svtkTriangle;
class svtkIncrementalPointLocator;

class SVTKCOMMONDATAMODEL_EXPORT svtkQuad : public svtkCell
{
public:
  static svtkQuad* New();
  svtkTypeMacro(svtkQuad, svtkCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * See the svtkCell API for descriptions of these methods.
   */
  int GetCellType() override { return SVTK_QUAD; }
  int GetCellDimension() override { return 2; }
  int GetNumberOfEdges() override { return 4; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int) override { return nullptr; }
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;
  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;
  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  double* GetParametricCoords() override;
  //@}

  /**
   * Return the center of the triangle in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  /**
   * Clip this quad using scalar value provided. Like contouring, except
   * that it cuts the quad to produce other quads and/or triangles.
   */
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;

  static void InterpolationFunctions(const double pcoords[3], double sf[4]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[8]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double sf[4]) override
  {
    svtkQuad::InterpolationFunctions(pcoords, sf);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[8]) override
  {
    svtkQuad::InterpolationDerivs(pcoords, derivs);
  }
  //@}

  /**
   * Return the ids of the vertices defining edge (`edgeId`).
   * Ids are related to the cell, not to the dataset.
   *
   * @note The return type changed. It used to be int*, it is now const svtkIdType*.
   * This is so ids are unified between svtkCell and svtkPoints, and so svtkCell ids
   * can be used as inputs in algorithms such as svtkPolygon::ComputeNormal.
   */
  const svtkIdType* GetEdgeArray(svtkIdType edgeId);

protected:
  svtkQuad();
  ~svtkQuad() override;

  svtkLine* Line;
  svtkTriangle* Triangle;

private:
  svtkQuad(const svtkQuad&) = delete;
  void operator=(const svtkQuad&) = delete;
};
//----------------------------------------------------------------------------
inline int svtkQuad::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = 0.5;
  pcoords[2] = 0.0;
  return 0;
}

#endif
