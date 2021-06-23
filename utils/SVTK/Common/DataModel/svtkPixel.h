/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPixel.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPixel
 * @brief   a cell that represents an orthogonal quadrilateral
 *
 * svtkPixel is a concrete implementation of svtkCell to represent a 2D
 * orthogonal quadrilateral. Unlike svtkQuad, the corners are at right angles,
 * and aligned along x-y-z coordinate axes leading to large increases in
 * computational efficiency.
 */

#ifndef svtkPixel_h
#define svtkPixel_h

#include "svtkCell.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkLine;
class svtkIncrementalPointLocator;

class SVTKCOMMONDATAMODEL_EXPORT svtkPixel : public svtkCell
{
public:
  static svtkPixel* New();
  svtkTypeMacro(svtkPixel, svtkCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * See the svtkCell API for descriptions of these methods.
   */
  int GetCellType() override { return SVTK_PIXEL; }
  int GetCellDimension() override { return 2; }
  int GetNumberOfEdges() override { return 4; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int) override { return nullptr; }
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
  //@}

  /**
   * Return the center of the triangle in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  double* GetParametricCoords() override;

  static void InterpolationFunctions(const double pcoords[3], double weights[4]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[8]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[4]) override
  {
    svtkPixel::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[8]) override
  {
    svtkPixel::InterpolationDerivs(pcoords, derivs);
  }
  //@}

protected:
  svtkPixel();
  ~svtkPixel() override;

  svtkLine* Line;

private:
  svtkPixel(const svtkPixel&) = delete;
  void operator=(const svtkPixel&) = delete;
};

//----------------------------------------------------------------------------
inline int svtkPixel::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = 0.5;
  pcoords[2] = 0.0;
  return 0;
}

#endif
