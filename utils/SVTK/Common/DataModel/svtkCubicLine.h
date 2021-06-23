/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCubicLine.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCubicLine
 * @brief   cell represents a cubic , isoparametric 1D line
 *
 * svtkCubicLine is a concrete implementation of svtkNonLinearCell to represent a 1D Cubic line.
 * The Cubic Line is the 4 nodes isoparametric parabolic line . The
 * interpolation is the standard finite element, cubic isoparametric
 * shape function. The cell includes two mid-edge nodes. The ordering of the
 * four points defining the cell is point ids (0,1,2,3) where id #2 and #3 are the
 * mid-edge nodes. Please note that the parametric coordinates lie between -1 and 1
 * in accordance with most standard documentations.
 * @par Thanks:
 * <verbatim>
 * This file has been developed by Oxalya - www.oxalya.com
 * Copyright (c) EDF - www.edf.fr
 * </verbatim>
 */

#ifndef svtkCubicLine_h
#define svtkCubicLine_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNonLinearCell.h"

class svtkLine;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkCubicLine : public svtkNonLinearCell
{
public:
  static svtkCubicLine* New();
  svtkTypeMacro(svtkCubicLine, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * See the svtkCell API for descriptions of these methods.
   */
  int GetCellType() override { return SVTK_CUBIC_LINE; }
  int GetCellDimension() override { return 1; }
  int GetNumberOfEdges() override { return 0; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int) override { return nullptr; }
  svtkCell* GetFace(int) override { return nullptr; }
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
  //@}

  /**
   * Return the distance of the parametric coordinate provided to the
   * cell. If inside the cell, a distance of zero is returned.
   */
  double GetParametricDistance(const double pcoords[3]) override;

  /**
   * Clip this line using scalar value provided. Like contouring, except
   * that it cuts the line to produce other lines.
   */
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* lines, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;

  /**
   * Return the center of the triangle in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  /**
   * Line-line intersection. Intersection has to occur within [0,1] parametric
   * coordinates and with specified tolerance.
   */
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;

  static void InterpolationFunctions(const double pcoords[3], double weights[4]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[4]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[4]) override
  {
    svtkCubicLine::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[4]) override
  {
    svtkCubicLine::InterpolationDerivs(pcoords, derivs);
  }
  //@}

protected:
  svtkCubicLine();
  ~svtkCubicLine() override;

  svtkLine* Line;
  svtkDoubleArray* Scalars; // used to avoid New/Delete in contouring/clipping

private:
  svtkCubicLine(const svtkCubicLine&) = delete;
  void operator=(const svtkCubicLine&) = delete;
};

//----------------------------------------------------------------------------
inline int svtkCubicLine::GetParametricCenter(double pcoords[3])
{

  pcoords[0] = pcoords[1] = pcoords[2] = 0.0;
  return 0;
}

#endif
