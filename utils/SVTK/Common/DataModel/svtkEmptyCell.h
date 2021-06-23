/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkEmptyCell.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkEmptyCell
 * @brief   an empty cell used as a place-holder during processing
 *
 * svtkEmptyCell is a concrete implementation of svtkCell. It is used
 * during processing to represented a deleted element.
 */

#ifndef svtkEmptyCell_h
#define svtkEmptyCell_h

#include "svtkCell.h"
#include "svtkCommonDataModelModule.h" // For export macro

class SVTKCOMMONDATAMODEL_EXPORT svtkEmptyCell : public svtkCell
{
public:
  static svtkEmptyCell* New();
  svtkTypeMacro(svtkEmptyCell, svtkCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * See the svtkCell API for descriptions of these methods.
   */
  int GetCellType() override { return SVTK_EMPTY_CELL; }
  int GetCellDimension() override { return 0; }
  int GetNumberOfEdges() override { return 0; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int) override { return nullptr; }
  svtkCell* GetFace(int) override { return nullptr; }
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;
  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts1, svtkCellArray* lines, svtkCellArray* verts2, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* pts, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
    svtkCellData* outCd, int insideOut) override;
  //@}

  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;

protected:
  svtkEmptyCell() {}
  ~svtkEmptyCell() override {}

private:
  svtkEmptyCell(const svtkEmptyCell&) = delete;
  void operator=(const svtkEmptyCell&) = delete;
};

#endif
