/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPolyLine.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPolyLine
 * @brief   cell represents a set of 1D lines
 *
 * svtkPolyLine is a concrete implementation of svtkCell to represent a set
 * of 1D lines.
 */

#ifndef svtkPolyLine_h
#define svtkPolyLine_h

#include "svtkCell.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkPoints;
class svtkCellArray;
class svtkLine;
class svtkDataArray;
class svtkIncrementalPointLocator;
class svtkCellData;

class SVTKCOMMONDATAMODEL_EXPORT svtkPolyLine : public svtkCell
{
public:
  static svtkPolyLine* New();
  svtkTypeMacro(svtkPolyLine, svtkCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Given points and lines, compute normals to lines. These are not true
   * normals, they are "orientation" normals used by classes like svtkTubeFilter
   * that control the rotation around the line. The normals try to stay pointing
   * in the same direction as much as possible (i.e., minimal rotation) w.r.t the
   * firstNormal (computed if nullptr). Always returns 1 (success).
   */
  static int GenerateSlidingNormals(svtkPoints*, svtkCellArray*, svtkDataArray*);
  static int GenerateSlidingNormals(svtkPoints*, svtkCellArray*, svtkDataArray*, double* firstNormal);
  //@}

  //@{
  /**
   * See the svtkCell API for descriptions of these methods.
   */
  int GetCellType() override { return SVTK_POLY_LINE; }
  int GetCellDimension() override { return 1; }
  int GetNumberOfEdges() override { return 0; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int svtkNotUsed(edgeId)) override { return nullptr; }
  svtkCell* GetFace(int svtkNotUsed(faceId)) override { return nullptr; }
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;
  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* lines, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;
  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  int IsPrimaryCell() override { return 0; }
  //@}

  /**
   * Return the center of the point cloud in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

protected:
  svtkPolyLine();
  ~svtkPolyLine() override;

  svtkLine* Line;

private:
  svtkPolyLine(const svtkPolyLine&) = delete;
  void operator=(const svtkPolyLine&) = delete;
};

#endif
