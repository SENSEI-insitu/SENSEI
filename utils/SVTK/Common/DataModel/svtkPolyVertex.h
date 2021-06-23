/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPolyVertex.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPolyVertex
 * @brief   cell represents a set of 0D vertices
 *
 * svtkPolyVertex is a concrete implementation of svtkCell to represent a
 * set of 3D vertices.
 */

#ifndef svtkPolyVertex_h
#define svtkPolyVertex_h

#include "svtkCell.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkVertex;
class svtkIncrementalPointLocator;

class SVTKCOMMONDATAMODEL_EXPORT svtkPolyVertex : public svtkCell
{
public:
  static svtkPolyVertex* New();
  svtkTypeMacro(svtkPolyVertex, svtkCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * See the svtkCell API for descriptions of these methods.
   */
  int GetCellType() override { return SVTK_POLY_VERTEX; }
  int GetCellDimension() override { return 0; }
  int GetNumberOfEdges() override { return 0; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int svtkNotUsed(edgeId)) override { return nullptr; }
  svtkCell* GetFace(int svtkNotUsed(faceId)) override { return nullptr; }
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;
  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
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
  svtkPolyVertex();
  ~svtkPolyVertex() override;

  svtkVertex* Vertex;

private:
  svtkPolyVertex(const svtkPolyVertex&) = delete;
  void operator=(const svtkPolyVertex&) = delete;
};

#endif
