/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHigherOrderTriangle.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkHigherOrderTriangle
 * @brief   A 2D cell that represents an arbitrary order HigherOrder triangle
 *
 * svtkHigherOrderTriangle is a concrete implementation of svtkCell to represent a
 * 2D triangle using HigherOrder shape functions of user specified order.
 *
 * The number of points in a HigherOrder cell determines the order over which they
 * are iterated relative to the parametric coordinate system of the cell. The
 * first points that are reported are vertices. They appear in the same order in
 * which they would appear in linear cells. Mid-edge points are reported next.
 * They are reported in sequence. For two- and three-dimensional (3D) cells, the
 * following set of points to be reported are face points. Finally, 3D cells
 * report points interior to their volume.
 */

#ifndef svtkHigherOrderTriangle_h
#define svtkHigherOrderTriangle_h

#include <functional> //For std::function

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNew.h"                   // For member variable.
#include "svtkNonLinearCell.h"
#include "svtkSmartPointer.h" // For member variable.

#include <vector> // For caching

class svtkDoubleArray;
class svtkHigherOrderCurve;
class svtkTriangle;

class SVTKCOMMONDATAMODEL_EXPORT svtkHigherOrderTriangle : public svtkNonLinearCell
{
public:
  svtkTypeMacro(svtkHigherOrderTriangle, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  int GetCellType() override = 0;
  int GetCellDimension() override { return 2; }
  int RequiresInitialization() override { return 1; }
  int GetNumberOfEdges() override { return 3; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int edgeId) override = 0;
  void SetEdgeIdsAndPoints(int edgeId,
    const std::function<void(const svtkIdType&)>& set_number_of_ids_and_points,
    const std::function<void(const svtkIdType&, const svtkIdType&)>& set_ids_and_points);
  svtkCell* GetFace(int) override { return nullptr; }

  void Initialize() override;

  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;
  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void JacobianInverse(const double pcoords[3], double** inverse, double* derivs);
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  void SetParametricCoords();
  double* GetParametricCoords() override;

  int GetParametricCenter(double pcoords[3]) override;
  double GetParametricDistance(const double pcoords[3]) override;

  void InterpolateFunctions(const double pcoords[3], double* weights) override = 0;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override = 0;

  svtkIdType GetOrder() const { return this->Order; }
  svtkIdType ComputeOrder();

  void ToBarycentricIndex(svtkIdType index, svtkIdType* bindex);
  svtkIdType ToIndex(const svtkIdType* bindex);

  static void BarycentricIndex(svtkIdType index, svtkIdType* bindex, svtkIdType order);
  static svtkIdType Index(const svtkIdType* bindex, svtkIdType order);

  static double eta(svtkIdType n, svtkIdType chi, double sigma);
  static double d_eta(svtkIdType n, svtkIdType chi, double sigma);
  virtual svtkHigherOrderCurve* getEdgeCell() = 0;

protected:
  svtkHigherOrderTriangle();
  ~svtkHigherOrderTriangle() override;

  svtkIdType GetNumberOfSubtriangles() const { return this->NumberOfSubtriangles; }
  svtkIdType ComputeNumberOfSubtriangles();

  // Description:
  // Given the index of the subtriangle, compute the barycentric indices of
  // the subtriangle's vertices.
  void SubtriangleBarycentricPointIndices(svtkIdType cellIndex, svtkIdType (&pointBIndices)[3][3]);

  svtkTriangle* Face;
  svtkDoubleArray* Scalars; // used to avoid New/Delete in contouring/clipping
  svtkIdType Order;
  svtkIdType NumberOfSubtriangles;
  svtkSmartPointer<svtkPoints> PointParametricCoordinates;

  std::vector<svtkIdType> BarycentricIndexMap;
  std::vector<svtkIdType> IndexMap;
  std::vector<svtkIdType> SubtriangleIndexMap;

private:
  svtkHigherOrderTriangle(const svtkHigherOrderTriangle&) = delete;
  void operator=(const svtkHigherOrderTriangle&) = delete;
};

#endif
