/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHigherOrderQuadrilateral.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME svtkHigherOrderQuadrilateral
// .SECTION Description
// .SECTION See Also

#ifndef svtkHigherOrderQuadrilateral_h
#define svtkHigherOrderQuadrilateral_h

#include <functional> //For std::function

#include "svtkCellType.h"              // For GetCellType.
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNew.h"                   // For member variable.
#include "svtkNonLinearCell.h"
#include "svtkSmartPointer.h" // For member variable.

class svtkCellData;
class svtkDoubleArray;
class svtkIdList;
class svtkHigherOrderCurve;
class svtkPointData;
class svtkPoints;
class svtkQuad;
class svtkVector3d;
class svtkVector3i;

class SVTKCOMMONDATAMODEL_EXPORT svtkHigherOrderQuadrilateral : public svtkNonLinearCell
{
public:
  svtkTypeMacro(svtkHigherOrderQuadrilateral, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  int GetCellType() override = 0;
  int GetCellDimension() override { return 2; }
  int RequiresInitialization() override { return 0; }
  int GetNumberOfEdges() override { return 4; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int edgeId) override = 0;
  svtkCell* GetFace(int svtkNotUsed(faceId)) override { return nullptr; }
  void SetEdgeIdsAndPoints(int edgeId,
    const std::function<void(const svtkIdType&)>& set_number_of_ids_and_points,
    const std::function<void(const svtkIdType&, const svtkIdType&)>& set_ids_and_points);

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
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  void SetParametricCoords();
  double* GetParametricCoords() override;
  int GetParametricCenter(double center[3]) override;

  double GetParametricDistance(const double pcoords[3]) override;

  virtual void SetOrderFromCellData(
    svtkCellData* cell_data, const svtkIdType numPts, const svtkIdType cell_id);
  virtual void SetUniformOrderFromNumPoints(const svtkIdType numPts);
  virtual void SetOrder(const int s, const int t);
  virtual const int* GetOrder();
  virtual int GetOrder(int i) { return this->GetOrder()[i]; }

  void InterpolateFunctions(const double pcoords[3], double* weights) override = 0;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override = 0;

  bool SubCellCoordinatesFromId(svtkVector3i& ijk, int subId);
  bool SubCellCoordinatesFromId(int& i, int& j, int& k, int subId);
  int PointIndexFromIJK(int i, int j, int k);
  static int PointIndexFromIJK(int i, int j, const int* order);
  bool TransformApproxToCellParams(int subCell, double* pcoords);

  virtual svtkHigherOrderCurve* getEdgeCell() = 0;

protected:
  svtkHigherOrderQuadrilateral();
  ~svtkHigherOrderQuadrilateral() override;

  svtkQuad* GetApprox();
  // The verion of GetApproximateQuad between Lagrange and Bezier is different because Bezier is
  // non-interpolatory
  void PrepareApproxData(
    svtkPointData* pd, svtkCellData* cd, svtkIdType cellId, svtkDataArray* cellScalars);
  virtual svtkQuad* GetApproximateQuad(
    int subId, svtkDataArray* scalarsIn = nullptr, svtkDataArray* scalarsOut = nullptr) = 0;

  int Order[3];
  svtkSmartPointer<svtkPoints> PointParametricCoordinates;
  svtkSmartPointer<svtkQuad> Approx;
  svtkSmartPointer<svtkPointData> ApproxPD;
  svtkSmartPointer<svtkCellData> ApproxCD;
  svtkNew<svtkDoubleArray> CellScalars;
  svtkNew<svtkDoubleArray> Scalars;
  svtkNew<svtkPoints> TmpPts;
  svtkNew<svtkIdList> TmpIds;

private:
  svtkHigherOrderQuadrilateral(const svtkHigherOrderQuadrilateral&) = delete;
  void operator=(const svtkHigherOrderQuadrilateral&) = delete;
};

inline int svtkHigherOrderQuadrilateral::GetParametricCenter(double center[3])
{
  center[0] = center[1] = 0.5;
  center[2] = 0.0;
  return 0;
}

#endif // svtkHigherOrderQuadrilateral_h
