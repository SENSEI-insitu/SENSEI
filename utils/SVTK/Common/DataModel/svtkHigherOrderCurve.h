/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHigherOrderCurve.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME svtkHigherOrderCurve
// .SECTION Description
// .SECTION See Also

#ifndef svtkHigherOrderCurve_h
#define svtkHigherOrderCurve_h

#include "svtkCellType.h"              // For GetCellType.
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNew.h"                   // For member variable.
#include "svtkNonLinearCell.h"
#include "svtkSmartPointer.h" // For member variable.

class svtkCellData;
class svtkDoubleArray;
class svtkIdList;
class svtkLine;
class svtkPointData;
class svtkPoints;
class svtkVector3d;
class svtkVector3i;

class SVTKCOMMONDATAMODEL_EXPORT svtkHigherOrderCurve : public svtkNonLinearCell
{
public:
  svtkTypeMacro(svtkHigherOrderCurve, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  int GetCellType() override = 0;
  int GetCellDimension() override { return 1; }
  int RequiresInitialization() override { return 1; }
  int GetNumberOfEdges() override { return 0; }
  int GetNumberOfFaces() override { return 0; }
  svtkCell* GetEdge(int) override { return nullptr; }
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
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  void SetParametricCoords();
  double* GetParametricCoords() override;
  int GetParametricCenter(double center[3]) override;

  double GetParametricDistance(const double pcoords[3]) override;

  const int* GetOrder();
  int GetOrder(int i) { return this->GetOrder()[i]; }

  void InterpolateFunctions(const double pcoords[3], double* weights) override = 0;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override = 0;

  bool SubCellCoordinatesFromId(svtkVector3i& ijk, int subId);
  bool SubCellCoordinatesFromId(int& i, int subId);
  int PointIndexFromIJK(int i, int, int);
  bool TransformApproxToCellParams(int subCell, double* pcoords);

protected:
  svtkHigherOrderCurve();
  ~svtkHigherOrderCurve() override;

  svtkLine* GetApprox();
  void PrepareApproxData(
    svtkPointData* pd, svtkCellData* cd, svtkIdType cellId, svtkDataArray* cellScalars);
  virtual svtkLine* GetApproximateLine(
    int subId, svtkDataArray* scalarsIn = nullptr, svtkDataArray* scalarsOut = nullptr) = 0;

  int Order[2];
  svtkSmartPointer<svtkPoints> PointParametricCoordinates;
  svtkSmartPointer<svtkLine> Approx;
  svtkSmartPointer<svtkPointData> ApproxPD;
  svtkSmartPointer<svtkCellData> ApproxCD;
  svtkNew<svtkDoubleArray> CellScalars;
  svtkNew<svtkDoubleArray> Scalars;
  svtkNew<svtkPoints> TmpPts;
  svtkNew<svtkIdList> TmpIds;

private:
  svtkHigherOrderCurve(const svtkHigherOrderCurve&) = delete;
  void operator=(const svtkHigherOrderCurve&) = delete;
};

inline int svtkHigherOrderCurve::GetParametricCenter(double center[3])
{
  center[0] = 0.5;
  center[1] = center[2] = 0.0;
  return 0;
}

#endif // svtkHigherOrderCurve_h
