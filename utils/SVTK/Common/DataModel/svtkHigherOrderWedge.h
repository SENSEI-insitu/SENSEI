/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHigherOrderWedge.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkHigherOrderWedge
 * @brief   A 3D cell that represents an arbitrary order HigherOrder wedge
 *
 * svtkHigherOrderWedge is a concrete implementation of svtkCell to represent a
 * 3D wedge using HigherOrder shape functions of user specified order.
 * A wedge consists of two triangular and three quadrilateral faces.
 * The first six points of the wedge (0-5) are the "corner" points
 * where the first three points are the base of the wedge. This wedge
 * point ordering is opposite the svtkWedge ordering though in that
 * the base of the wedge defined by the first three points (0,1,2) form
 * a triangle whose normal points inward (toward the triangular face (3,4,5)).
 * While this is opposite the svtkWedge convention it is consistent with
 * every other cell type in SVTK. The first 2 parametric coordinates of the
 * HigherOrder wedge or for the triangular base and vary between 0 and 1. The
 * third parametric coordinate is between the two triangular faces and goes
 * from 0 to 1 as well.
 */

#ifndef svtkHigherOrderWedge_h
#define svtkHigherOrderWedge_h

#include <functional> //For std::function

#include "svtkCellType.h"              // For GetCellType.
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNew.h"                   // For member variable.
#include "svtkNonLinearCell.h"
#include "svtkSmartPointer.h" // For member variable.

class svtkCellData;
class svtkDoubleArray;
class svtkWedge;
class svtkIdList;
class svtkPointData;
class svtkPoints;
class svtkVector3d;
class svtkVector3i;
class svtkHigherOrderCurve;
class svtkHigherOrderInterpolation;
class svtkHigherOrderQuadrilateral;
class svtkHigherOrderTriangle;

class SVTKCOMMONDATAMODEL_EXPORT svtkHigherOrderWedge : public svtkNonLinearCell
{
public:
  svtkTypeMacro(svtkHigherOrderWedge, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  int GetCellType() override = 0;
  int GetCellDimension() override { return 3; }
  int RequiresInitialization() override { return 1; }
  int GetNumberOfEdges() override { return 9; }
  int GetNumberOfFaces() override { return 5; }
  svtkCell* GetEdge(int edgeId) override = 0;
  void SetEdgeIdsAndPoints(int edgeId,
    const std::function<void(const svtkIdType&)>& set_number_of_ids_and_points,
    const std::function<void(const svtkIdType&, const svtkIdType&)>& set_ids_and_points);
  svtkCell* GetFace(int faceId) override = 0;

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
  virtual void SetOrder(const int s, const int t, const int u, const svtkIdType numPts);
  virtual const int* GetOrder();
  virtual int GetOrder(int i) { return this->GetOrder()[i]; }

  void InterpolateFunctions(const double pcoords[3], double* weights) override = 0;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override = 0;

  bool SubCellCoordinatesFromId(svtkVector3i& ijk, int subId);
  bool SubCellCoordinatesFromId(int& i, int& j, int& k, int subId);
  static int PointIndexFromIJK(int i, int j, int k, const int* order);
  int PointIndexFromIJK(int i, int j, int k);
  bool TransformApproxToCellParams(int subCell, double* pcoords);
  bool TransformFaceToCellParams(int bdyFace, double* pcoords);

  static int GetNumberOfApproximatingWedges(const int* order);
  int GetNumberOfApproximatingWedges()
  {
    return svtkHigherOrderWedge::GetNumberOfApproximatingWedges(this->GetOrder());
  }
  virtual svtkHigherOrderQuadrilateral* getBdyQuad() = 0;
  virtual svtkHigherOrderTriangle* getBdyTri() = 0;
  virtual svtkHigherOrderCurve* getEdgeCell() = 0;
  virtual svtkHigherOrderInterpolation* getInterp() = 0;

protected:
  svtkHigherOrderWedge();
  ~svtkHigherOrderWedge() override;

  svtkWedge* GetApprox();
  void PrepareApproxData(
    svtkPointData* pd, svtkCellData* cd, svtkIdType cellId, svtkDataArray* cellScalars);
  svtkWedge* GetApproximateWedge(
    int subId, svtkDataArray* scalarsIn = nullptr, svtkDataArray* scalarsOut = nullptr);

  void GetTriangularFace(svtkHigherOrderTriangle* result, int faceId,
    const std::function<void(const svtkIdType&)>& set_number_of_ids_and_points,
    const std::function<void(const svtkIdType&, const svtkIdType&)>& set_ids_and_points);
  void GetQuadrilateralFace(svtkHigherOrderQuadrilateral* result, int faceId,
    const std::function<void(const svtkIdType&)>& set_number_of_ids_and_points,
    const std::function<void(const svtkIdType&, const svtkIdType&)>& set_ids_and_points);

  int Order[4];
  svtkSmartPointer<svtkPoints> PointParametricCoordinates;
  svtkSmartPointer<svtkWedge> Approx;
  svtkSmartPointer<svtkPointData> ApproxPD;
  svtkSmartPointer<svtkCellData> ApproxCD;
  svtkNew<svtkDoubleArray> CellScalars;
  svtkNew<svtkDoubleArray> Scalars;
  svtkNew<svtkPoints> TmpPts;
  svtkNew<svtkIdList> TmpIds;

private:
  svtkHigherOrderWedge(const svtkHigherOrderWedge&) = delete;
  void operator=(const svtkHigherOrderWedge&) = delete;
};

inline int svtkHigherOrderWedge::GetParametricCenter(double center[3])
{
  center[0] = center[1] = 1. / 3.;
  center[2] = 0.5;
  return 0;
}

#endif // svtkHigherOrderWedge_h
