/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHigherOrderHexahedron.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkHigherOrderHexahedron
 * @brief   A 3D cell that represents an arbitrary order HigherOrder hex
 *
 * svtkHigherOrderHexahedron is a concrete implementation of svtkCell to represent a
 * 3D hexahedron using HigherOrder shape functions of user specified order.
 *
 * @sa
 * svtkHexahedron
 */

#ifndef svtkHigherOrderHexahedron_h
#define svtkHigherOrderHexahedron_h

#include "svtkCellType.h"              // For GetCellType.
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkNew.h"                   // For member variable.
#include "svtkNonLinearCell.h"
#include "svtkSmartPointer.h" // For member variable.
#include <functional>        //For std::function

class svtkCellData;
class svtkDoubleArray;
class svtkHexahedron;
class svtkIdList;
class svtkHigherOrderCurve;
class svtkHigherOrderInterpolation;
class svtkHigherOrderQuadrilateral;
class svtkPointData;
class svtkPoints;
class svtkVector3d;
class svtkVector3i;

class SVTKCOMMONDATAMODEL_EXPORT svtkHigherOrderHexahedron : public svtkNonLinearCell
{
public:
  svtkTypeMacro(svtkHigherOrderHexahedron, svtkNonLinearCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  int GetCellType() override = 0;
  int GetCellDimension() override { return 3; }
  int RequiresInitialization() override { return 1; }
  int GetNumberOfEdges() override { return 12; }
  int GetNumberOfFaces() override { return 6; }
  svtkCell* GetEdge(int edgeId) override = 0;
  svtkCell* GetFace(int faceId) override = 0;
  void SetEdgeIdsAndPoints(int edgeId,
    const std::function<void(const svtkIdType&)>& set_number_of_ids_and_points,
    const std::function<void(const svtkIdType&, const svtkIdType&)>& set_ids_and_points);
  void SetFaceIdsAndPoints(svtkHigherOrderQuadrilateral* result, int faceId,
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
  virtual void SetOrder(const int s, const int t, const int u);
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
  virtual svtkHigherOrderCurve* getEdgeCell() = 0;
  virtual svtkHigherOrderQuadrilateral* getFaceCell() = 0;
  virtual svtkHigherOrderInterpolation* getInterp() = 0;

  static svtkIdType NodeNumberingMappingFromSVTK8To9(
    const int order[3], const svtkIdType node_id_svtk8);

protected:
  svtkHigherOrderHexahedron();
  ~svtkHigherOrderHexahedron() override;

  svtkHexahedron* GetApprox();
  void PrepareApproxData(
    svtkPointData* pd, svtkCellData* cd, svtkIdType cellId, svtkDataArray* cellScalars);
  virtual svtkHexahedron* GetApproximateHex(
    int subId, svtkDataArray* scalarsIn = nullptr, svtkDataArray* scalarsOut = nullptr) = 0;

  int Order[4];
  svtkSmartPointer<svtkPoints> PointParametricCoordinates;
  svtkSmartPointer<svtkHexahedron> Approx;
  svtkSmartPointer<svtkPointData> ApproxPD;
  svtkSmartPointer<svtkCellData> ApproxCD;
  svtkNew<svtkDoubleArray> CellScalars;
  svtkNew<svtkDoubleArray> Scalars;
  svtkNew<svtkPoints> TmpPts;
  svtkNew<svtkIdList> TmpIds;

private:
  svtkHigherOrderHexahedron(const svtkHigherOrderHexahedron&) = delete;
  void operator=(const svtkHigherOrderHexahedron&) = delete;
};

inline int svtkHigherOrderHexahedron::GetParametricCenter(double center[3])
{
  center[0] = center[1] = center[2] = 0.5;
  return 0;
}

#endif // svtkHigherOrderHexahedron_h
