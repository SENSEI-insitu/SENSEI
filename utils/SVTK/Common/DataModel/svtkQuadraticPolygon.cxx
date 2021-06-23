/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadraticPolygon.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkQuadraticPolygon.h"

#include "svtkCellData.h"
#include "svtkDataArray.h"
#include "svtkIdTypeArray.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkPolygon.h"
#include "svtkQuadraticEdge.h"

svtkStandardNewMacro(svtkQuadraticPolygon);

//----------------------------------------------------------------------------
// Instantiate quadratic polygon.
svtkQuadraticPolygon::svtkQuadraticPolygon()
{
  this->Polygon = svtkPolygon::New();
  this->Edge = svtkQuadraticEdge::New();
  this->UseMVCInterpolation = true;
}

//----------------------------------------------------------------------------
svtkQuadraticPolygon::~svtkQuadraticPolygon()
{
  this->Polygon->Delete();
  this->Edge->Delete();
}

//----------------------------------------------------------------------------
svtkCell* svtkQuadraticPolygon::GetEdge(int edgeId)
{
  int numEdges = this->GetNumberOfEdges();

  edgeId = (edgeId < 0 ? 0 : (edgeId > numEdges - 1 ? numEdges - 1 : edgeId));
  int p = (edgeId + 1) % numEdges;

  // load point id's
  this->Edge->PointIds->SetId(0, this->PointIds->GetId(edgeId));
  this->Edge->PointIds->SetId(1, this->PointIds->GetId(p));
  this->Edge->PointIds->SetId(2, this->PointIds->GetId(edgeId + numEdges));

  // load coordinates
  this->Edge->Points->SetPoint(0, this->Points->GetPoint(edgeId));
  this->Edge->Points->SetPoint(1, this->Points->GetPoint(p));
  this->Edge->Points->SetPoint(2, this->Points->GetPoint(edgeId + numEdges));

  return this->Edge;
}

//----------------------------------------------------------------------------
int svtkQuadraticPolygon::EvaluatePosition(const double x[3], double closestPoint[3], int& subId,
  double pcoords[3], double& minDist2, double weights[])
{
  this->InitializePolygon();
  int result = this->Polygon->EvaluatePosition(x, closestPoint, subId, pcoords, minDist2, weights);
  svtkQuadraticPolygon::PermuteFromPolygon(this->GetNumberOfPoints(), weights);
  return result;
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::EvaluateLocation(
  int& subId, const double pcoords[3], double x[3], double* weights)
{
  this->InitializePolygon();
  this->Polygon->EvaluateLocation(subId, pcoords, x, weights);
  svtkQuadraticPolygon::PermuteFromPolygon(this->GetNumberOfPoints(), weights);
}

//----------------------------------------------------------------------------
int svtkQuadraticPolygon::CellBoundary(int subId, const double pcoords[3], svtkIdList* pts)
{
  this->InitializePolygon();
  return this->Polygon->CellBoundary(subId, pcoords, pts);
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::Contour(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* verts, svtkCellArray* lines,
  svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
  svtkCellData* outCd)
{
  this->InitializePolygon();

  svtkDataArray* convertedCellScalars = cellScalars->NewInstance();
  svtkQuadraticPolygon::PermuteToPolygon(cellScalars, convertedCellScalars);

  this->Polygon->Contour(
    value, convertedCellScalars, locator, verts, lines, polys, inPd, outPd, inCd, cellId, outCd);

  convertedCellScalars->Delete();
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::Clip(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd,
  svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd, int insideOut)
{
  this->InitializePolygon();

  svtkDataArray* convertedCellScalars = cellScalars->NewInstance();
  svtkQuadraticPolygon::PermuteToPolygon(cellScalars, convertedCellScalars);

  this->Polygon->Clip(
    value, convertedCellScalars, locator, polys, inPd, outPd, inCd, cellId, outCd, insideOut);

  convertedCellScalars->Delete();
}

//----------------------------------------------------------------------------
int svtkQuadraticPolygon::IntersectWithLine(
  const double* p1, const double* p2, double tol, double& t, double* x, double* pcoords, int& subId)
{
  this->InitializePolygon();
  return this->Polygon->IntersectWithLine(p1, p2, tol, t, x, pcoords, subId);
}

//----------------------------------------------------------------------------
int svtkQuadraticPolygon::Triangulate(svtkIdList* outTris)
{
  this->InitializePolygon();
  int result = this->Polygon->Triangulate(outTris);
  svtkQuadraticPolygon::ConvertFromPolygon(outTris);
  return result;
}

//----------------------------------------------------------------------------
int svtkQuadraticPolygon::Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts)
{
  this->InitializePolygon();
  return this->Polygon->Triangulate(index, ptIds, pts);
}

//----------------------------------------------------------------------------
int svtkQuadraticPolygon::NonDegenerateTriangulate(svtkIdList* outTris)
{
  this->InitializePolygon();
  int result = this->Polygon->NonDegenerateTriangulate(outTris);
  svtkQuadraticPolygon::ConvertFromPolygon(outTris);
  return result;
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::InterpolateFunctions(const double x[3], double* weights)
{
  this->InitializePolygon();
  this->Polygon->SetUseMVCInterpolation(UseMVCInterpolation);
  this->Polygon->InterpolateFunctions(x, weights);
  svtkQuadraticPolygon::PermuteFromPolygon(this->GetNumberOfPoints(), weights);
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "UseMVCInterpolation: " << this->UseMVCInterpolation << "\n";
  os << indent << "Edge:\n";
  this->Edge->PrintSelf(os, indent.GetNextIndent());
  os << indent << "Polygon:\n";
  this->Polygon->PrintSelf(os, indent.GetNextIndent());
}

//----------------------------------------------------------------------------
double svtkQuadraticPolygon::DistanceToPolygon(
  double x[3], int numPts, double* pts, double bounds[6], double closest[3])
{
  double* convertedPts = new double[numPts * 3];
  svtkQuadraticPolygon::PermuteToPolygon(numPts, pts, convertedPts);

  double result = svtkPolygon::DistanceToPolygon(x, numPts, convertedPts, bounds, closest);

  delete[] convertedPts;

  return result;
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::ComputeCentroid(svtkIdTypeArray* ids, svtkPoints* p, double c[3])
{
  svtkPoints* convertedPts = svtkPoints::New();
  svtkQuadraticPolygon::PermuteToPolygon(p, convertedPts);

  svtkIdTypeArray* convertedIds = svtkIdTypeArray::New();
  svtkQuadraticPolygon::PermuteToPolygon(ids, convertedIds);

  svtkPolygon::ComputeCentroid(convertedIds, convertedPts, c);

  convertedPts->Delete();
  convertedIds->Delete();
}

//----------------------------------------------------------------------------
int svtkQuadraticPolygon::ParameterizePolygon(
  double* p0, double* p10, double& l10, double* p20, double& l20, double* n)
{
  this->InitializePolygon();
  return this->Polygon->ParameterizePolygon(p0, p10, l10, p20, l20, n);
}

//----------------------------------------------------------------------------
int svtkQuadraticPolygon::IntersectPolygonWithPolygon(int npts, double* pts, double bounds[6],
  int npts2, double* pts2, double bounds2[6], double tol2, double x[3])
{
  double* convertedPts = new double[npts * 3];
  svtkQuadraticPolygon::PermuteToPolygon(npts, pts, convertedPts);

  double* convertedPts2 = new double[npts2 * 3];
  svtkQuadraticPolygon::PermuteToPolygon(npts2, pts2, convertedPts2);

  int result = svtkPolygon::IntersectPolygonWithPolygon(
    npts, convertedPts, bounds, npts2, convertedPts2, bounds2, tol2, x);

  delete[] convertedPts;
  delete[] convertedPts2;

  return result;
}

//----------------------------------------------------------------------------
int svtkQuadraticPolygon::IntersectConvex2DCells(
  svtkCell* cell1, svtkCell* cell2, double tol, double p0[3], double p1[3])
{
  svtkPolygon* convertedCell1 = nullptr;
  svtkPolygon* convertedCell2 = nullptr;

  svtkQuadraticPolygon* qp1 = dynamic_cast<svtkQuadraticPolygon*>(cell1);
  if (qp1)
  {
    convertedCell1 = svtkPolygon::New();
    svtkQuadraticPolygon::PermuteToPolygon(cell1, convertedCell1);
  }

  svtkQuadraticPolygon* qp2 = dynamic_cast<svtkQuadraticPolygon*>(cell2);
  if (qp2)
  {
    convertedCell2 = svtkPolygon::New();
    svtkQuadraticPolygon::PermuteToPolygon(cell2, convertedCell2);
  }

  int result = svtkPolygon::IntersectConvex2DCells((convertedCell1 ? convertedCell1 : cell1),
    (convertedCell2 ? convertedCell2 : cell2), tol, p0, p1);

  if (convertedCell1)
  {
    convertedCell1->Delete();
  }
  if (convertedCell2)
  {
    convertedCell2->Delete();
  }

  return result;
}

//----------------------------------------------------------------------------
int svtkQuadraticPolygon::PointInPolygon(
  double x[3], int numPts, double* pts, double bounds[6], double* n)
{
  double* convertedPts = new double[numPts * 3];
  svtkQuadraticPolygon::PermuteToPolygon(numPts, pts, convertedPts);

  int result = svtkPolygon::PointInPolygon(x, numPts, convertedPts, bounds, n);

  delete[] convertedPts;

  return result;
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::GetPermutationFromPolygon(svtkIdType nb, svtkIdList* permutation)
{
  permutation->SetNumberOfIds(nb);
  for (svtkIdType i = 0; i < nb; i++)
  {
    permutation->SetId(i, ((i % 2) ? (i + nb) / 2 : i / 2));
  }
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::PermuteToPolygon(svtkIdType nbPoints, double* inPoints, double* outPoints)
{
  svtkIdList* permutation = svtkIdList::New();
  svtkQuadraticPolygon::GetPermutationFromPolygon(nbPoints, permutation);

  for (svtkIdType i = 0; i < nbPoints; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      outPoints[3 * i + j] = inPoints[3 * permutation->GetId(i) + j];
    }
  }

  permutation->Delete();
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::PermuteToPolygon(svtkPoints* inPoints, svtkPoints* outPoints)
{
  svtkIdType nbPoints = inPoints->GetNumberOfPoints();

  svtkIdList* permutation = svtkIdList::New();
  svtkQuadraticPolygon::GetPermutationFromPolygon(nbPoints, permutation);

  outPoints->SetNumberOfPoints(nbPoints);
  for (svtkIdType i = 0; i < nbPoints; i++)
  {
    outPoints->SetPoint(i, inPoints->GetPoint(permutation->GetId(i)));
  }

  permutation->Delete();
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::PermuteToPolygon(svtkIdTypeArray* inIds, svtkIdTypeArray* outIds)
{
  svtkIdType nbIds = inIds->GetNumberOfTuples();

  svtkIdList* permutation = svtkIdList::New();
  svtkQuadraticPolygon::GetPermutationFromPolygon(nbIds, permutation);

  outIds->SetNumberOfTuples(nbIds);
  for (svtkIdType i = 0; i < nbIds; i++)
  {
    outIds->SetValue(i, inIds->GetValue(permutation->GetId(i)));
  }

  permutation->Delete();
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::PermuteToPolygon(svtkDataArray* inDataArray, svtkDataArray* outDataArray)
{
  svtkIdType nb = inDataArray->GetNumberOfTuples();

  svtkIdList* permutation = svtkIdList::New();
  svtkQuadraticPolygon::GetPermutationFromPolygon(nb, permutation);

  outDataArray->SetNumberOfComponents(inDataArray->GetNumberOfComponents());
  outDataArray->SetNumberOfTuples(nb);
  inDataArray->GetTuples(permutation, outDataArray);

  permutation->Delete();
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::PermuteToPolygon(svtkCell* inCell, svtkCell* outCell)
{
  svtkIdType nbPoints = inCell->GetNumberOfPoints();

  svtkIdList* permutation = svtkIdList::New();
  svtkQuadraticPolygon::GetPermutationFromPolygon(nbPoints, permutation);

  outCell->Points->SetNumberOfPoints(nbPoints);
  outCell->PointIds->SetNumberOfIds(nbPoints);

  for (svtkIdType i = 0; i < nbPoints; i++)
  {
    outCell->PointIds->SetId(i, inCell->PointIds->GetId(permutation->GetId(i)));
    outCell->Points->SetPoint(i, inCell->Points->GetPoint(permutation->GetId(i)));
  }

  permutation->Delete();
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::InitializePolygon()
{
  svtkQuadraticPolygon::PermuteToPolygon(this, this->Polygon);
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::GetPermutationToPolygon(svtkIdType nb, svtkIdList* permutation)
{
  permutation->SetNumberOfIds(nb);
  for (svtkIdType i = 0; i < nb; i++)
  {
    permutation->SetId(i, (i < nb / 2) ? (i * 2) : (i * 2 + 1 - nb));
  }
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::PermuteFromPolygon(svtkIdType nb, double* values)
{
  svtkIdList* permutation = svtkIdList::New();
  svtkQuadraticPolygon::GetPermutationToPolygon(nb, permutation);

  double* save = new double[nb];
  for (svtkIdType i = 0; i < nb; i++)
  {
    save[i] = values[i];
  }
  for (svtkIdType i = 0; i < nb; i++)
  {
    values[i] = save[permutation->GetId(i)];
  }

  permutation->Delete();
  delete[] save;
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::ConvertFromPolygon(svtkIdList* ids)
{
  svtkIdType nbIds = ids->GetNumberOfIds();

  svtkIdList* permutation = svtkIdList::New();
  svtkQuadraticPolygon::GetPermutationFromPolygon(nbIds, permutation);

  svtkIdList* saveList = svtkIdList::New();
  saveList->SetNumberOfIds(nbIds);
  ids->SetNumberOfIds(nbIds);

  for (svtkIdType i = 0; i < nbIds; i++)
  {
    saveList->SetId(i, ids->GetId(i));
  }
  for (svtkIdType i = 0; i < nbIds; i++)
  {
    ids->SetId(i, permutation->GetId(saveList->GetId(i)));
  }

  permutation->Delete();
  saveList->Delete();
}

//----------------------------------------------------------------------------
void svtkQuadraticPolygon::Derivatives(int svtkNotUsed(subId), const double svtkNotUsed(pcoords)[3],
  const double* svtkNotUsed(values), int svtkNotUsed(dim), double* svtkNotUsed(derivs))
{
}
