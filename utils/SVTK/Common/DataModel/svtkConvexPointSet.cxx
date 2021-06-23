/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkConvexPointSet.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkConvexPointSet.h"

#include "svtkCellArray.h"
#include "svtkCellArrayIterator.h"
#include "svtkDoubleArray.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkOrderedTriangulator.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkTetra.h"
#include "svtkTriangle.h"

svtkStandardNewMacro(svtkConvexPointSet);

//----------------------------------------------------------------------------
// Construct the hexahedron with eight points.
svtkConvexPointSet::svtkConvexPointSet()
{
  this->Tetra = svtkTetra::New();
  this->TetraIds = svtkIdList::New();
  this->TetraPoints = svtkPoints::New();
  this->TetraScalars = svtkDoubleArray::New();
  this->TetraScalars->SetNumberOfTuples(4);
  this->BoundaryTris = svtkCellArray::New();
  this->BoundaryTris->AllocateEstimate(128, 3);
  this->Triangle = svtkTriangle::New();
  this->Triangulator = svtkOrderedTriangulator::New();
  this->Triangulator->PreSortedOff();
  this->Triangulator->UseTemplatesOff();
  this->ParametricCoords = nullptr;
}

//----------------------------------------------------------------------------
svtkConvexPointSet::~svtkConvexPointSet()
{
  this->Tetra->Delete();
  this->TetraIds->Delete();
  this->TetraPoints->Delete();
  this->TetraScalars->Delete();
  this->BoundaryTris->Delete();
  this->Triangle->Delete();
  if (this->ParametricCoords)
  {
    this->ParametricCoords->Delete();
  }
}

//----------------------------------------------------------------------------
// Should be called by GetCell() prior to any other method invocation
void svtkConvexPointSet::Initialize()
{
  // Initialize
  svtkIdType numPts = this->GetNumberOfPoints();
  if (numPts < 1)
    return;

  this->Triangulate(0, this->TetraIds, this->TetraPoints);
}

//----------------------------------------------------------------------------
int svtkConvexPointSet::GetNumberOfFaces()
{
  this->BoundaryTris->Reset();
  this->Triangulator->AddTriangles(this->BoundaryTris);
  return this->BoundaryTris->GetNumberOfCells();
}

//----------------------------------------------------------------------------
svtkCell* svtkConvexPointSet::GetFace(int faceId)
{
  int numCells = this->BoundaryTris->GetNumberOfCells();
  if (faceId < 0 || faceId >= numCells)
  {
    return nullptr;
  }

  svtkIdType numPts;
  const svtkIdType* cptr;
  this->BoundaryTris->GetCellAtId(faceId, numPts, cptr);
  assert(numPts == 3);

  // Each triangle has three points plus number of points
  for (int i = 0; i < 3; i++)
  {
    this->Triangle->PointIds->SetId(i, this->PointIds->GetId(cptr[i]));
    this->Triangle->Points->SetPoint(i, this->Points->GetPoint(cptr[i]));
  }

  return this->Triangle;
}

//----------------------------------------------------------------------------
int svtkConvexPointSet::Triangulate(int svtkNotUsed(index), svtkIdList* ptIds, svtkPoints* pts)
{
  svtkIdType numPts = this->GetNumberOfPoints();
  double x[3];
  svtkIdType ptId;

  // Initialize
  ptIds->Reset();
  pts->Reset();
  if (numPts < 1)
  {
    return 0;
  }

  // Initialize Delaunay insertion process.
  // No more than numPts points can be inserted.
  this->Triangulator->InitTriangulation(this->GetBounds(), numPts);

  // Inject cell points into triangulation. Recall that the PreSortedOff()
  // flag was set which means that the triangulator will order the points
  // according to point id. We insert points with id == the index into the
  // svtkConvexPointSet::PointIds and Points; but sort on the global point
  // id.
  for (svtkIdType i = 0; i < numPts; i++)
  {
    ptId = this->PointIds->GetId(i);
    this->Points->GetPoint(i, x);
    this->Triangulator->InsertPoint(i, ptId, x, x, 0);
  } // for all points

  // triangulate the points
  this->Triangulator->Triangulate();

  // Add the triangulation to the mesh
  this->Triangulator->AddTetras(0, ptIds, pts);

  return 1;
}

//----------------------------------------------------------------------------
void svtkConvexPointSet::Contour(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* verts, svtkCellArray* lines,
  svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
  svtkCellData* outCd)
{
  // For each tetra, contour it
  svtkIdType ptId, localId;
  svtkIdType numTets = this->TetraIds->GetNumberOfIds() / 4;
  for (svtkIdType i = 0; i < numTets; i++)
  {
    for (svtkIdType j = 0; j < 4; j++)
    {
      localId = this->TetraIds->GetId(4 * i + j);
      ptId = this->PointIds->GetId(localId);
      this->Tetra->PointIds->SetId(j, ptId);
      this->Tetra->Points->SetPoint(j, this->TetraPoints->GetPoint(4 * i + j));
      this->TetraScalars->SetValue(j, cellScalars->GetTuple1(localId));
    }
    this->Tetra->Contour(
      value, this->TetraScalars, locator, verts, lines, polys, inPd, outPd, inCd, cellId, outCd);
  }
}

//----------------------------------------------------------------------------
void svtkConvexPointSet::Clip(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* tets, svtkPointData* inPD, svtkPointData* outPD,
  svtkCellData* inCD, svtkIdType cellId, svtkCellData* outCD, int insideOut)
{
  // For each tetra, contour it
  int i, j;
  svtkIdType ptId, localId;
  int numTets = this->TetraIds->GetNumberOfIds() / 4;
  for (i = 0; i < numTets; i++)
  {
    for (j = 0; j < 4; j++)
    {
      localId = this->TetraIds->GetId(4 * i + j);
      ptId = this->PointIds->GetId(localId);
      this->Tetra->PointIds->SetId(j, ptId);
      this->Tetra->Points->SetPoint(j, this->TetraPoints->GetPoint(4 * i + j));
      this->TetraScalars->SetValue(j, cellScalars->GetTuple1(localId));
    }
    this->Tetra->Clip(
      value, this->TetraScalars, locator, tets, inPD, outPD, inCD, cellId, outCD, insideOut);
  }
}

//----------------------------------------------------------------------------
int svtkConvexPointSet::CellBoundary(int subId, const double pcoords[3], svtkIdList* pts)
{
  // This function was re-written to fix bug #9550.
  // Thanks go to Bart Janssens.
  svtkIdType pntIndx;
  for (int i = 0; i < 4; i++)
  {
    pntIndx = this->PointIds->GetId(this->TetraIds->GetId((subId << 2) + i));
    this->Tetra->PointIds->SetId(i, pntIndx);
    this->Tetra->Points->SetPoint(i, this->TetraPoints->GetPoint((subId << 2) + i));
  }

  // find the parametrically nearest triangle.
  return this->Tetra->CellBoundary(subId, pcoords, pts);
}

//----------------------------------------------------------------------------
int svtkConvexPointSet::EvaluatePosition(const double x[3], double svtkNotUsed(closestPoint)[3],
  int& subId, double pcoords[3], double& minDist2, double weights[])
{
  double pc[3], dist2;
  int ignoreId, i, j, k, returnStatus = 0, status;
  double tempWeights[4];
  double closest[3];
  svtkIdType ptId;
  int numPnts = this->GetNumberOfPoints();
  int numTets = this->TetraIds->GetNumberOfIds() >> 2;

  for (minDist2 = SVTK_DOUBLE_MAX, i = 0; i < numTets; i++)
  {
    for (j = 0; j < 4; j++)
    {
      ptId = this->PointIds->GetId(this->TetraIds->GetId((i << 2) + j));
      this->Tetra->PointIds->SetId(j, ptId);
      this->Tetra->Points->SetPoint(j, this->TetraPoints->GetPoint((i << 2) + j));
    }

    status = this->Tetra->EvaluatePosition(x, closest, ignoreId, pc, dist2, tempWeights);
    if (status != -1 && dist2 < minDist2)
    {
      // init (clear) all the weights since only the vertices of the closest
      // tetrahedron are assigned with valid weights while the rest vertices
      // (of those farther tetrahedra) are simply inited with zero weights
      // (to make no any contribution). This fixes bug #9453
      for (k = 0; k < numPnts; k++)
      {
        weights[k] = 0.0;
      }

      returnStatus = status;
      minDist2 = dist2;
      subId = i;
      pcoords[0] = pc[0];
      pcoords[1] = pc[1];
      pcoords[2] = pc[2];

      // assign valid weights to the vertices of this closest tetrahedron only
      // This fixes bug #9453.
      weights[this->TetraIds->GetId((i << 2))] = tempWeights[0];
      weights[this->TetraIds->GetId((i << 2) + 1)] = tempWeights[1];
      weights[this->TetraIds->GetId((i << 2) + 2)] = tempWeights[2];
      weights[this->TetraIds->GetId((i << 2) + 3)] = tempWeights[3];
    }
  }

  return returnStatus;
}

//----------------------------------------------------------------------------
void svtkConvexPointSet::EvaluateLocation(
  int& subId, const double pcoords[3], double x[3], double* weights)
{
  int i;
  int numPnts;
  double tmpWgts[4];
  svtkIdType pntIndx;

  for (i = 0; i < 4; i++)
  {
    pntIndx = this->PointIds->GetId(this->TetraIds->GetId((subId << 2) + i));
    this->Tetra->PointIds->SetId(i, pntIndx);
    this->Tetra->Points->SetPoint(i, this->TetraPoints->GetPoint((subId << 2) + i));
  }

  // use tmpWgts to collect the valid weights of the tetra's four vertices
  this->Tetra->EvaluateLocation(subId, pcoords, x, tmpWgts);

  // init the actual array of weights (possibly greater than 4)
  numPnts = this->GetNumberOfPoints();
  for (i = 0; i < numPnts; i++)
  {
    weights[i] = 0.0;
  }

  // update the target weights only
  weights[this->TetraIds->GetId((subId << 2))] = tmpWgts[0];
  weights[this->TetraIds->GetId((subId << 2) + 1)] = tmpWgts[1];
  weights[this->TetraIds->GetId((subId << 2) + 2)] = tmpWgts[2];
  weights[this->TetraIds->GetId((subId << 2) + 3)] = tmpWgts[3];
}

//----------------------------------------------------------------------------
int svtkConvexPointSet::IntersectWithLine(const double p1[3], const double p2[3], double tol,
  double& minT, double x[3], double pcoords[3], int& subId)
{
  int subTest, i, j;
  svtkIdType ptId;
  double t, pc[3], xTemp[3];

  int numTets = this->TetraIds->GetNumberOfIds() / 4;
  int status = 0;

  for (minT = SVTK_DOUBLE_MAX, i = 0; i < numTets; i++)
  {
    for (j = 0; j < 4; j++)
    {
      ptId = this->PointIds->GetId(this->TetraIds->GetId(4 * i + j));
      this->Tetra->PointIds->SetId(j, ptId);
      this->Tetra->Points->SetPoint(j, this->TetraPoints->GetPoint(4 * i + j));
    }

    if (this->Tetra->IntersectWithLine(p1, p2, tol, t, xTemp, pc, subTest) && t < minT)
    {
      status = 1;
      subId = i;
      minT = t;
      x[0] = xTemp[0];
      x[1] = xTemp[1];
      x[2] = xTemp[2];
      pcoords[0] = pc[0];
      pcoords[1] = pc[1];
      pcoords[2] = pc[2];
    }
  }

  return status;
}

//----------------------------------------------------------------------------
void svtkConvexPointSet::Derivatives(
  int subId, const double pcoords[3], const double* values, int dim, double* derivs)
{
  svtkIdType ptId;

  for (int j = 0; j < 4; j++)
  {
    ptId = this->PointIds->GetId(this->TetraIds->GetId(4 * subId + j));
    this->Tetra->PointIds->SetId(j, ptId);
    this->Tetra->Points->SetPoint(j, this->TetraPoints->GetPoint(4 * subId + j));
  }

  this->Tetra->Derivatives(subId, pcoords, values, dim, derivs);
}

//----------------------------------------------------------------------------
double* svtkConvexPointSet::GetParametricCoords()
{
  int numPts = this->PointIds->GetNumberOfIds();
  if (!this->ParametricCoords)
  {
    this->ParametricCoords = svtkDoubleArray::New();
  }

  this->ParametricCoords->SetNumberOfComponents(3);
  this->ParametricCoords->SetNumberOfTuples(numPts);
  double p[3], x[3];
  const double* bounds = this->GetBounds();
  int i, j;
  for (i = 0; i < numPts; i++)
  {
    this->Points->GetPoint(i, x);
    for (j = 0; j < 3; j++)
    {
      p[j] = (x[j] - bounds[2 * j]) / (bounds[2 * j + 1] - bounds[2 * j]);
    }
    this->ParametricCoords->SetTuple(i, p);
  }

  return this->ParametricCoords->GetPointer(0);
}

//----------------------------------------------------------------------------
void svtkConvexPointSet::InterpolateFunctions(const double pcoords[3], double* sf)
{
  (void)pcoords;
  (void)sf;
}

//----------------------------------------------------------------------------
void svtkConvexPointSet::InterpolateDerivs(const double pcoords[3], double* derivs)
{
  (void)pcoords;
  (void)derivs;
}

//----------------------------------------------------------------------------
void svtkConvexPointSet::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Tetra:\n";
  this->Tetra->PrintSelf(os, indent.GetNextIndent());
  os << indent << "TetraIds:\n";
  this->TetraIds->PrintSelf(os, indent.GetNextIndent());
  os << indent << "TetraPoints:\n";
  this->TetraPoints->PrintSelf(os, indent.GetNextIndent());
  os << indent << "TetraScalars:\n";
  this->TetraScalars->PrintSelf(os, indent.GetNextIndent());

  os << indent << "BoundaryTris:\n";
  this->BoundaryTris->PrintSelf(os, indent.GetNextIndent());
  os << indent << "Triangle:\n";
  this->Triangle->PrintSelf(os, indent.GetNextIndent());
  if (this->ParametricCoords)
  {
    os << indent << "ParametricCoords " << this->ParametricCoords << "\n";
  }
  else
  {
    os << indent << "ParametricCoords: (null)\n";
  }
}
