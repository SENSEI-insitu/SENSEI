/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHigherOrderTetra.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkHigherOrderTetra.h"

#include "svtkDoubleArray.h"
#include "svtkHigherOrderCurve.h"
#include "svtkHigherOrderTriangle.h"
#include "svtkLine.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkTetra.h"
#include "svtkType.h"

#define ENABLE_CACHING
#define FIFTEEN_POINT_TETRA

namespace
{
// The linearized tetra is comprised of four linearized faces. Each face is
// comprised of three vertices. These must be consistent with svtkTetra.
/*
  static constexpr svtkIdType FaceVertices[4][3] = {{0,1,3}, {1,2,3},
                                               {2,0,3}, {0,2,1}};
*/

// The linearized tetra is comprised of six linearized edges. Each edge is
// comprised of two vertices. These must be consistent with svtkTetra.
static constexpr svtkIdType EdgeVertices[6][2] = { { 0, 1 }, { 1, 2 }, { 2, 0 }, { 0, 3 }, { 1, 3 },
  { 2, 3 } };

// The barycentric coordinates of the four vertices of the linear tetra.
static constexpr svtkIdType LinearVertices[4][4] = { { 0, 0, 0, 1 }, { 1, 0, 0, 0 }, { 0, 1, 0, 0 },
  { 0, 0, 1, 0 } };

// When describing a linearized tetra face, there is a mapping between the
// four-component barycentric tetra system and the three-component barycentric
// triangle system. These are the relevant indices within the four-component
// system for each face (e.g. face 0 varies across the barycentric tetra
// coordinates 0, 2 and 3).
static constexpr svtkIdType FaceBCoords[4][3] = { { 0, 2, 3 }, { 2, 0, 1 }, { 2, 1, 3 },
  { 1, 0, 3 } };

// When describing a linearized tetra face, there is a mapping between the
// four-component barycentric tetra system and the three-component barycentric
// triangle system. These are the constant indices within the four-component
// system for each face (e.g. face 0 holds barycentric tetra coordinate 1
// constant).
static constexpr svtkIdType FaceMinCoord[4] = { 1, 3, 0, 2 };

// Each linearized tetra edge holds two barycentric tetra coordinates constant
// and varies the other two. These are the coordinates that are held constant
// for each edge.
static constexpr svtkIdType EdgeMinCoords[6][2] = { { 1, 2 }, { 2, 3 }, { 0, 2 }, { 0, 1 }, { 1, 3 },
  { 0, 3 } };

// The coordinate that increments when traversing an edge (i.e. the coordinate
// of the nonzero component of the second vertex of the edge).
static constexpr svtkIdType EdgeCountingCoord[6] = { 0, 1, 3, 2, 2, 2 };

// When a linearized tetra vertex is cast into barycentric coordinates, one of
// its coordinates is maximal and the other three are minimal. These are the
// indices of the maximal barycentric coordinate for each vertex.
static constexpr svtkIdType VertexMaxCoords[4] = { 3, 0, 1, 2 };

// There are three different layouts for breaking an octahedron into four
// tetras. given the six vertices of the octahedron, these are the layouts for
// each of the three four-tetra configurations.
static constexpr svtkIdType LinearTetras[3][4][4] = { { { 2, 0, 1, 4 }, { 2, 1, 5, 4 },
                                                       { 2, 5, 3, 4 }, { 2, 3, 0, 4 } },
  { { 0, 4, 1, 5 }, { 0, 1, 2, 5 }, { 0, 2, 3, 5 }, { 0, 3, 4, 5 } },
  { { 1, 5, 2, 3 }, { 1, 2, 0, 3 }, { 1, 0, 4, 3 }, { 1, 4, 5, 3 } } };

#ifdef FIFTEEN_POINT_TETRA
double FifteenPointTetraCoords[15 * 3] = { 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., .5, 0.,
  0., .5, .5, 0., 0., .5, 0., 0., 0., .5, .5, 0., .5, 0., .5, .5, 1. / 3., 1. / 3., 0., 1. / 3., 0.,
  1. / 3., 1. / 3., 1. / 3, 1. / 3., 0., 1. / 3., 1. / 3., .25, .25, .25 };

static constexpr svtkIdType FifteenPointTetraSubtetras[28][4] = { { 0, 4, 10, 14 }, { 1, 4, 10, 14 },
  { 1, 5, 10, 14 }, { 2, 5, 10, 14 }, { 2, 6, 10, 14 }, { 0, 6, 10, 14 }, { 0, 7, 11, 14 },
  { 3, 7, 11, 14 }, { 3, 8, 11, 14 }, { 1, 8, 11, 14 }, { 1, 4, 11, 14 }, { 0, 4, 11, 14 },
  { 1, 5, 12, 14 }, { 2, 5, 12, 14 }, { 2, 9, 12, 14 }, { 3, 9, 12, 14 }, { 3, 8, 12, 14 },
  { 1, 8, 12, 14 }, { 0, 7, 13, 14 }, { 3, 7, 13, 14 }, { 3, 9, 13, 14 }, { 2, 9, 13, 14 },
  { 2, 6, 13, 14 }, { 0, 6, 13, 14 } };
#endif
}

//----------------------------------------------------------------------------
svtkHigherOrderTetra::svtkHigherOrderTetra()
{
  this->Order = 0;

  this->Tetra = svtkTetra::New();
  this->Scalars = svtkDoubleArray::New();
  this->Scalars->SetNumberOfTuples(4);

  this->Points->SetNumberOfPoints(4);
  this->PointIds->SetNumberOfIds(4);
  for (svtkIdType i = 0; i < 4; i++)
  {
    this->Points->SetPoint(i, 0.0, 0.0, 0.0);
    this->PointIds->SetId(i, 0);
  }
}

//----------------------------------------------------------------------------
svtkHigherOrderTetra::~svtkHigherOrderTetra()
{
  this->Tetra->Delete();
  this->Scalars->Delete();
}

//------------------------------------------------------------------------------
void svtkHigherOrderTetra::SetEdgeIdsAndPoints(int edgeId,
  const std::function<void(const svtkIdType&)>& set_number_of_ids_and_points,
  const std::function<void(const svtkIdType&, const svtkIdType&)>& set_ids_and_points)
{
  svtkIdType order = this->GetOrder();

  set_number_of_ids_and_points(order + 1);

  svtkIdType bindex[4] = { 0, 0, 0, 0 };
  bindex[EdgeVertices[edgeId][0]] = order;
  for (svtkIdType i = 0; i <= order; i++)
  {
    set_ids_and_points(i, this->ToIndex(bindex));
    bindex[EdgeVertices[edgeId][0]]--;
    bindex[EdgeVertices[edgeId][1]]++;
  }
}

//------------------------------------------------------------------------------
void svtkHigherOrderTetra::SetFaceIdsAndPoints(svtkHigherOrderTriangle* result, int faceId,
  const std::function<void(const svtkIdType&)>& set_number_of_ids_and_points,
  const std::function<void(const svtkIdType&, const svtkIdType&)>& set_ids_and_points)
{
  assert(faceId >= 0 && faceId < 4);

  svtkIdType order = this->GetOrder();

  svtkIdType nPoints = (order + 1) * (order + 2) / 2;

#ifdef FIFTEEN_POINT_TETRA
  if (this->Points->GetNumberOfPoints() == 15)
  {
    nPoints = 7;
  }
#endif
  set_number_of_ids_and_points(nPoints);

  svtkIdType tetBCoords[4], triBCoords[3];
  for (svtkIdType p = 0; p < nPoints; p++)
  {
    svtkHigherOrderTriangle::BarycentricIndex(p, triBCoords, order);

    for (svtkIdType coord = 0; coord < 3; coord++)
    {
      tetBCoords[FaceBCoords[faceId][coord]] = triBCoords[coord];
    }
    tetBCoords[FaceMinCoord[faceId]] = 0;

    svtkIdType pointIndex = svtkHigherOrderTetra::Index(tetBCoords, order);
    set_ids_and_points(p, pointIndex);
  }

#ifdef FIFTEEN_POINT_TETRA
  if (this->Points->GetNumberOfPoints() == 15)
  {
    svtkIdType pointIndex = 10 + ((faceId + 1) % 4);
    set_ids_and_points(6, pointIndex);
  }
#endif

  result->Initialize();
}

//----------------------------------------------------------------------------
void svtkHigherOrderTetra::Initialize()
{
  svtkIdType order = this->ComputeOrder();

  if (this->Order != order)
  {
    // Reset our caches
    this->Order = order;

    this->NumberOfSubtetras = this->ComputeNumberOfSubtetras();

    EdgeIds.resize(this->Order + 1);

#ifdef ENABLE_CACHING
    this->BarycentricIndexMap.resize(4 * this->GetPointIds()->GetNumberOfIds());
    for (svtkIdType i = 0; i < this->GetPointIds()->GetNumberOfIds(); i++)
    {
      this->BarycentricIndexMap[4 * i] = -1;
    }

    // we sacrifice memory for efficiency here
    svtkIdType nIndexMap = (this->Order + 1) * (this->Order + 1) * (this->Order + 1);
    this->IndexMap.resize(nIndexMap);
    for (svtkIdType i = 0; i < nIndexMap; i++)
    {
      this->IndexMap[i] = -1;
    }

    svtkIdType nSubtetras = this->GetNumberOfSubtetras();
    this->SubtetraIndexMap.resize(16 * nSubtetras);
    for (svtkIdType i = 0; i < nSubtetras; i++)
    {
      this->SubtetraIndexMap[16 * i] = -1;
    }
#endif
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkHigherOrderTetra::ComputeNumberOfSubtetras()
{
#ifdef FIFTEEN_POINT_TETRA
  if (this->Points->GetNumberOfPoints() == 15)
  {
    return 28;
  }
#endif
  svtkIdType order = this->GetOrder();

  // # of rightside-up tetras: order*(order+1)*(order+2)/6
  // # of octahedra: (order-1)*order*(order+1)/6
  // # of upside-down tetras: (order-2)*(order-1)*order/6

  svtkIdType nRightSideUp = order * (order + 1) * (order + 2) / 6;
  svtkIdType nOctahedra = (order - 1) * order * (order + 1) / 6;
  svtkIdType nUpsideDown = (order > 2 ? (order - 2) * (order - 1) * order / 6 : 0);

  return nRightSideUp + 4 * nOctahedra + nUpsideDown;
}

//----------------------------------------------------------------------------
void svtkHigherOrderTetra::SubtetraBarycentricPointIndices(
  svtkIdType cellIndex, svtkIdType (&pointBIndices)[4][4])
{
  // We tesselllate a tetrahedron into a tetrahedral-octahedral honeycomb, and
  // then discretize each octahedron into 4 tetrahedra. The pattern is as
  // follows: for each additional level in our tetrahedron (propagating
  // downwards in parametric z), a pattern of upside-down and rightside-up
  // triangles are formed. The rightside-up triangles form tetrahedra with the
  // single point above them, and the upside-down triangles form octahedra with
  // the righteside-up triangles above them.

  assert(cellIndex < this->GetNumberOfSubtetras());

#ifdef FIFTEEN_POINT_TETRA
  if (this->Points->GetNumberOfPoints() == 15)
  {
    pointBIndices[0][0] = FifteenPointTetraSubtetras[cellIndex][0];
    pointBIndices[1][0] = FifteenPointTetraSubtetras[cellIndex][1];
    pointBIndices[2][0] = FifteenPointTetraSubtetras[cellIndex][2];
    pointBIndices[3][0] = FifteenPointTetraSubtetras[cellIndex][3];
    return;
  }
#endif

#ifdef ENABLE_CACHING
  svtkIdType cellIndexStart = cellIndex * 16;
  if (this->SubtetraIndexMap[cellIndexStart] == -1)
#endif
  {
    svtkIdType order = this->GetOrder();

    if (order == 1)
    {
      for (svtkIdType i = 0; i < 4; i++)
      {
        for (svtkIdType j = 0; j < 4; j++)
        {
          pointBIndices[i][j] = LinearVertices[i][j];
        }
      }
    }
    else
    {
      svtkIdType nRightSideUp = order * (order + 1) * (order + 2) / 6;
      svtkIdType nOctahedra = (order - 1) * order * (order + 1) / 6;

      if (cellIndex < nRightSideUp)
      {
        // there are nRightSideUp subtetras whose orientation is the same as the
        // projected tetra. We traverse them here.
        svtkHigherOrderTetra::BarycentricIndex(cellIndex, pointBIndices[0], order - 1);

        pointBIndices[0][3] += 1;

        pointBIndices[1][0] = pointBIndices[0][0];
        pointBIndices[1][1] = pointBIndices[0][1] + 1;
        pointBIndices[1][2] = pointBIndices[0][2];
        pointBIndices[1][3] = pointBIndices[0][3] - 1;

        pointBIndices[3][0] = pointBIndices[0][0] + 1;
        pointBIndices[3][1] = pointBIndices[0][1];
        pointBIndices[3][2] = pointBIndices[0][2];
        pointBIndices[3][3] = pointBIndices[0][3] - 1;

        pointBIndices[2][0] = pointBIndices[0][0];
        pointBIndices[2][1] = pointBIndices[0][1];
        pointBIndices[2][2] = pointBIndices[0][2] + 1;
        pointBIndices[2][3] = pointBIndices[0][3] - 1;
      }
      else if (cellIndex < nRightSideUp + 4 * nOctahedra)
      {
        // the next set of subtetras are embedded in octahedra, so we need to
        // identify and subdivide the octahedra. We traverse them here.
        cellIndex -= nRightSideUp;

        svtkIdType octIndex = cellIndex / 4;
        svtkIdType tetIndex = cellIndex % 4;

        svtkIdType octBIndices[6][4];

        if (order == 2)
        {
          octBIndices[2][0] = octBIndices[2][1] = octBIndices[2][2] = octBIndices[2][3] = 0;
        }
        else
        {
          svtkHigherOrderTetra::BarycentricIndex(octIndex, octBIndices[2], order - 2);
        }
        octBIndices[2][1] += 1;
        octBIndices[2][3] += 1;

        octBIndices[1][0] = octBIndices[2][0] + 1;
        octBIndices[1][1] = octBIndices[2][1];
        octBIndices[1][2] = octBIndices[2][2];
        octBIndices[1][3] = octBIndices[2][3] - 1;

        octBIndices[0][0] = octBIndices[2][0] + 1;
        octBIndices[0][1] = octBIndices[2][1] - 1;
        octBIndices[0][2] = octBIndices[2][2];
        octBIndices[0][3] = octBIndices[2][3];

        octBIndices[3][0] = octBIndices[0][0] - 1;
        octBIndices[3][1] = octBIndices[0][1];
        octBIndices[3][2] = octBIndices[0][2] + 1;
        octBIndices[3][3] = octBIndices[0][3];

        octBIndices[4][0] = octBIndices[3][0] + 1;
        octBIndices[4][1] = octBIndices[3][1];
        octBIndices[4][2] = octBIndices[3][2];
        octBIndices[4][3] = octBIndices[3][3] - 1;

        octBIndices[5][0] = octBIndices[3][0];
        octBIndices[5][1] = octBIndices[3][1] + 1;
        octBIndices[5][2] = octBIndices[3][2];
        octBIndices[5][3] = octBIndices[3][3] - 1;

        this->TetraFromOctahedron(tetIndex, octBIndices, pointBIndices);
      }
      else
      {
        // there are nUpsideDown subtetras whose orientation is inverted w.r.t.
        // the projected tetra. We traverse them here.
        cellIndex -= (nRightSideUp + 4 * nOctahedra);

        if (order == 3)
        {
          pointBIndices[2][0] = pointBIndices[2][1] = pointBIndices[2][2] = pointBIndices[2][3] = 0;
        }
        else
        {
          svtkHigherOrderTetra::BarycentricIndex(cellIndex, pointBIndices[2], order - 3);
        }
        pointBIndices[2][0] += 1;
        pointBIndices[2][1] += 1;
        pointBIndices[2][3] += 1;

        pointBIndices[1][0] = pointBIndices[2][0] - 1;
        pointBIndices[1][1] = pointBIndices[2][1];
        pointBIndices[1][2] = pointBIndices[2][2] + 1;
        pointBIndices[1][3] = pointBIndices[2][3];

        pointBIndices[3][0] = pointBIndices[2][0];
        pointBIndices[3][1] = pointBIndices[2][1] - 1;
        pointBIndices[3][2] = pointBIndices[2][2] + 1;
        pointBIndices[3][3] = pointBIndices[2][3];

        pointBIndices[0][0] = pointBIndices[2][0];
        pointBIndices[0][1] = pointBIndices[2][1];
        pointBIndices[0][2] = pointBIndices[2][2] + 1;
        pointBIndices[0][3] = pointBIndices[2][3] - 1;
      }
    }

#ifdef ENABLE_CACHING
    for (svtkIdType i = 0; i < 4; i++)
    {
      for (svtkIdType j = 0; j < 4; j++)
      {
        this->SubtetraIndexMap[cellIndexStart + 4 * i + j] = pointBIndices[i][j];
      }
    }
#endif
  }
#ifdef ENABLE_CACHING
  else
  {
    for (svtkIdType i = 0; i < 4; i++)
    {
      for (svtkIdType j = 0; j < 4; j++)
      {
        pointBIndices[i][j] = this->SubtetraIndexMap[cellIndexStart + 4 * i + j];
      }
    }
  }
#endif
}

//----------------------------------------------------------------------------
void svtkHigherOrderTetra::TetraFromOctahedron(
  svtkIdType cellIndex, const svtkIdType (&octBIndices)[6][4], svtkIdType (&tetraBIndices)[4][4])
{
  // TODO: intelligently select which of the three linearizations reduce
  // artifacts. For now, we always choose the first linearization.
  static svtkIdType linearization = 0;

  for (svtkIdType i = 0; i < 4; i++)
  {
    for (svtkIdType j = 0; j < 4; j++)
    {
      tetraBIndices[i][j] = octBIndices[LinearTetras[linearization][cellIndex][i]][j];
    }
  }
}

//----------------------------------------------------------------------------
int svtkHigherOrderTetra::CellBoundary(
  int svtkNotUsed(subId), const double pcoords[3], svtkIdList* pts)
{
  const double ijk = 1.0 - pcoords[0] - pcoords[1] - pcoords[2];
  int axis = 3;
  double dmin = ijk;
  for (int ii = 0; ii < 3; ++ii)
  {
    if (dmin > pcoords[ii])
    {
      axis = ii;
      dmin = pcoords[ii];
    }
  }

  const int closestFaceByAxis[4][3] = { { 0, 3, 2 }, { 0, 1, 3 }, { 0, 2, 1 }, { 1, 2, 3 } };

  pts->SetNumberOfIds(3);
  for (int ii = 0; ii < 3; ++ii)
  {
    pts->SetId(ii, this->PointIds->GetId(closestFaceByAxis[axis][ii]));
  }

  return pcoords[0] < 0 || pcoords[0] > 1.0 || pcoords[1] < 0 || pcoords[1] > 1.0 ||
      pcoords[2] < 0 || pcoords[2] > 1.0 || ijk < 0 || ijk > 1.0
    ? 0
    : 1;
}

//----------------------------------------------------------------------------
int svtkHigherOrderTetra::EvaluatePosition(const double x[3], double closestPoint[3], int& subId,
  double pcoords[3], double& minDist2, double weights[])
{
  double pc[3], dist2, tempWeights[4], closest[3];
  double pcoordsMin[3] = { 0., 0., 0. };
  int returnStatus = 0, status, ignoreId;
  svtkIdType minBIndices[4][4], bindices[4][4], pointIndices[4];

  svtkIdType order = this->GetOrder();
  svtkIdType numberOfSubtetras = this->GetNumberOfSubtetras();

  minDist2 = SVTK_DOUBLE_MAX;
  for (svtkIdType subCellId = 0; subCellId < numberOfSubtetras; subCellId++)
  {
    this->SubtetraBarycentricPointIndices(subCellId, bindices);

    for (svtkIdType i = 0; i < 4; i++)
    {
      pointIndices[i] = this->ToIndex(bindices[i]);
      this->Tetra->Points->SetPoint(i, this->Points->GetPoint(pointIndices[i]));
    }

    status = this->Tetra->EvaluatePosition(x, closest, ignoreId, pc, dist2, tempWeights);

    if (status != -1 && dist2 < minDist2)
    {
      returnStatus = status;
      minDist2 = dist2;
      subId = subCellId;
      pcoordsMin[0] = pc[0];
      pcoordsMin[1] = pc[1];
      pcoordsMin[2] = pc[2];
      for (svtkIdType i = 0; i < 4; i++)
      {
        for (svtkIdType j = 0; j < 4; j++)
        {
          minBIndices[i][j] = bindices[i][j];
        }
      }
    }
  }

  // adjust parametric coordinates
  if (returnStatus != -1)
  {
    for (svtkIdType i = 0; i < 3; i++)
    {
      pcoords[i] = (minBIndices[0][i] + pcoordsMin[0] * (minBIndices[1][i] - minBIndices[0][i]) +
                     pcoordsMin[1] * (minBIndices[2][i] - minBIndices[0][i]) +
                     pcoordsMin[2] * (minBIndices[3][i] - minBIndices[0][i])) /
        order;
    }

    if (closestPoint != nullptr)
    {
      // Compute both closestPoint and weights
      this->EvaluateLocation(subId, pcoords, closestPoint, weights);
    }
    else
    {
      // Compute weights only
      this->InterpolateFunctions(pcoords, weights);
    }
  }

  return returnStatus;
}

//----------------------------------------------------------------------------
void svtkHigherOrderTetra::EvaluateLocation(
  int& svtkNotUsed(subId), const double pcoords[3], double x[3], double* weights)
{
  x[0] = x[1] = x[2] = 0.;

  this->InterpolateFunctions(pcoords, weights);

  double p[3];
  svtkIdType nPoints = this->GetPoints()->GetNumberOfPoints();
  for (svtkIdType idx = 0; idx < nPoints; idx++)
  {
    this->Points->GetPoint(idx, p);
    for (svtkIdType jdx = 0; jdx < 3; jdx++)
    {
      x[jdx] += p[jdx] * weights[idx];
    }
  }
}

//----------------------------------------------------------------------------
void svtkHigherOrderTetra::Contour(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* verts, svtkCellArray* lines,
  svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
  svtkCellData* outCd)
{
  svtkIdType bindices[4][4];
  svtkIdType numberOfSubtetras = this->GetNumberOfSubtetras();

  for (svtkIdType subCellId = 0; subCellId < numberOfSubtetras; subCellId++)
  {
    this->SubtetraBarycentricPointIndices(subCellId, bindices);

    for (svtkIdType i = 0; i < 4; i++)
    {
      svtkIdType pointIndex = this->ToIndex(bindices[i]);
      this->Tetra->Points->SetPoint(i, this->Points->GetPoint(pointIndex));
      if (outPd)
      {
        this->Tetra->PointIds->SetId(i, this->PointIds->GetId(pointIndex));
      }
      this->Scalars->SetTuple(i, cellScalars->GetTuple(pointIndex));
    }

    this->Tetra->Contour(
      value, this->Scalars, locator, verts, lines, polys, inPd, outPd, inCd, cellId, outCd);
  }
}

//----------------------------------------------------------------------------
void svtkHigherOrderTetra::Clip(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd,
  svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd, int insideOut)
{
  svtkIdType bindices[4][4];
  svtkIdType numberOfSubtetras = this->GetNumberOfSubtetras();

  for (svtkIdType subCellId = 0; subCellId < numberOfSubtetras; subCellId++)
  {
    this->SubtetraBarycentricPointIndices(subCellId, bindices);

    for (svtkIdType i = 0; i < 4; i++)
    {
      svtkIdType pointIndex = this->ToIndex(bindices[i]);
      this->Tetra->Points->SetPoint(i, this->Points->GetPoint(pointIndex));
      if (outPd)
      {
        this->Tetra->PointIds->SetId(i, this->PointIds->GetId(pointIndex));
      }
      this->Scalars->SetTuple(i, cellScalars->GetTuple(pointIndex));
    }

    this->Tetra->Clip(
      value, this->Scalars, locator, polys, inPd, outPd, inCd, cellId, outCd, insideOut);
  }
}

//----------------------------------------------------------------------------
int svtkHigherOrderTetra::IntersectWithLine(
  const double* p1, const double* p2, double tol, double& t, double* x, double* pcoords, int& subId)
{
  int subTest;

  t = SVTK_DOUBLE_MAX;
  double tTmp;
  double xMin[3], pcoordsMin[3];

  for (int i = 0; i < this->GetNumberOfFaces(); i++)
  {
    if (this->GetFace(i)->IntersectWithLine(p1, p2, tol, tTmp, xMin, pcoordsMin, subTest) &&
      tTmp < t)
    {
      for (svtkIdType j = 0; j < 3; j++)
      {
        x[j] = xMin[j];
        if (FaceBCoords[i][j] != 3)
        {
          pcoords[FaceBCoords[i][j]] = pcoordsMin[j];
        }
      }
      if (FaceMinCoord[i] != 3)
      {
        pcoords[FaceMinCoord[i]] = 0.;
      }
      t = tTmp;
    }
  }
  subId = 0;
  return (t == SVTK_DOUBLE_MAX ? 0 : 1);
}

//----------------------------------------------------------------------------
int svtkHigherOrderTetra::Triangulate(int svtkNotUsed(index), svtkIdList* ptIds, svtkPoints* pts)
{
  pts->Reset();
  ptIds->Reset();

  svtkIdType bindices[4][4];
  svtkIdType numberOfSubtetras = this->GetNumberOfSubtetras();

  pts->SetNumberOfPoints(4 * numberOfSubtetras);
  ptIds->SetNumberOfIds(4 * numberOfSubtetras);
  for (svtkIdType subCellId = 0; subCellId < numberOfSubtetras; subCellId++)
  {
    this->SubtetraBarycentricPointIndices(subCellId, bindices);

    for (svtkIdType i = 0; i < 4; i++)
    {
      svtkIdType pointIndex = this->ToIndex(bindices[i]);
      ptIds->SetId(4 * subCellId + i, this->PointIds->GetId(pointIndex));
      pts->SetPoint(4 * subCellId + i, this->Points->GetPoint(pointIndex));
    }
  }
  return 1;
}

//----------------------------------------------------------------------------
void svtkHigherOrderTetra::JacobianInverse(const double pcoords[3], double** inverse, double* derivs)
{
  // Given parametric coordinates compute inverse Jacobian transformation
  // matrix. Returns 9 elements of 3x3 inverse Jacobian plus interpolation
  // function derivatives.

  int i, j, k;
  double *m[3], m0[3], m1[3], m2[3];
  double x[3];

  svtkIdType numberOfPoints = this->Points->GetNumberOfPoints();

  // compute interpolation function derivatives
  this->InterpolateDerivs(pcoords, derivs);

  // create Jacobian matrix
  m[0] = m0;
  m[1] = m1;
  m[2] = m2;
  for (i = 0; i < 3; i++) // initialize matrix
  {
    m0[i] = m1[i] = m2[i] = 0.0;
  }

  for (j = 0; j < numberOfPoints; j++)
  {
    this->Points->GetPoint(j, x);
    for (i = 0; i < 3; i++)
    {
      for (k = 0; k < 3; k++)
      {
        m[k][i] += x[i] * derivs[numberOfPoints * k + j];
      }
    }
  }

  if (!svtkMath::InvertMatrix(m, inverse, 3))
  {
    svtkErrorMacro(<< "Jacobian inverse not found");
    return;
  }
}

//----------------------------------------------------------------------------
void svtkHigherOrderTetra::Derivatives(
  int svtkNotUsed(subId), const double pcoords[3], const double* values, int dim, double* derivs)
{
  double *jI[3], j0[3], j1[3], j2[3];
  svtkIdType numberOfPoints = this->Points->GetNumberOfPoints();
  std::vector<double> fDs(3 * numberOfPoints);
  double sum[3];
  int i, j, k;

  // compute inverse Jacobian and interpolation function derivatives
  jI[0] = j0;
  jI[1] = j1;
  jI[2] = j2;
  this->JacobianInverse(pcoords, jI, &fDs[0]);

  // now compute derivatives of values provided
  for (k = 0; k < dim; k++) // loop over values per vertex
  {
    sum[0] = sum[1] = sum[2] = 0.0;
    for (i = 0; i < numberOfPoints; i++) // loop over interp. function derivatives
    {
      sum[0] += fDs[i] * values[dim * i + k];
      sum[1] += fDs[numberOfPoints + i] * values[dim * i + k];
      sum[2] += fDs[numberOfPoints * 2 + i] * values[dim * i + k];
    }
    for (j = 0; j < 3; j++) // loop over derivative directions
    {
      derivs[3 * k + j] = 0.;
      for (i = 0; i < 3; i++)
      {
        derivs[3 * k + j] += sum[i] * jI[j][i];
      }
    }
  }
}

//----------------------------------------------------------------------------

void svtkHigherOrderTetra::SetParametricCoords()
{
  svtkIdType nPoints = this->Points->GetNumberOfPoints();
#ifdef FIFTEEN_POINT_TETRA
  if (nPoints == 15)
  {
    return;
  }
#endif

  if (!this->PointParametricCoordinates)
  {
    this->PointParametricCoordinates = svtkSmartPointer<svtkPoints>::New();
    this->PointParametricCoordinates->SetDataTypeToDouble();
  }

  // Ensure Order is up-to-date and check that current point size matches:
  if (this->PointParametricCoordinates->GetNumberOfPoints() != nPoints)
  {
    this->PointParametricCoordinates->Initialize();
    double order_d = static_cast<svtkIdType>(this->GetOrder());
    this->PointParametricCoordinates->SetNumberOfPoints(nPoints);

    svtkIdType bindex[4];
    for (svtkIdType p = 0; p < nPoints; p++)
    {
      this->ToBarycentricIndex(p, bindex);
      this->PointParametricCoordinates->SetPoint(
        p, bindex[0] / order_d, bindex[1] / order_d, bindex[2] / order_d);
    }
  }
}

double* svtkHigherOrderTetra::GetParametricCoords()
{
#ifdef FIFTEEN_POINT_TETRA
  if (this->Points->GetNumberOfPoints() == 15)
  {
    return FifteenPointTetraCoords;
  }
#endif
  this->SetParametricCoords();

  return svtkDoubleArray::SafeDownCast(this->PointParametricCoordinates->GetData())->GetPointer(0);
}

//----------------------------------------------------------------------------
int svtkHigherOrderTetra::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = pcoords[2] = 0.25;
  return 0;
}

//----------------------------------------------------------------------------
double svtkHigherOrderTetra::GetParametricDistance(const double pcoords[3])
{
  int i;
  double pDist, pDistMax = 0.0;
  double pc[4];

  pc[0] = pcoords[0];
  pc[1] = pcoords[1];
  pc[2] = pcoords[2];
  pc[3] = 1.0 - pcoords[0] - pcoords[1] - pcoords[2];

  for (i = 0; i < 4; i++)
  {
    if (pc[i] < 0.0)
    {
      pDist = -pc[i];
    }
    else if (pc[i] > 1.0)
    {
      pDist = pc[i] - 1.0;
    }
    else // inside the cell in the parametric direction
    {
      pDist = 0.0;
    }
    if (pDist > pDistMax)
    {
      pDistMax = pDist;
    }
  }

  return pDistMax;
}

//----------------------------------------------------------------------------
svtkIdType svtkHigherOrderTetra::ComputeOrder()
{
  return svtkHigherOrderTetra::ComputeOrder(this->Points->GetNumberOfPoints());
}

svtkIdType svtkHigherOrderTetra::ComputeOrder(const svtkIdType nPoints)
{
  switch (nPoints)
  {
    case 1:
      return 0;
    case 4:
      return 1;
    case 10:
      return 2;
#ifdef FIFTEEN_POINT_TETRA
    case 15:
      return 2;
#endif
    case 20:
      return 3;
    case 35:
      return 4;
    case 56:
      return 5;
    case 84:
      return 6;
    case 120:
      return 7;
    case 165:
      return 8;
    case 220:
      return 9;
    case 286:
      return 10;

    // this is a iterative solution strategy to find the nearest integer ( order ) given the number
    // of points in the tetrahedron. the order is the root of following cubit equation
    // nPointsForOrder = (order + 1) * (order + 2) * (order + 3) / 6;
    // nPointsForOrder =  ( x3 + 6x2 + 11x + 6 ) / 6
    default:
    {
      svtkIdType order = 1;
      svtkIdType nPointsForOrder = 4;
      while (nPointsForOrder < nPoints)
      {
        order++;
        nPointsForOrder = (order + 1) * (order + 2) * (order + 3) / 6;
      }
      assert(nPoints == nPointsForOrder);
      return order;
    }
  }
}

//----------------------------------------------------------------------------
void svtkHigherOrderTetra::ToBarycentricIndex(svtkIdType index, svtkIdType* bindex)
{
#ifdef ENABLE_CACHING
  if (this->BarycentricIndexMap[4 * index] == -1)
  {
    svtkHigherOrderTetra::BarycentricIndex(
      index, &this->BarycentricIndexMap[4 * index], this->GetOrder());
  }
  for (svtkIdType i = 0; i < 4; i++)
  {
    bindex[i] = this->BarycentricIndexMap[4 * index + i];
  }
#else
  return svtkHigherOrderTetra::BarycentricIndex(index, bindex, this->GetOrder());
#endif
}

//----------------------------------------------------------------------------
svtkIdType svtkHigherOrderTetra::ToIndex(const svtkIdType* bindex)
{
#ifdef FIFTEEN_POINT_TETRA
  if (this->Points->GetNumberOfPoints() == 15)
  {
    return bindex[0];
  }
#endif

#ifdef ENABLE_CACHING
  svtkIdType cacheIdx =
    ((this->Order + 1) * (this->Order + 1) * bindex[0] + (this->Order + 1) * bindex[1] + bindex[2]);
  if (this->IndexMap[cacheIdx] == -1)
  {
    this->IndexMap[cacheIdx] = svtkHigherOrderTetra::Index(bindex, this->GetOrder());
  }
  return this->IndexMap[cacheIdx];
#else
  return svtkHigherOrderTetra::Index(bindex, this->GetOrder());
#endif
}

//----------------------------------------------------------------------------
void svtkHigherOrderTetra::BarycentricIndex(svtkIdType index, svtkIdType* bindex, svtkIdType order)
{
  // "Barycentric index" is a set of 4 integers, each running from 0 to
  // <Order>. It is the index of a point in the tetrahedron in barycentric
  // coordinates.

  assert(order >= 1);

  svtkIdType max = order;
  svtkIdType min = 0;

  // scope into the correct tetra
  while (index >= 2 * (order * order + 1) && index != 0 && order > 3)
  {
    index -= 2 * (order * order + 1);
    max -= 3;
    min++;
    order -= 4;
  }

  if (index < 4)
  {
    // we are on a vertex
    for (svtkIdType coord = 0; coord < 4; coord++)
    {
      bindex[coord] = (coord == VertexMaxCoords[index] ? max : min);
    }
    return;
  }
  else if (index - 4 < 6 * (order - 1))
  {
    // we are on an edge
    svtkIdType edgeId = (index - 4) / (order - 1);
    svtkIdType vertexId = (index - 4) % (order - 1);
    for (svtkIdType coord = 0; coord < 4; coord++)
    {
      bindex[coord] = min +
        (LinearVertices[EdgeVertices[edgeId][0]][coord] * (max - min - 1 - vertexId) +
          LinearVertices[EdgeVertices[edgeId][1]][coord] * (1 + vertexId));
    }
    return;
  }
  else
  {
    // we are on a face
    svtkIdType faceId = (index - 4 - 6 * (order - 1)) / ((order - 2) * (order - 1) / 2);
    svtkIdType vertexId = (index - 4 - 6 * (order - 1)) % ((order - 2) * (order - 1) / 2);

    svtkIdType projectedBIndex[3];
    if (order == 3)
    {
      projectedBIndex[0] = projectedBIndex[1] = projectedBIndex[2] = 0;
    }
    else
    {
      svtkHigherOrderTriangle::BarycentricIndex(vertexId, projectedBIndex, order - 3);
    }

    for (svtkIdType i = 0; i < 3; i++)
    {
      bindex[FaceBCoords[faceId][i]] = (min + 1 + projectedBIndex[i]);
    }
    bindex[FaceMinCoord[faceId]] = min;
    return;
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkHigherOrderTetra::Index(const svtkIdType* bindex, svtkIdType order)
{
  svtkIdType index = 0;

  assert(order >= 1);
  assert(bindex[0] + bindex[1] + bindex[2] + bindex[3] == order);

  svtkIdType max = order;
  svtkIdType min = 0;

  svtkIdType bmin = std::min(std::min(std::min(bindex[0], bindex[1]), bindex[2]), bindex[3]);

  // scope into the correct tetra
  while (bmin > min)
  {
    index += 2 * (order * order + 1);
    max -= 3;
    min++;
    order -= 4;
  }

  for (svtkIdType vertex = 0; vertex < 4; vertex++)
  {
    if (bindex[VertexMaxCoords[vertex]] == max)
    {
      // we are on a vertex
      return index;
    }
    index++;
  }

  for (svtkIdType edge = 0; edge < 6; edge++)
  {
    if (bindex[EdgeMinCoords[edge][0]] == min && bindex[EdgeMinCoords[edge][1]] == min)
    {
      // we are on an edge
      return index + bindex[EdgeCountingCoord[edge]] - (min + 1);
    }
    index += max - (min + 1);
  }

  for (svtkIdType face = 0; face < 4; face++)
  {
    if (bindex[FaceMinCoord[face]] == min)
    {
      // we are on a face
      svtkIdType projectedBIndex[3];
      for (svtkIdType i = 0; i < 3; i++)
      {
        projectedBIndex[i] = bindex[FaceBCoords[face][i]] - min;
      }
      // we must subtract the indices of the face's vertices and edges, which
      // total to 3*order
      return (index + svtkHigherOrderTriangle::Index(projectedBIndex, order) - 3 * order);
    }
    index += (order + 1) * (order + 2) / 2 - 3 * order;
  }
  return index;
}

//----------------------------------------------------------------------------
void svtkHigherOrderTetra::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
