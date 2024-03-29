/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPolyLine.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkPolyLine.h"

#include "svtkCellArray.h"
#include "svtkDoubleArray.h"
#include "svtkIdList.h"
#include "svtkIncrementalPointLocator.h"
#include "svtkLine.h"
#include "svtkMath.h"
#include "svtkNew.h"
#include "svtkObjectFactory.h"
#include "svtkPoints.h"
#include "svtkVector.h"
#include "svtkVectorOperators.h"

#include <algorithm>

svtkStandardNewMacro(svtkPolyLine);

//----------------------------------------------------------------------------
svtkPolyLine::svtkPolyLine()
{
  this->Line = svtkLine::New();
}

//----------------------------------------------------------------------------
svtkPolyLine::~svtkPolyLine()
{
  this->Line->Delete();
}

//----------------------------------------------------------------------------
int svtkPolyLine::GenerateSlidingNormals(svtkPoints* pts, svtkCellArray* lines, svtkDataArray* normals)
{
  return svtkPolyLine::GenerateSlidingNormals(pts, lines, normals, nullptr);
}

inline svtkIdType FindNextValidSegment(svtkPoints* points, svtkIdList* pointIds, svtkIdType start)
{
  svtkVector3d ps;
  points->GetPoint(pointIds->GetId(start), ps.GetData());

  svtkIdType end = start + 1;
  while (end < pointIds->GetNumberOfIds())
  {
    svtkVector3d pe;
    points->GetPoint(pointIds->GetId(end), pe.GetData());
    if (ps != pe)
    {
      return end - 1;
    }
    ++end;
  }

  return pointIds->GetNumberOfIds();
}

//----------------------------------------------------------------------------
// Given points and lines, compute normals to lines. These are not true
// normals, they are "orientation" normals used by classes like svtkTubeFilter
// that control the rotation around the line. The normals try to stay pointing
// in the same direction as much as possible (i.e., minimal rotation) w.r.t the
// firstNormal (computed if nullptr). Always returns 1 (success).
int svtkPolyLine::GenerateSlidingNormals(
  svtkPoints* pts, svtkCellArray* lines, svtkDataArray* normals, double* firstNormal)
{
  svtkVector3d normal(0.0, 0.0, 1.0); // arbitrary default value

  //svtkIdType lid = 0;
  svtkNew<svtkIdList> linePts;
  for (lines->InitTraversal(); lines->GetNextCell(linePts); /*++lid*/)
  {
    svtkIdType npts = linePts->GetNumberOfIds();
    if (npts <= 0)
    {
      continue;
    }
    if (npts == 1) // return arbitrary
    {
      normals->InsertTuple(linePts->GetId(0), normal.GetData());
      continue;
    }

    svtkIdType sNextId = 0;
    svtkVector3d sPrev, sNext;

    sNextId = FindNextValidSegment(pts, linePts, 0);
    if (sNextId != npts) // at least one valid segment
    {
      svtkVector3d pt1, pt2;
      pts->GetPoint(linePts->GetId(sNextId), pt1.GetData());
      pts->GetPoint(linePts->GetId(sNextId + 1), pt2.GetData());
      sPrev = (pt2 - pt1).Normalized();
    }
    else // no valid segments
    {
      for (svtkIdType i = 0; i < npts; ++i)
      {
        normals->InsertTuple(linePts->GetId(i), normal.GetData());
      }
      continue;
    }

    // compute first normal
    if (firstNormal)
    {
      normal = svtkVector3d(firstNormal);
    }
    else
    {
      // find the next valid, non-parallel segment
      while (++sNextId < npts)
      {
        sNextId = FindNextValidSegment(pts, linePts, sNextId);
        if (sNextId != npts)
        {
          svtkVector3d pt1, pt2;
          pts->GetPoint(linePts->GetId(sNextId), pt1.GetData());
          pts->GetPoint(linePts->GetId(sNextId + 1), pt2.GetData());
          sNext = (pt2 - pt1).Normalized();

          // now the starting normal should simply be the cross product
          // in the following if statement we check for the case where
          // the two segments are parallel, in which case, continue searching
          // for the next valid segment
          svtkVector3d n;
          n = sPrev.Cross(sNext);
          if (n.Norm() > 1.0E-3)
          {
            normal = n;
            sPrev = sNext;
            break;
          }
        }
      }

      if (sNextId >= npts) // only one valid segment
      {
        // a little trick to find othogonal normal
        for (int i = 0; i < 3; ++i)
        {
          if (sPrev[i] != 0.0)
          {
            normal[(i + 2) % 3] = 0.0;
            normal[(i + 1) % 3] = 1.0;
            normal[i] = -sPrev[(i + 1) % 3] / sPrev[i];
            break;
          }
        }
      }
    }
    normal.Normalize();

    // compute remaining normals
    svtkIdType lastNormalId = 0;
    while (++sNextId < npts)
    {
      sNextId = FindNextValidSegment(pts, linePts, sNextId);
      if (sNextId == npts)
      {
        break;
      }

      svtkVector3d pt1, pt2;
      pts->GetPoint(linePts->GetId(sNextId), pt1.GetData());
      pts->GetPoint(linePts->GetId(sNextId + 1), pt2.GetData());
      sNext = (pt2 - pt1).Normalized();

      // compute rotation vector
      svtkVector3d w = sPrev.Cross(normal);
      if (w.Normalize() == 0.0) // can't use this segment
      {
        continue;
      }

      // compute rotation of line segment
      svtkVector3d q = sNext.Cross(sPrev);
      if (q.Normalize() == 0.0) // can't use this segment
      {
        continue;
      }

      double f1 = q.Dot(normal);
      double f2 = 1.0 - (f1 * f1);
      if (f2 > 0.0)
      {
        f2 = sqrt(1.0 - (f1 * f1));
      }
      else
      {
        f2 = 0.0;
      }

      svtkVector3d c = (sNext + sPrev).Normalized();
      w = c.Cross(q);
      c = sPrev.Cross(q);
      if ((normal.Dot(c) * w.Dot(c)) < 0)
      {
        f2 = -1.0 * f2;
      }

      // insert current normal before updating
      for (svtkIdType i = lastNormalId; i < sNextId; ++i)
      {
        normals->InsertTuple(linePts->GetId(i), normal.GetData());
      }
      lastNormalId = sNextId;
      sPrev = sNext;

      // compute next normal
      normal = (f1 * q) + (f2 * w);
    }

    // insert last normal for the remaining points
    for (svtkIdType i = lastNormalId; i < npts; ++i)
    {
      normals->InsertTuple(linePts->GetId(i), normal.GetData());
    }
  }

  return 1;
}

//----------------------------------------------------------------------------
int svtkPolyLine::EvaluatePosition(const double x[3], double closestPoint[3], int& subId,
  double pcoords[3], double& minDist2, double weights[])
{
  double closest[3];
  double pc[3], dist2;
  int ignoreId, i, return_status, status;
  double lineWeights[2], closestWeights[2];

  pcoords[1] = pcoords[2] = 0.0;

  return_status = 0;
  subId = -1;
  closestWeights[0] = closestWeights[1] = 0.0; // Shut up, compiler
  for (minDist2 = SVTK_DOUBLE_MAX, i = 0; i < this->Points->GetNumberOfPoints() - 1; i++)
  {
    this->Line->Points->SetPoint(0, this->Points->GetPoint(i));
    this->Line->Points->SetPoint(1, this->Points->GetPoint(i + 1));
    status = this->Line->EvaluatePosition(x, closest, ignoreId, pc, dist2, lineWeights);
    if (status != -1 && dist2 < minDist2)
    {
      return_status = status;
      if (closestPoint)
      {
        closestPoint[0] = closest[0];
        closestPoint[1] = closest[1];
        closestPoint[2] = closest[2];
      }
      minDist2 = dist2;
      subId = i;
      pcoords[0] = pc[0];
      closestWeights[0] = lineWeights[0];
      closestWeights[1] = lineWeights[1];
    }
  }

  std::fill_n(weights, this->Points->GetNumberOfPoints(), 0.0);
  if (subId >= 0)
  {
    weights[subId] = closestWeights[0];
    weights[subId + 1] = closestWeights[1];
  }

  return return_status;
}

//----------------------------------------------------------------------------
void svtkPolyLine::EvaluateLocation(
  int& subId, const double pcoords[3], double x[3], double* weights)
{
  int i;
  double a1[3];
  double a2[3];
  this->Points->GetPoint(subId, a1);
  this->Points->GetPoint(subId + 1, a2);

  for (i = 0; i < 3; i++)
  {
    x[i] = a1[i] + pcoords[0] * (a2[i] - a1[i]);
  }

  weights[0] = 1.0 - pcoords[0];
  weights[1] = pcoords[0];
}

//----------------------------------------------------------------------------
int svtkPolyLine::CellBoundary(int subId, const double pcoords[3], svtkIdList* pts)
{
  pts->SetNumberOfIds(1);

  if (pcoords[0] >= 0.5)
  {
    pts->SetId(0, this->PointIds->GetId(subId + 1));
    if (pcoords[0] > 1.0)
    {
      return 0;
    }
    else
    {
      return 1;
    }
  }
  else
  {
    pts->SetId(0, this->PointIds->GetId(subId));
    if (pcoords[0] < 0.0)
    {
      return 0;
    }
    else
    {
      return 1;
    }
  }
}

//----------------------------------------------------------------------------
void svtkPolyLine::Contour(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* verts, svtkCellArray* lines,
  svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
  svtkCellData* outCd)
{
  int i, numLines = this->Points->GetNumberOfPoints() - 1;
  svtkDataArray* lineScalars = cellScalars->NewInstance();
  lineScalars->SetNumberOfComponents(cellScalars->GetNumberOfComponents());
  lineScalars->SetNumberOfTuples(2);

  for (i = 0; i < numLines; i++)
  {
    this->Line->Points->SetPoint(0, this->Points->GetPoint(i));
    this->Line->Points->SetPoint(1, this->Points->GetPoint(i + 1));

    if (outPd)
    {
      this->Line->PointIds->SetId(0, this->PointIds->GetId(i));
      this->Line->PointIds->SetId(1, this->PointIds->GetId(i + 1));
    }

    lineScalars->SetTuple(0, cellScalars->GetTuple(i));
    lineScalars->SetTuple(1, cellScalars->GetTuple(i + 1));

    this->Line->Contour(
      value, lineScalars, locator, verts, lines, polys, inPd, outPd, inCd, cellId, outCd);
  }
  lineScalars->Delete();
}

//----------------------------------------------------------------------------
// Intersect with sub-lines
//
int svtkPolyLine::IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t,
  double x[3], double pcoords[3], int& subId)
{
  int subTest, numLines = this->Points->GetNumberOfPoints() - 1;

  for (subId = 0; subId < numLines; subId++)
  {
    this->Line->Points->SetPoint(0, this->Points->GetPoint(subId));
    this->Line->Points->SetPoint(1, this->Points->GetPoint(subId + 1));

    if (this->Line->IntersectWithLine(p1, p2, tol, t, x, pcoords, subTest))
    {
      return 1;
    }
  }

  return 0;
}

//----------------------------------------------------------------------------
int svtkPolyLine::Triangulate(int svtkNotUsed(index), svtkIdList* ptIds, svtkPoints* pts)
{
  int numLines = this->Points->GetNumberOfPoints() - 1;
  pts->Reset();
  ptIds->Reset();

  for (int subId = 0; subId < numLines; subId++)
  {
    pts->InsertNextPoint(this->Points->GetPoint(subId));
    ptIds->InsertNextId(this->PointIds->GetId(subId));

    pts->InsertNextPoint(this->Points->GetPoint(subId + 1));
    ptIds->InsertNextId(this->PointIds->GetId(subId + 1));
  }

  return 1;
}

//----------------------------------------------------------------------------
void svtkPolyLine::Derivatives(
  int subId, const double pcoords[3], const double* values, int dim, double* derivs)
{
  this->Line->PointIds->SetNumberOfIds(2);

  this->Line->Points->SetPoint(0, this->Points->GetPoint(subId));
  this->Line->Points->SetPoint(1, this->Points->GetPoint(subId + 1));

  this->Line->Derivatives(0, pcoords, values + dim * subId, dim, derivs);
}

//----------------------------------------------------------------------------
void svtkPolyLine::Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
  svtkCellArray* lines, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
  svtkCellData* outCd, int insideOut)
{
  int i, numLines = this->Points->GetNumberOfPoints() - 1;
  svtkDoubleArray* lineScalars = svtkDoubleArray::New();
  lineScalars->SetNumberOfTuples(2);

  for (i = 0; i < numLines; i++)
  {
    this->Line->Points->SetPoint(0, this->Points->GetPoint(i));
    this->Line->Points->SetPoint(1, this->Points->GetPoint(i + 1));

    this->Line->PointIds->SetId(0, this->PointIds->GetId(i));
    this->Line->PointIds->SetId(1, this->PointIds->GetId(i + 1));

    lineScalars->SetComponent(0, 0, cellScalars->GetComponent(i, 0));
    lineScalars->SetComponent(1, 0, cellScalars->GetComponent(i + 1, 0));

    this->Line->Clip(
      value, lineScalars, locator, lines, inPd, outPd, inCd, cellId, outCd, insideOut);
  }

  lineScalars->Delete();
}

//----------------------------------------------------------------------------
// Return the center of the point cloud in parametric coordinates.
int svtkPolyLine::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = 0.5;
  pcoords[1] = pcoords[2] = 0.0;
  return ((this->Points->GetNumberOfPoints() - 1) / 2);
}

//----------------------------------------------------------------------------
void svtkPolyLine::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Line:\n";
  this->Line->PrintSelf(os, indent.GetNextIndent());
}
