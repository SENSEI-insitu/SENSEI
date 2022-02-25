/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCell3D.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkCell3D.h"

#include "svtkCellArray.h"
#include "svtkCellData.h"
#include "svtkDoubleArray.h"
#include "svtkMath.h"
#include "svtkOrderedTriangulator.h"
#include "svtkPointData.h"
#include "svtkPointLocator.h"
#include "svtkPoints.h"
#include "svtkPolygon.h"
#include "svtkTetra.h"

svtkCell3D::svtkCell3D()
{
  this->Triangulator = nullptr;
  this->MergeTolerance = 0.01;
  this->ClipTetra = nullptr;
  this->ClipScalars = nullptr;
}

svtkCell3D::~svtkCell3D()
{
  if (this->Triangulator)
  {
    this->Triangulator->Delete();
    this->Triangulator = nullptr;
  }
  if (this->ClipTetra)
  {
    this->ClipTetra->Delete();
    this->ClipTetra = nullptr;
    this->ClipScalars->Delete();
    this->ClipScalars = nullptr;
  }
}

bool svtkCell3D::IsInsideOut()
{
  // Strategy:
  // - Compute the centroid of the cell.
  // - Accumulate a signed projected distance on the normal between the faces and the centroid.
  // - Check the sign to see if the cell is inside out or not.
  double centroid[3], point[3], normal[3];
  this->GetCentroid(centroid);
  double signedDistanceToCentroid = 0.0;
  for (svtkIdType faceId = 0; faceId < this->GetNumberOfFaces(); ++faceId)
  {
    const svtkIdType* pointIds;
    svtkIdType faceSize = this->GetFacePoints(faceId, pointIds);
    if (faceSize)
    {
      this->Points->GetPoint(pointIds[0], point);
      svtkPolygon::ComputeNormal(this->Points, faceSize, pointIds, normal);
      signedDistanceToCentroid +=
        svtkPolygon::ComputeArea(this->Points, faceSize, pointIds, normal) *
        (svtkMath::Dot(normal, centroid) - svtkMath::Dot(normal, point));
    }
  }
  return signedDistanceToCentroid > 0.0;
}

void svtkCell3D::Contour(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* verts, svtkCellArray* lines,
  svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
  svtkCellData* outCd)
{
  int numPts = this->GetNumberOfPoints();
  int numEdges = this->GetNumberOfEdges();
  const svtkIdType* tets;
  int v1, v2;
  int i, j;
  int type;
  svtkIdType id, ptId;
  svtkIdType internalId[SVTK_CELL_SIZE];
  double s1, s2, x[3], t, p1[3], p2[3], deltaScalar;

  // Create a triangulator if necessary.
  if (!this->Triangulator)
  {
    this->Triangulator = svtkOrderedTriangulator::New();
    this->Triangulator->PreSortedOff();
    this->Triangulator->UseTemplatesOn();
    this->ClipTetra = svtkTetra::New();
    this->ClipScalars = svtkDoubleArray::New();
    this->ClipScalars->SetNumberOfTuples(4);
  }

  // If here, the ordered triangulator is going to be used so the triangulation
  // has to be initialized.
  this->Triangulator->InitTriangulation(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, (numPts + numEdges));

  // Cells with fixed topology are triangulated with templates.
  double *p, *pPtr = this->GetParametricCoords();
  if (this->IsPrimaryCell())
  {
    // Some cell types support templates for interior clipping. Templates
    // are a heck of a lot faster.
    type = 0; // inside
    for (p = pPtr, i = 0; i < numPts; i++, p += 3)
    {
      ptId = this->PointIds->GetId(i);
      this->Points->GetPoint(i, x);
      this->Triangulator->InsertPoint(ptId, x, p, type);
    } // for all cell points of fixed topology

    this->Triangulator->TemplateTriangulate(this->GetCellType(), numPts, numEdges);

    // Otherwise we have produced tetrahedra and now contour these using
    // the faster svtkTetra::Contour() method.
    for (this->Triangulator->InitTetraTraversal();
         this->Triangulator->GetNextTetra(0, this->ClipTetra, cellScalars, this->ClipScalars);)
    {
      this->ClipTetra->Contour(
        value, this->ClipScalars, locator, verts, lines, polys, inPd, outPd, inCd, cellId, outCd);
    }
    return;
  } // if we are clipping fixed topology

  // If here we're left with a non-fixed topology cell (e.g. convex point set).
  // Inject cell points into triangulation. Recall that the PreSortedOff()
  // flag was set which means that the triangulator will order the points
  // according to point id.
  for (p = pPtr, i = 0; i < numPts; i++, p += 3)
  {
    ptId = this->PointIds->GetId(i);

    // Currently all points are injected because of the possibility
    // of intersection point merging.
    s1 = cellScalars->GetComponent(i, 0);
    type = 0; // inside

    // Below is the old code since 2001. Will may take a look
    // at this at some point and see if there is a place to
    // improve it.
    //
    // if ( (s1 >= value) || (s1 < value) )
    // {
    //   type = 0; //inside
    // }
    // else
    // {
    //   type = 4; //outside, its type might change later (nearby intersection)
    // }

    this->Points->GetPoint(i, x);
    if (locator->InsertUniquePoint(x, id))
    {
      outPd->CopyData(inPd, ptId, id);
    }
    internalId[i] = this->Triangulator->InsertPoint(id, x, p, type);
  } // for all points

  // For each edge intersection point, insert into triangulation. Edge
  // intersections come from contouring value. Have to be careful of
  // intersections near existing points (causes bad Delaunay behavior).
  // Intersections near existing points are collapsed to existing point.
  double pc[3], *pc1, *pc2;
  for (int edgeNum = 0; edgeNum < numEdges; edgeNum++)
  {
    this->GetEdgePoints(edgeNum, tets);

    // Calculate a preferred interpolation direction.
    // Has to be done in same direction to insure coincident points are
    // merged (different interpolation direction causes perturbations).
    s1 = cellScalars->GetComponent(tets[0], 0);
    s2 = cellScalars->GetComponent(tets[1], 0);

    if ((s1 <= value && s2 >= value) || (s1 >= value && s2 <= value))
    {
      deltaScalar = s2 - s1;

      if (deltaScalar > 0)
      {
        v1 = tets[0];
        v2 = tets[1];
      }
      else
      {
        v1 = tets[1];
        v2 = tets[0];
        deltaScalar = -deltaScalar;
      }

      // linear interpolation
      t = (deltaScalar == 0.0 ? 0.0 : (value - cellScalars->GetComponent(v1, 0)) / deltaScalar);

      if (t < this->MergeTolerance)
      {
        this->Triangulator->UpdatePointType(internalId[v1], 2);
        continue;
      }
      else if (t > (1.0 - this->MergeTolerance))
      {
        this->Triangulator->UpdatePointType(internalId[v2], 2);
        continue;
      }

      this->Points->GetPoint(v1, p1);
      this->Points->GetPoint(v2, p2);
      pc1 = pPtr + 3 * v1;
      pc2 = pPtr + 3 * v2;

      for (j = 0; j < 3; j++)
      {
        x[j] = p1[j] + t * (p2[j] - p1[j]);
        pc[j] = pc1[j] + t * (pc2[j] - pc1[j]);
      }

      // Incorporate point into output and interpolate edge data as necessary
      if (locator->InsertUniquePoint(x, ptId))
      {
        outPd->InterpolateEdge(inPd, ptId, this->PointIds->GetId(v1), this->PointIds->GetId(v2), t);
      }

      // Insert intersection point into Delaunay triangulation
      this->Triangulator->InsertPoint(ptId, x, pc, 2);

    } // if edge intersects value
  }   // for all edges

  // triangulate the points
  this->Triangulator->Triangulate();

  // Add the triangulation to the mesh
  this->Triangulator->AddTetras(0, polys);
}

void svtkCell3D::Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
  svtkCellArray* tets, svtkPointData* inPD, svtkPointData* outPD, svtkCellData* inCD, svtkIdType cellId,
  svtkCellData* outCD, int insideOut)
{
  svtkCell3D* cell3D = static_cast<svtkCell3D*>(this); // has to be in this method
  int numPts = this->GetNumberOfPoints();
  int numEdges = this->GetNumberOfEdges();
  const svtkIdType* verts;
  int v1, v2;
  int i, j;
  int type;
  svtkIdType id, ptId;
  svtkIdType internalId[SVTK_CELL_SIZE];
  double s1, s2, x[3], t, p1[3], p2[3], deltaScalar;
  int allInside = 1, allOutside = 1;

  // Create a triangulator if necessary.
  if (!this->Triangulator)
  {
    this->Triangulator = svtkOrderedTriangulator::New();
    this->Triangulator->PreSortedOff();
    this->Triangulator->UseTemplatesOn();
    this->ClipTetra = svtkTetra::New();
    this->ClipScalars = svtkDoubleArray::New();
    this->ClipScalars->SetNumberOfTuples(4);
  }

  // Make sure it's worth continuing by treating the interior and exterior
  // cases as special cases.
  for (i = 0; i < numPts; i++)
  {
    s1 = cellScalars->GetComponent(i, 0);
    if ((s1 >= value && !insideOut) || (s1 < value && insideOut))
    {
      allOutside = 0;
    }
    else
    {
      allInside = 0;
    }
  }

  if (allOutside)
  {
    return;
  }

  // If here, the ordered triangulator is going to be used so the triangulation
  // has to be initialized.
  this->Triangulator->InitTriangulation(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, (numPts + numEdges));

  // Cells with fixed topology are triangulated with templates.
  double *p, *pPtr = this->GetParametricCoords();
  if (this->IsPrimaryCell())
  {
    // Some cell types support templates for interior clipping. Templates
    // are a heck of a lot faster.
    type = 0; // inside
    for (p = pPtr, i = 0; i < numPts; i++, p += 3)
    {
      ptId = this->PointIds->GetId(i);
      this->Points->GetPoint(i, x);
      if (locator->InsertUniquePoint(x, id))
      {
        outPD->CopyData(inPD, ptId, id);
      }
      this->Triangulator->InsertPoint(id, x, p, type);
    } // for all cell points of fixed topology

    this->Triangulator->TemplateTriangulate(this->GetCellType(), numPts, numEdges);
    // If the cell is interior we are done.
    if (allInside)
    {
      svtkIdType numTetras = tets->GetNumberOfCells();
      this->Triangulator->AddTetras(0, tets);
      svtkIdType numAddedTetras = tets->GetNumberOfCells() - numTetras;
      for (j = 0; j < numAddedTetras; j++)
      {
        outCD->CopyData(inCD, cellId, numTetras + j);
      }
    }
    // Otherwise we have produced tetrahedra and now clip these using
    // the faster svtkTetra::Clip() method.
    else
    {
      for (this->Triangulator->InitTetraTraversal();
           this->Triangulator->GetNextTetra(0, this->ClipTetra, cellScalars, this->ClipScalars);)
      {
        // VERY IMPORTANT: Notice that the outPD is used twice. This is because the
        // tetra has been defined in terms of point ids that are defined in the
        // output (because of the templates).
        this->ClipTetra->Clip(
          value, this->ClipScalars, locator, tets, outPD, outPD, inCD, cellId, outCD, insideOut);
      }
    } // if boundary cell
    return;
  } // if we are clipping fixed topology

  // If here we're left with a non-fixed topology cell (e.g. convex point set).
  // Inject cell points into triangulation. Recall that the PreSortedOff()
  // flag was set which means that the triangulator will order the points
  // according to point id.
  for (p = pPtr, i = 0; i < numPts; i++, p += 3)
  {
    ptId = this->PointIds->GetId(i);

    // Currently all points are injected because of the possibility
    // of intersection point merging.
    s1 = cellScalars->GetComponent(i, 0);
    if ((s1 >= value && !insideOut) || (s1 < value && insideOut))
    {
      type = 0; // inside
    }
    else
    {
      type = 4; // outside, its type might change later (nearby intersection)
    }

    this->Points->GetPoint(i, x);
    if (locator->InsertUniquePoint(x, id))
    {
      outPD->CopyData(inPD, ptId, id);
    }
    internalId[i] = this->Triangulator->InsertPoint(id, x, p, type);
  } // for all points

  // For each edge intersection point, insert into triangulation. Edge
  // intersections come from clipping value. Have to be careful of
  // intersections near existing points (causes bad Delaunay behavior).
  // Intersections near existing points are collapsed to existing point.
  double pc[3], *pc1, *pc2;
  for (int edgeNum = 0; edgeNum < numEdges; edgeNum++)
  {
    cell3D->GetEdgePoints(edgeNum, verts);

    // Calculate a preferred interpolation direction.
    // Has to be done in same direction to insure coincident points are
    // merged (different interpolation direction causes perturbations).
    s1 = cellScalars->GetComponent(verts[0], 0);
    s2 = cellScalars->GetComponent(verts[1], 0);

    if ((s1 <= value && s2 >= value) || (s1 >= value && s2 <= value))
    {
      deltaScalar = s2 - s1;

      if (deltaScalar > 0)
      {
        v1 = verts[0];
        v2 = verts[1];
      }
      else
      {
        v1 = verts[1];
        v2 = verts[0];
        deltaScalar = -deltaScalar;
      }

      // linear interpolation
      t = (deltaScalar == 0.0 ? 0.0 : (value - cellScalars->GetComponent(v1, 0)) / deltaScalar);

      if (t < this->MergeTolerance)
      {
        this->Triangulator->UpdatePointType(internalId[v1], 2);
        continue;
      }
      else if (t > (1.0 - this->MergeTolerance))
      {
        this->Triangulator->UpdatePointType(internalId[v2], 2);
        continue;
      }

      this->Points->GetPoint(v1, p1);
      this->Points->GetPoint(v2, p2);
      pc1 = pPtr + 3 * v1;
      pc2 = pPtr + 3 * v2;

      for (j = 0; j < 3; j++)
      {
        x[j] = p1[j] + t * (p2[j] - p1[j]);
        pc[j] = pc1[j] + t * (pc2[j] - pc1[j]);
      }

      // Incorporate point into output and interpolate edge data as necessary
      if (locator->InsertUniquePoint(x, ptId))
      {
        outPD->InterpolateEdge(inPD, ptId, this->PointIds->GetId(v1), this->PointIds->GetId(v2), t);
      }

      // Insert intersection point into Delaunay triangulation
      this->Triangulator->InsertPoint(ptId, x, pc, 2);

    } // if edge intersects value
  }   // for all edges

  // triangulate the points
  this->Triangulator->Triangulate();

  // Add the triangulation to the mesh
  this->Triangulator->AddTetras(0, tets);
}

void svtkCell3D::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Merge Tolerance: " << this->MergeTolerance << "\n";
}
