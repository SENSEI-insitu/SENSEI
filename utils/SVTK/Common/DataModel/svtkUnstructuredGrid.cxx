/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnstructuredGrid.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkUnstructuredGrid.h"

#include "svtkArrayDispatch.h"
#include "svtkBezierCurve.h"
#include "svtkBezierHexahedron.h"
#include "svtkBezierQuadrilateral.h"
#include "svtkBezierTetra.h"
#include "svtkBezierTriangle.h"
#include "svtkBezierWedge.h"
#include "svtkBiQuadraticQuad.h"
#include "svtkBiQuadraticQuadraticHexahedron.h"
#include "svtkBiQuadraticQuadraticWedge.h"
#include "svtkBiQuadraticTriangle.h"
#include "svtkCellArray.h"
#include "svtkCellArrayIterator.h"
#include "svtkCellData.h"
#include "svtkCellLinks.h"
#include "svtkCellTypes.h"
#include "svtkConvexPointSet.h"
#include "svtkCubicLine.h"
#include "svtkDataArrayRange.h"
#include "svtkDoubleArray.h"
#include "svtkEmptyCell.h"
#include "svtkGenericCell.h"
#include "svtkHexagonalPrism.h"
#include "svtkHexahedron.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkLagrangeCurve.h"
#include "svtkLagrangeHexahedron.h"
#include "svtkLagrangeQuadrilateral.h"
#include "svtkLagrangeTetra.h"
#include "svtkLagrangeTriangle.h"
#include "svtkLagrangeWedge.h"
#include "svtkLine.h"
#include "svtkNew.h"
#include "svtkObjectFactory.h"
#include "svtkPentagonalPrism.h"
#include "svtkPixel.h"
#include "svtkPointData.h"
#include "svtkPolyLine.h"
#include "svtkPolyVertex.h"
#include "svtkPolygon.h"
#include "svtkPolyhedron.h"
#include "svtkPyramid.h"
#include "svtkQuad.h"
#include "svtkQuadraticEdge.h"
#include "svtkQuadraticHexahedron.h"
#include "svtkQuadraticLinearQuad.h"
#include "svtkQuadraticLinearWedge.h"
#include "svtkQuadraticPolygon.h"
#include "svtkQuadraticPyramid.h"
#include "svtkQuadraticQuad.h"
#include "svtkQuadraticTetra.h"
#include "svtkQuadraticTriangle.h"
#include "svtkQuadraticWedge.h"
#include "svtkSMPThreadLocalObject.h"
#include "svtkSMPTools.h"
#include "svtkStaticCellLinks.h"
#include "svtkTetra.h"
#include "svtkTriQuadraticHexahedron.h"
#include "svtkTriangle.h"
#include "svtkTriangleStrip.h"
#include "svtkUnsignedCharArray.h"
#include "svtkUnstructuredGridCellIterator.h"
#include "svtkVertex.h"
#include "svtkVoxel.h"
#include "svtkWedge.h"

#include "svtkSMPTools.h"
#include "svtkTimerLog.h"

#include <algorithm>
#include <limits>
#include <set>

svtkStandardNewMacro(svtkUnstructuredGrid);

//----------------------------------------------------------------------------
svtkIdTypeArray* svtkUnstructuredGrid::GetCellLocationsArray()
{
  if (!this->CellLocations)
  {
    this->CellLocations = svtkSmartPointer<svtkIdTypeArray>::New();
  }
  this->CellLocations->DeepCopy(this->Connectivity->GetOffsetsArray());
  this->CellLocations->SetNumberOfValues(this->GetNumberOfCells());

  return this->CellLocations;
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::SetCells(
  svtkUnsignedCharArray* cellTypes, svtkIdTypeArray*, svtkCellArray* cells)
{
  this->SetCells(cellTypes, cells);
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::SetCells(svtkUnsignedCharArray* cellTypes, svtkIdTypeArray*,
  svtkCellArray* cells, svtkIdTypeArray* faceLocations, svtkIdTypeArray* faces)
{
  this->SetCells(cellTypes, cells, faceLocations, faces);
}

svtkUnstructuredGrid::svtkUnstructuredGrid()
{
  this->Vertex = nullptr;
  this->PolyVertex = nullptr;
  this->BezierCurve = nullptr;
  this->BezierQuadrilateral = nullptr;
  this->BezierHexahedron = nullptr;
  this->BezierTriangle = nullptr;
  this->BezierTetra = nullptr;
  this->BezierWedge = nullptr;
  this->LagrangeCurve = nullptr;
  this->LagrangeQuadrilateral = nullptr;
  this->LagrangeHexahedron = nullptr;
  this->LagrangeTriangle = nullptr;
  this->LagrangeTetra = nullptr;
  this->LagrangeWedge = nullptr;
  this->Line = nullptr;
  this->PolyLine = nullptr;
  this->Triangle = nullptr;
  this->TriangleStrip = nullptr;
  this->Pixel = nullptr;
  this->Quad = nullptr;
  this->Polygon = nullptr;
  this->Tetra = nullptr;
  this->Voxel = nullptr;
  this->Hexahedron = nullptr;
  this->Wedge = nullptr;
  this->Pyramid = nullptr;
  this->PentagonalPrism = nullptr;
  this->HexagonalPrism = nullptr;
  this->QuadraticEdge = nullptr;
  this->QuadraticTriangle = nullptr;
  this->QuadraticQuad = nullptr;
  this->QuadraticPolygon = nullptr;
  this->QuadraticTetra = nullptr;
  this->QuadraticHexahedron = nullptr;
  this->QuadraticWedge = nullptr;
  this->QuadraticPyramid = nullptr;
  this->QuadraticLinearQuad = nullptr;
  this->BiQuadraticQuad = nullptr;
  this->TriQuadraticHexahedron = nullptr;
  this->QuadraticLinearWedge = nullptr;
  this->BiQuadraticQuadraticWedge = nullptr;
  this->BiQuadraticQuadraticHexahedron = nullptr;
  this->BiQuadraticTriangle = nullptr;
  this->CubicLine = nullptr;

  this->ConvexPointSet = nullptr;
  this->Polyhedron = nullptr;
  this->EmptyCell = nullptr;

  this->Information->Set(svtkDataObject::DATA_EXTENT_TYPE(), SVTK_PIECES_EXTENT);
  this->Information->Set(svtkDataObject::DATA_PIECE_NUMBER(), -1);
  this->Information->Set(svtkDataObject::DATA_NUMBER_OF_PIECES(), 1);
  this->Information->Set(svtkDataObject::DATA_NUMBER_OF_GHOST_LEVELS(), 0);

  this->DistinctCellTypesUpdateMTime = 0;

  this->AllocateExact(1024, 1024);
}

//----------------------------------------------------------------------------
svtkUnstructuredGrid::~svtkUnstructuredGrid()
{
  if (this->Vertex)
  {
    this->Vertex->Delete();
  }
  if (this->PolyVertex)
  {
    this->PolyVertex->Delete();
  }
  if (this->BezierCurve)
  {
    this->BezierCurve->Delete();
  }
  if (this->BezierQuadrilateral)
  {
    this->BezierQuadrilateral->Delete();
  }
  if (this->BezierHexahedron)
  {
    this->BezierHexahedron->Delete();
  }
  if (this->BezierTriangle)
  {
    this->BezierTriangle->Delete();
  }
  if (this->BezierTetra)
  {
    this->BezierTetra->Delete();
  }
  if (this->BezierWedge)
  {
    this->BezierWedge->Delete();
  }
  if (this->LagrangeCurve)
  {
    this->LagrangeCurve->Delete();
  }
  if (this->LagrangeQuadrilateral)
  {
    this->LagrangeQuadrilateral->Delete();
  }
  if (this->LagrangeHexahedron)
  {
    this->LagrangeHexahedron->Delete();
  }
  if (this->LagrangeTriangle)
  {
    this->LagrangeTriangle->Delete();
  }
  if (this->LagrangeTetra)
  {
    this->LagrangeTetra->Delete();
  }
  if (this->LagrangeWedge)
  {
    this->LagrangeWedge->Delete();
  }
  if (this->Line)
  {
    this->Line->Delete();
  }
  if (this->PolyLine)
  {
    this->PolyLine->Delete();
  }
  if (this->Triangle)
  {
    this->Triangle->Delete();
  }
  if (this->TriangleStrip)
  {
    this->TriangleStrip->Delete();
  }
  if (this->Pixel)
  {
    this->Pixel->Delete();
  }
  if (this->Quad)
  {
    this->Quad->Delete();
  }
  if (this->Polygon)
  {
    this->Polygon->Delete();
  }
  if (this->Tetra)
  {
    this->Tetra->Delete();
  }
  if (this->Voxel)
  {
    this->Voxel->Delete();
  }
  if (this->Hexahedron)
  {
    this->Hexahedron->Delete();
  }
  if (this->Wedge)
  {
    this->Wedge->Delete();
  }
  if (this->Pyramid)
  {
    this->Pyramid->Delete();
  }
  if (this->PentagonalPrism)
  {
    this->PentagonalPrism->Delete();
  }
  if (this->HexagonalPrism)
  {
    this->HexagonalPrism->Delete();
  }
  if (this->QuadraticEdge)
  {
    this->QuadraticEdge->Delete();
  }
  if (this->QuadraticTriangle)
  {
    this->QuadraticTriangle->Delete();
  }
  if (this->QuadraticQuad)
  {
    this->QuadraticQuad->Delete();
  }
  if (this->QuadraticPolygon)
  {
    this->QuadraticPolygon->Delete();
  }
  if (this->QuadraticTetra)
  {
    this->QuadraticTetra->Delete();
  }
  if (this->QuadraticHexahedron)
  {
    this->QuadraticHexahedron->Delete();
  }
  if (this->QuadraticWedge)
  {
    this->QuadraticWedge->Delete();
  }
  if (this->QuadraticPyramid)
  {
    this->QuadraticPyramid->Delete();
  }
  if (this->QuadraticLinearQuad)
  {
    this->QuadraticLinearQuad->Delete();
  }
  if (this->BiQuadraticQuad)
  {
    this->BiQuadraticQuad->Delete();
  }
  if (this->TriQuadraticHexahedron)
  {
    this->TriQuadraticHexahedron->Delete();
  }
  if (this->QuadraticLinearWedge)
  {
    this->QuadraticLinearWedge->Delete();
  }
  if (this->BiQuadraticQuadraticWedge)
  {
    this->BiQuadraticQuadraticWedge->Delete();
  }
  if (this->BiQuadraticQuadraticHexahedron)
  {
    this->BiQuadraticQuadraticHexahedron->Delete();
  }
  if (this->BiQuadraticTriangle)
  {
    this->BiQuadraticTriangle->Delete();
  }
  if (this->CubicLine)
  {
    this->CubicLine->Delete();
  }

  if (this->ConvexPointSet)
  {
    this->ConvexPointSet->Delete();
  }
  if (this->Polyhedron)
  {
    this->Polyhedron->Delete();
  }
  if (this->EmptyCell)
  {
    this->EmptyCell->Delete();
  }
}

//----------------------------------------------------------------------------
int svtkUnstructuredGrid::GetPiece()
{
  return this->Information->Get(svtkDataObject::DATA_PIECE_NUMBER());
}

//----------------------------------------------------------------------------
int svtkUnstructuredGrid::GetNumberOfPieces()
{
  return this->Information->Get(svtkDataObject::DATA_NUMBER_OF_PIECES());
}

//----------------------------------------------------------------------------
int svtkUnstructuredGrid::GetGhostLevel()
{
  return this->Information->Get(svtkDataObject::DATA_NUMBER_OF_GHOST_LEVELS());
}

//----------------------------------------------------------------------------
// Copy the geometric and topological structure of an input unstructured grid.
void svtkUnstructuredGrid::CopyStructure(svtkDataSet* ds)
{
  // If ds is a svtkUnstructuredGrid, do a shallow copy of the cell data.
  if (svtkUnstructuredGrid* ug = svtkUnstructuredGrid::SafeDownCast(ds))
  {
    this->Connectivity = ug->Connectivity;
    this->Links = ug->Links;
    this->Types = ug->Types;
    this->DistinctCellTypes = nullptr;
    this->DistinctCellTypesUpdateMTime = 0;
    this->Faces = ug->Faces;
    this->FaceLocations = ug->FaceLocations;
  }

  this->Superclass::CopyStructure(ds);
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::Cleanup()
{
  this->Connectivity = nullptr;
  this->Links = nullptr;
  this->Types = nullptr;
  this->DistinctCellTypes = nullptr;
  this->DistinctCellTypesUpdateMTime = 0;
  this->Faces = nullptr;
  this->FaceLocations = nullptr;
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::Initialize()
{
  svtkPointSet::Initialize();

  this->Cleanup();

  if (this->Information)
  {
    this->Information->Set(svtkDataObject::DATA_PIECE_NUMBER(), -1);
    this->Information->Set(svtkDataObject::DATA_NUMBER_OF_PIECES(), 0);
    this->Information->Set(svtkDataObject::DATA_NUMBER_OF_GHOST_LEVELS(), 0);
  }
}

//----------------------------------------------------------------------------
int svtkUnstructuredGrid::GetCellType(svtkIdType cellId)
{
  svtkDebugMacro(<< "Returning cell type " << static_cast<int>(this->Types->GetValue(cellId)));
  return static_cast<int>(this->Types->GetValue(cellId));
}

//----------------------------------------------------------------------------
svtkCell* svtkUnstructuredGrid::GetCell(svtkIdType cellId)
{
  svtkIdType numPts;
  const svtkIdType* pts;
  this->Connectivity->GetCellAtId(cellId, numPts, pts);

  svtkCell* cell = nullptr;
  switch (this->Types->GetValue(cellId))
  {
    case SVTK_VERTEX:
      if (!this->Vertex)
      {
        this->Vertex = svtkVertex::New();
      }
      cell = this->Vertex;
      break;

    case SVTK_POLY_VERTEX:
      if (!this->PolyVertex)
      {
        this->PolyVertex = svtkPolyVertex::New();
      }
      cell = this->PolyVertex;
      break;

    case SVTK_LINE:
      if (!this->Line)
      {
        this->Line = svtkLine::New();
      }
      cell = this->Line;
      break;

    case SVTK_LAGRANGE_CURVE:
      if (!this->LagrangeCurve)
      {
        this->LagrangeCurve = svtkLagrangeCurve::New();
      }
      cell = this->LagrangeCurve;
      break;

    case SVTK_LAGRANGE_QUADRILATERAL:
      if (!this->LagrangeQuadrilateral)
      {
        this->LagrangeQuadrilateral = svtkLagrangeQuadrilateral::New();
      }
      if (GetCellData()->SetActiveAttribute(
            "HigherOrderDegrees", svtkDataSetAttributes::AttributeTypes::HIGHERORDERDEGREES) != -1)
      {
        double degs[3];
        svtkDataArray* v = GetCellData()->GetHigherOrderDegrees();
        v->GetTuple(cellId, degs);
        this->LagrangeQuadrilateral->SetOrder(degs[0], degs[1]);
      }
      else
      {
        this->LagrangeQuadrilateral->SetUniformOrderFromNumPoints(numPts);
      }
      cell = this->LagrangeQuadrilateral;
      break;

    case SVTK_LAGRANGE_HEXAHEDRON:
      if (!this->LagrangeHexahedron)
      {
        this->LagrangeHexahedron = svtkLagrangeHexahedron::New();
      }
      if (GetCellData()->SetActiveAttribute(
            "HigherOrderDegrees", svtkDataSetAttributes::AttributeTypes::HIGHERORDERDEGREES) != -1)
      {
        double degs[3];
        svtkDataArray* v = GetCellData()->GetHigherOrderDegrees();
        v->GetTuple(cellId, degs);
        this->LagrangeHexahedron->SetOrder(degs[0], degs[1], degs[2]);
      }
      else
      {
        this->LagrangeHexahedron->SetUniformOrderFromNumPoints(numPts);
      }
      cell = this->LagrangeHexahedron;
      break;

    case SVTK_LAGRANGE_TRIANGLE:
      if (!this->LagrangeTriangle)
      {
        this->LagrangeTriangle = svtkLagrangeTriangle::New();
      }
      cell = this->LagrangeTriangle;
      break;

    case SVTK_LAGRANGE_TETRAHEDRON:
      if (!this->LagrangeTetra)
      {
        this->LagrangeTetra = svtkLagrangeTetra::New();
      }
      cell = this->LagrangeTetra;
      break;

    case SVTK_LAGRANGE_WEDGE:
      if (!this->LagrangeWedge)
      {
        this->LagrangeWedge = svtkLagrangeWedge::New();
      }
      if (GetCellData()->SetActiveAttribute(
            "HigherOrderDegrees", svtkDataSetAttributes::AttributeTypes::HIGHERORDERDEGREES) != -1)
      {
        double degs[3];
        svtkDataArray* v = GetCellData()->GetHigherOrderDegrees();
        v->GetTuple(cellId, degs);
        this->LagrangeWedge->SetOrder(degs[0], degs[1], degs[2], numPts);
      }
      else
      {
        this->LagrangeWedge->SetUniformOrderFromNumPoints(numPts);
      }
      cell = this->LagrangeWedge;
      break;

    case SVTK_BEZIER_CURVE:
      if (!this->BezierCurve)
      {
        this->BezierCurve = svtkBezierCurve::New();
      }
      if (GetPointData()->SetActiveAttribute(
            "RationalWeights", svtkDataSetAttributes::AttributeTypes::RATIONALWEIGHTS) != -1)
      {
        svtkDataArray* v = GetPointData()->GetRationalWeights();
        this->BezierCurve->GetRationalWeights()->SetNumberOfTuples(numPts);
        for (int i = 0; i < numPts; i++)
        {
          this->BezierCurve->GetRationalWeights()->SetValue(i, v->GetTuple1(pts[i]));
        }
      }
      else
        this->BezierCurve->GetRationalWeights()->Reset();
      cell = this->BezierCurve;
      break;

    case SVTK_BEZIER_QUADRILATERAL:
      if (!this->BezierQuadrilateral)
      {
        this->BezierQuadrilateral = svtkBezierQuadrilateral::New();
      }
      if (GetCellData()->SetActiveAttribute(
            "HigherOrderDegrees", svtkDataSetAttributes::AttributeTypes::HIGHERORDERDEGREES) != -1)
      {
        double degs[3];
        svtkDataArray* v = GetCellData()->GetHigherOrderDegrees();
        v->GetTuple(cellId, degs);
        this->BezierQuadrilateral->SetOrder(degs[0], degs[1]);
      }
      else
      {
        this->BezierQuadrilateral->SetUniformOrderFromNumPoints(numPts);
      }
      if (GetPointData()->SetActiveAttribute(
            "RationalWeights", svtkDataSetAttributes::AttributeTypes::RATIONALWEIGHTS) != -1)
      {
        svtkDataArray* v = GetPointData()->GetRationalWeights();
        this->BezierQuadrilateral->GetRationalWeights()->SetNumberOfTuples(numPts);
        for (int i = 0; i < numPts; i++)
        {
          this->BezierQuadrilateral->GetRationalWeights()->SetValue(i, v->GetTuple1(pts[i]));
        }
      }
      else
        this->BezierQuadrilateral->GetRationalWeights()->Reset();
      cell = this->BezierQuadrilateral;
      break;

    case SVTK_BEZIER_HEXAHEDRON:
      if (!this->BezierHexahedron)
      {
        this->BezierHexahedron = svtkBezierHexahedron::New();
      }
      if (GetCellData()->SetActiveAttribute(
            "HigherOrderDegrees", svtkDataSetAttributes::AttributeTypes::HIGHERORDERDEGREES) != -1)
      {
        double degs[3];
        svtkDataArray* v = GetCellData()->GetHigherOrderDegrees();
        v->GetTuple(cellId, degs);
        this->BezierHexahedron->SetOrder(degs[0], degs[1], degs[2]);
      }
      else
      {
        this->BezierHexahedron->SetUniformOrderFromNumPoints(numPts);
      }
      if (GetPointData()->SetActiveAttribute(
            "RationalWeights", svtkDataSetAttributes::AttributeTypes::RATIONALWEIGHTS) != -1)
      {
        svtkDataArray* v = GetPointData()->GetRationalWeights();
        this->BezierHexahedron->GetRationalWeights()->SetNumberOfTuples(numPts);
        for (int i = 0; i < numPts; i++)
        {
          this->BezierHexahedron->GetRationalWeights()->SetValue(i, v->GetTuple1(pts[i]));
        }
      }
      else
        this->BezierHexahedron->GetRationalWeights()->Reset();
      cell = this->BezierHexahedron;
      break;

    case SVTK_BEZIER_TRIANGLE:
      if (!this->BezierTriangle)
      {
        this->BezierTriangle = svtkBezierTriangle::New();
      }
      if (GetPointData()->SetActiveAttribute(
            "RationalWeights", svtkDataSetAttributes::AttributeTypes::RATIONALWEIGHTS) != -1)
      {
        svtkDataArray* v = GetPointData()->GetRationalWeights();
        this->BezierTriangle->GetRationalWeights()->SetNumberOfTuples(numPts);
        for (int i = 0; i < numPts; i++)
        {
          this->BezierTriangle->GetRationalWeights()->SetValue(i, v->GetTuple1(pts[i]));
        }
      }
      else
        this->BezierTriangle->GetRationalWeights()->Reset();
      cell = this->BezierTriangle;
      break;

    case SVTK_BEZIER_TETRAHEDRON:
      if (!this->BezierTetra)
      {
        this->BezierTetra = svtkBezierTetra::New();
      }
      if (GetPointData()->SetActiveAttribute(
            "RationalWeights", svtkDataSetAttributes::AttributeTypes::RATIONALWEIGHTS) != -1)
      {
        svtkDataArray* v = GetPointData()->GetRationalWeights();
        this->BezierTetra->GetRationalWeights()->SetNumberOfTuples(numPts);
        for (int i = 0; i < numPts; i++)
        {
          this->BezierTetra->GetRationalWeights()->SetValue(i, v->GetTuple1(pts[i]));
        }
      }
      else
        this->BezierTetra->GetRationalWeights()->Reset();
      cell = this->BezierTetra;
      break;

    case SVTK_BEZIER_WEDGE:
      if (!this->BezierWedge)
      {
        this->BezierWedge = svtkBezierWedge::New();
      }
      if (GetCellData()->SetActiveAttribute(
            "HigherOrderDegrees", svtkDataSetAttributes::AttributeTypes::HIGHERORDERDEGREES) != -1)
      {
        double degs[3];
        svtkDataArray* v = GetCellData()->GetHigherOrderDegrees();
        v->GetTuple(cellId, degs);
        this->BezierWedge->SetOrder(degs[0], degs[1], degs[2], numPts);
      }
      else
      {
        this->BezierWedge->SetUniformOrderFromNumPoints(numPts);
      }
      if (GetPointData()->SetActiveAttribute(
            "RationalWeights", svtkDataSetAttributes::AttributeTypes::RATIONALWEIGHTS) != -1)
      {
        svtkDataArray* v = GetPointData()->GetRationalWeights();
        this->BezierWedge->GetRationalWeights()->SetNumberOfTuples(numPts);
        for (int i = 0; i < numPts; i++)
        {
          this->BezierWedge->GetRationalWeights()->SetValue(i, v->GetTuple1(pts[i]));
        }
      }
      else
        this->BezierWedge->GetRationalWeights()->Reset();
      cell = this->BezierWedge;
      break;

    case SVTK_POLY_LINE:
      if (!this->PolyLine)
      {
        this->PolyLine = svtkPolyLine::New();
      }
      cell = this->PolyLine;
      break;

    case SVTK_TRIANGLE:
      if (!this->Triangle)
      {
        this->Triangle = svtkTriangle::New();
      }
      cell = this->Triangle;
      break;

    case SVTK_TRIANGLE_STRIP:
      if (!this->TriangleStrip)
      {
        this->TriangleStrip = svtkTriangleStrip::New();
      }
      cell = this->TriangleStrip;
      break;

    case SVTK_PIXEL:
      if (!this->Pixel)
      {
        this->Pixel = svtkPixel::New();
      }
      cell = this->Pixel;
      break;

    case SVTK_QUAD:
      if (!this->Quad)
      {
        this->Quad = svtkQuad::New();
      }
      cell = this->Quad;
      break;

    case SVTK_POLYGON:
      if (!this->Polygon)
      {
        this->Polygon = svtkPolygon::New();
      }
      cell = this->Polygon;
      break;

    case SVTK_TETRA:
      if (!this->Tetra)
      {
        this->Tetra = svtkTetra::New();
      }
      cell = this->Tetra;
      break;

    case SVTK_VOXEL:
      if (!this->Voxel)
      {
        this->Voxel = svtkVoxel::New();
      }
      cell = this->Voxel;
      break;

    case SVTK_HEXAHEDRON:
      if (!this->Hexahedron)
      {
        this->Hexahedron = svtkHexahedron::New();
      }
      cell = this->Hexahedron;
      break;

    case SVTK_WEDGE:
      if (!this->Wedge)
      {
        this->Wedge = svtkWedge::New();
      }
      cell = this->Wedge;
      break;

    case SVTK_PYRAMID:
      if (!this->Pyramid)
      {
        this->Pyramid = svtkPyramid::New();
      }
      cell = this->Pyramid;
      break;

    case SVTK_PENTAGONAL_PRISM:
      if (!this->PentagonalPrism)
      {
        this->PentagonalPrism = svtkPentagonalPrism::New();
      }
      cell = this->PentagonalPrism;
      break;

    case SVTK_HEXAGONAL_PRISM:
      if (!this->HexagonalPrism)
      {
        this->HexagonalPrism = svtkHexagonalPrism::New();
      }
      cell = this->HexagonalPrism;
      break;

    case SVTK_QUADRATIC_EDGE:
      if (!this->QuadraticEdge)
      {
        this->QuadraticEdge = svtkQuadraticEdge::New();
      }
      cell = this->QuadraticEdge;
      break;

    case SVTK_QUADRATIC_TRIANGLE:
      if (!this->QuadraticTriangle)
      {
        this->QuadraticTriangle = svtkQuadraticTriangle::New();
      }
      cell = this->QuadraticTriangle;
      break;

    case SVTK_QUADRATIC_QUAD:
      if (!this->QuadraticQuad)
      {
        this->QuadraticQuad = svtkQuadraticQuad::New();
      }
      cell = this->QuadraticQuad;
      break;

    case SVTK_QUADRATIC_POLYGON:
      if (!this->QuadraticPolygon)
      {
        this->QuadraticPolygon = svtkQuadraticPolygon::New();
      }
      cell = this->QuadraticPolygon;
      break;

    case SVTK_QUADRATIC_TETRA:
      if (!this->QuadraticTetra)
      {
        this->QuadraticTetra = svtkQuadraticTetra::New();
      }
      cell = this->QuadraticTetra;
      break;

    case SVTK_QUADRATIC_HEXAHEDRON:
      if (!this->QuadraticHexahedron)
      {
        this->QuadraticHexahedron = svtkQuadraticHexahedron::New();
      }
      cell = this->QuadraticHexahedron;
      break;

    case SVTK_QUADRATIC_WEDGE:
      if (!this->QuadraticWedge)
      {
        this->QuadraticWedge = svtkQuadraticWedge::New();
      }
      cell = this->QuadraticWedge;
      break;

    case SVTK_QUADRATIC_PYRAMID:
      if (!this->QuadraticPyramid)
      {
        this->QuadraticPyramid = svtkQuadraticPyramid::New();
      }
      cell = this->QuadraticPyramid;
      break;

    case SVTK_QUADRATIC_LINEAR_QUAD:
      if (!this->QuadraticLinearQuad)
      {
        this->QuadraticLinearQuad = svtkQuadraticLinearQuad::New();
      }
      cell = this->QuadraticLinearQuad;
      break;

    case SVTK_BIQUADRATIC_QUAD:
      if (!this->BiQuadraticQuad)
      {
        this->BiQuadraticQuad = svtkBiQuadraticQuad::New();
      }
      cell = this->BiQuadraticQuad;
      break;

    case SVTK_TRIQUADRATIC_HEXAHEDRON:
      if (!this->TriQuadraticHexahedron)
      {
        this->TriQuadraticHexahedron = svtkTriQuadraticHexahedron::New();
      }
      cell = this->TriQuadraticHexahedron;
      break;

    case SVTK_QUADRATIC_LINEAR_WEDGE:
      if (!this->QuadraticLinearWedge)
      {
        this->QuadraticLinearWedge = svtkQuadraticLinearWedge::New();
      }
      cell = this->QuadraticLinearWedge;
      break;

    case SVTK_BIQUADRATIC_QUADRATIC_WEDGE:
      if (!this->BiQuadraticQuadraticWedge)
      {
        this->BiQuadraticQuadraticWedge = svtkBiQuadraticQuadraticWedge::New();
      }
      cell = this->BiQuadraticQuadraticWedge;
      break;

    case SVTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON:
      if (!this->BiQuadraticQuadraticHexahedron)
      {
        this->BiQuadraticQuadraticHexahedron = svtkBiQuadraticQuadraticHexahedron::New();
      }
      cell = this->BiQuadraticQuadraticHexahedron;
      break;
    case SVTK_BIQUADRATIC_TRIANGLE:
      if (!this->BiQuadraticTriangle)
      {
        this->BiQuadraticTriangle = svtkBiQuadraticTriangle::New();
      }
      cell = this->BiQuadraticTriangle;
      break;
    case SVTK_CUBIC_LINE:
      if (!this->CubicLine)
      {
        this->CubicLine = svtkCubicLine::New();
      }
      cell = this->CubicLine;
      break;

    case SVTK_CONVEX_POINT_SET:
      if (!this->ConvexPointSet)
      {
        this->ConvexPointSet = svtkConvexPointSet::New();
      }
      cell = this->ConvexPointSet;
      break;

    case SVTK_POLYHEDRON:
      if (!this->Polyhedron)
      {
        this->Polyhedron = svtkPolyhedron::New();
      }
      this->Polyhedron->SetFaces(this->GetFaces(cellId));
      cell = this->Polyhedron;
      break;

    case SVTK_EMPTY_CELL:
      if (!this->EmptyCell)
      {
        this->EmptyCell = svtkEmptyCell::New();
      }
      cell = this->EmptyCell;
      break;
  }

  if (!cell)
  {
    return nullptr;
  }

  // Copy the points over to the cell.
  cell->PointIds->SetNumberOfIds(numPts);
  cell->Points->SetNumberOfPoints(numPts);
  for (svtkIdType i = 0; i < numPts; i++)
  {
    cell->PointIds->SetId(i, pts[i]);
    cell->Points->SetPoint(i, this->Points->GetPoint(pts[i]));
  }

  // Some cells require special initialization to build data structures
  // and such.
  if (cell->RequiresInitialization())
  {
    cell->Initialize();
  }

  return cell;
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::GetCell(svtkIdType cellId, svtkGenericCell* cell)
{

  int cellType = static_cast<int>(this->Types->GetValue(cellId));
  cell->SetCellType(cellType);

  svtkIdType numPts;
  const svtkIdType* pts;
  this->Connectivity->GetCellAtId(cellId, numPts, pts);

  cell->PointIds->SetNumberOfIds(numPts);

  std::copy(pts, pts + numPts, cell->PointIds->GetPointer(0));
  this->Points->GetPoints(cell->PointIds, cell->Points);

  // Explicit face representation
  if (cell->RequiresExplicitFaceRepresentation())
  {
    cell->SetFaces(this->GetFaces(cellId));
  }

  // Some cells require special initialization to build data structures
  // and such.
  if (cell->RequiresInitialization())
  {
    cell->Initialize();
  }
  this->SetCellOrderAndRationalWeights(cellId, cell);
}

//----------------------------------------------------------------------------
// Support GetCellBounds()
namespace
{ // anonymous
struct ComputeCellBoundsWorker
{
  struct Visitor
  {
    // svtkCellArray::Visit entry point:
    template <typename CellStateT, typename PointArrayT>
    void operator()(
      CellStateT& state, PointArrayT* ptArray, svtkIdType cellId, double bounds[6]) const
    {
      using IdType = typename CellStateT::ValueType;

      const auto ptIds = state.GetCellRange(cellId);
      if (ptIds.size() == 0)
      {
        svtkMath::UninitializeBounds(bounds);
        return;
      }

      const auto points = svtk::DataArrayTupleRange<3>(ptArray);

      // Initialize bounds to first point:
      {
        const auto pt = points[ptIds[0]];

        // Explicitly reusing a local will improve performance when virtual
        // calls are involved in the iterator read:
        const double x = static_cast<double>(pt[0]);
        const double y = static_cast<double>(pt[1]);
        const double z = static_cast<double>(pt[2]);

        bounds[0] = x;
        bounds[1] = x;
        bounds[2] = y;
        bounds[3] = y;
        bounds[4] = z;
        bounds[5] = z;
      }

      // Reduce bounds with the rest of the ids:
      for (const IdType ptId : ptIds.GetSubRange(1))
      {
        const auto pt = points[ptId];

        // Explicitly reusing a local will improve performance when virtual
        // calls are involved in the iterator read:
        const double x = static_cast<double>(pt[0]);
        const double y = static_cast<double>(pt[1]);
        const double z = static_cast<double>(pt[2]);

        bounds[0] = std::min(bounds[0], x);
        bounds[1] = std::max(bounds[1], x);
        bounds[2] = std::min(bounds[2], y);
        bounds[3] = std::max(bounds[3], y);
        bounds[4] = std::min(bounds[4], z);
        bounds[5] = std::max(bounds[5], z);
      }
    }
  };

  // svtkArrayDispatch entry point:
  template <typename PointArrayT>
  void operator()(
    PointArrayT* ptArray, svtkCellArray* conn, svtkIdType cellId, double bounds[6]) const
  {
    conn->Visit(Visitor{}, ptArray, cellId, bounds);
  }
};

} // anonymous

//----------------------------------------------------------------------------
// Faster implementation of GetCellBounds().  Bounds are calculated without
// constructing a cell.
void svtkUnstructuredGrid::GetCellBounds(svtkIdType cellId, double bounds[6])
{
  // Fast path for float/double:
  using svtkArrayDispatch::Reals;
  using Dispatcher = svtkArrayDispatch::DispatchByValueType<Reals>;
  ComputeCellBoundsWorker worker;

  svtkDataArray* ptArray = this->Points->GetData();
  if (!Dispatcher::Execute(ptArray, worker, this->Connectivity, cellId, bounds))
  { // fallback for weird types:
    worker(ptArray, this->Connectivity, cellId, bounds);
  }
}

//----------------------------------------------------------------------------
// Return the number of points from the cell defined by the maximum number of
// points/
int svtkUnstructuredGrid::GetMaxCellSize()
{
  if (this->Connectivity)
  { // The internal implementation is threaded.
    return this->Connectivity->GetMaxCellSize();
  }
  else
  {
    return 0;
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkUnstructuredGrid::GetNumberOfCells()
{
  svtkDebugMacro(<< "NUMBER OF CELLS = "
                << (this->Connectivity ? this->Connectivity->GetNumberOfCells() : 0));
  return (this->Connectivity ? this->Connectivity->GetNumberOfCells() : 0);
}

//----------------------------------------------------------------------------
// Insert/create cell in object by type and list of point ids defining
// cell topology. Using a special input format, this function also support
// polyhedron cells.
svtkIdType svtkUnstructuredGrid::InternalInsertNextCell(int type, svtkIdList* ptIds)
{
  if (type == SVTK_POLYHEDRON)
  {
    // For polyhedron cell, input ptIds is of format:
    // (numCellFaces, numFace0Pts, id1, id2, id3, numFace1Pts,id1, id2, id3, ...)
    svtkIdType* dataPtr = ptIds->GetPointer(0);
    return this->InsertNextCell(type, dataPtr[0], dataPtr + 1);
  }

  this->Connectivity->InsertNextCell(ptIds);

  // If faces have been created, we need to pad them (we are not creating
  // a polyhedral cell in this method)
  if (this->FaceLocations)
  {
    this->FaceLocations->InsertNextValue(-1);
  }

  // insert cell type
  return this->Types->InsertNextValue(static_cast<unsigned char>(type));
}

//----------------------------------------------------------------------------
// Insert/create cell in object by type and list of point ids defining
// cell topology. Using a special input format, this function also support
// polyhedron cells.
svtkIdType svtkUnstructuredGrid::InternalInsertNextCell(
  int type, svtkIdType npts, const svtkIdType ptIds[])
{
  if (type != SVTK_POLYHEDRON)
  {
    // insert connectivity
    this->Connectivity->InsertNextCell(npts, ptIds);

    // If faces have been created, we need to pad them (we are not creating
    // a polyhedral cell in this method)
    if (this->FaceLocations)
    {
      this->FaceLocations->InsertNextValue(-1);
    }
  }
  else
  {
    // For polyhedron, npts is actually number of faces, ptIds is of format:
    // (numFace0Pts, id1, id2, id3, numFace1Pts,id1, id2, id3, ...)
    svtkIdType realnpts;

    // We defer allocation for the faces because they are not commonly used and
    // we only want to allocate when necessary.
    if (!this->Faces)
    {
      this->Faces = svtkSmartPointer<svtkIdTypeArray>::New();
      this->Faces->Allocate(this->Types->GetSize());
      this->FaceLocations = svtkSmartPointer<svtkIdTypeArray>::New();
      this->FaceLocations->Allocate(this->Types->GetSize());
      // FaceLocations must be padded until the current position
      for (svtkIdType i = 0; i <= this->Types->GetMaxId(); i++)
      {
        this->FaceLocations->InsertNextValue(-1);
      }
    }

    // insert face location
    this->FaceLocations->InsertNextValue(this->Faces->GetMaxId() + 1);

    // insert cell connectivity and faces stream
    svtkUnstructuredGrid::DecomposeAPolyhedronCell(
      npts, ptIds, realnpts, this->Connectivity, this->Faces);
  }

  return this->Types->InsertNextValue(static_cast<unsigned char>(type));
}

//----------------------------------------------------------------------------
// Insert/create cell in object by type and list of point and face ids
// defining cell topology. This method is meant for face-explicit cells (e.g.
// polyhedron).
svtkIdType svtkUnstructuredGrid::InternalInsertNextCell(
  int type, svtkIdType npts, const svtkIdType pts[], svtkIdType nfaces, const svtkIdType faces[])
{
  if (type != SVTK_POLYHEDRON)
  {
    return this->InsertNextCell(type, npts, pts);
  }
  // Insert connectivity (points that make up polyhedron)
  this->Connectivity->InsertNextCell(npts, pts);

  // Now insert faces; allocate storage if necessary.
  // We defer allocation for the faces because they are not commonly used and
  // we only want to allocate when necessary.
  if (!this->Faces)
  {
    this->Faces = svtkSmartPointer<svtkIdTypeArray>::New();
    this->Faces->Allocate(this->Types->GetSize());
    this->FaceLocations = svtkSmartPointer<svtkIdTypeArray>::New();
    this->FaceLocations->Allocate(this->Types->GetSize());
    // FaceLocations must be padded until the current position
    for (svtkIdType i = 0; i <= this->Types->GetMaxId(); i++)
    {
      this->FaceLocations->InsertNextValue(-1);
    }
  }

  // Okay the faces go in
  this->FaceLocations->InsertNextValue(this->Faces->GetMaxId() + 1);
  this->Faces->InsertNextValue(nfaces);

  for (int faceNum = 0; faceNum < nfaces; ++faceNum)
  {
    npts = faces[0];
    this->Faces->InsertNextValue(npts);
    for (svtkIdType i = 1; i <= npts; ++i)
    {
      this->Faces->InsertNextValue(faces[i]);
    }
    faces += npts + 1;
  } // for all faces

  return this->Types->InsertNextValue(static_cast<unsigned char>(type));
}

//----------------------------------------------------------------------------
int svtkUnstructuredGrid::InitializeFacesRepresentation(svtkIdType numPrevCells)
{
  if (this->Faces || this->FaceLocations)
  {
    svtkErrorMacro("Face information already exist for this unstuructured grid. "
                  "InitializeFacesRepresentation returned without execution.");
    return 0;
  }

  this->Faces = svtkSmartPointer<svtkIdTypeArray>::New();
  this->Faces->Allocate(this->Types->GetSize());

  this->FaceLocations = svtkSmartPointer<svtkIdTypeArray>::New();
  this->FaceLocations->Allocate(this->Types->GetSize());
  // FaceLocations must be padded until the current position
  for (svtkIdType i = 0; i < numPrevCells; i++)
  {
    this->FaceLocations->InsertNextValue(-1);
  }

  return 1;
}

//----------------------------------------------------------------------------
svtkMTimeType svtkUnstructuredGrid::GetMeshMTime()
{
  return svtkMath::Max(this->Points ? this->Points->GetMTime() : 0,
    this->Connectivity ? this->Connectivity->GetMTime() : 0);
}

//----------------------------------------------------------------------------
// Return faces for a polyhedral cell (or face-explicit cell).
svtkIdType* svtkUnstructuredGrid::GetFaces(svtkIdType cellId)
{
  // Get the locations of the face
  svtkIdType loc;
  if (!this->Faces || cellId < 0 || cellId > this->FaceLocations->GetMaxId() ||
    (loc = this->FaceLocations->GetValue(cellId)) == -1)
  {
    return nullptr;
  }

  return this->Faces->GetPointer(loc);
}

//----------------------------------------------------------------------------
svtkIdTypeArray* svtkUnstructuredGrid::GetFaces()
{
  return this->Faces;
}

//----------------------------------------------------------------------------
svtkIdTypeArray* svtkUnstructuredGrid::GetFaceLocations()
{
  return this->FaceLocations;
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::SetCells(int type, svtkCellArray* cells)
{
  svtkNew<svtkUnsignedCharArray> types;
  types->SetNumberOfComponents(1);
  types->SetNumberOfValues(cells->GetNumberOfCells());
  types->FillValue(static_cast<unsigned char>(type));

  this->SetCells(types, cells);
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::SetCells(int* types, svtkCellArray* cells)
{
  const svtkIdType ncells = cells->GetNumberOfCells();

  // Convert the types into a svtkUnsignedCharArray:
  svtkNew<svtkUnsignedCharArray> cellTypes;
  cellTypes->SetNumberOfTuples(ncells);
  auto typeRange = svtk::DataArrayValueRange<1>(cellTypes);
  std::transform(types, types + ncells, typeRange.begin(),
    [](int t) -> unsigned char { return static_cast<unsigned char>(t); });

  this->SetCells(cellTypes, cells);
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::SetCells(svtkUnsignedCharArray* cellTypes, svtkCellArray* cells)
{
  // check if cells contain any polyhedron cell
  const svtkIdType ncells = cells->GetNumberOfCells();
  const auto typeRange = svtk::DataArrayValueRange<1>(cellTypes);
  const bool containPolyhedron =
    std::find(typeRange.cbegin(), typeRange.cend(), SVTK_POLYHEDRON) != typeRange.cend();

  if (!containPolyhedron)
  {
    this->SetCells(cellTypes, cells, nullptr, nullptr);
    return;
  }

  // If a polyhedron cell exists, its input cellArray is of special format.
  // [nCell0Faces, nFace0Pts, i, j, k, nFace1Pts, i, j, k, ...]
  // We need to convert it into new cell connectivities of standard format,
  // update cellLocations as well as create faces and facelocations.
  svtkNew<svtkCellArray> newCells;
  newCells->AllocateExact(ncells, cells->GetNumberOfConnectivityIds());

  svtkNew<svtkIdTypeArray> faces;
  faces->Allocate(ncells + cells->GetNumberOfConnectivityIds());

  svtkNew<svtkIdTypeArray> faceLocations;
  faceLocations->Allocate(ncells);

  auto cellIter = svtkSmartPointer<svtkCellArrayIterator>::Take(cells->NewIterator());

  for (cellIter->GoToFirstCell(); !cellIter->IsDoneWithTraversal(); cellIter->GoToNextCell())
  {
    svtkIdType npts;
    const svtkIdType* pts;
    cellIter->GetCurrentCell(npts, pts);
    const svtkIdType cellId = cellIter->GetCurrentCellId();

    if (cellTypes->GetValue(cellId) != SVTK_POLYHEDRON)
    {
      newCells->InsertNextCell(npts, pts);
      faceLocations->InsertNextValue(-1);
    }
    else
    {
      svtkIdType realnpts;
      svtkIdType nfaces;
      faceLocations->InsertNextValue(faces->GetMaxId() + 1);
      svtkUnstructuredGrid::DecomposeAPolyhedronCell(pts, realnpts, nfaces, newCells, faces);
    }
  }

  this->SetCells(cellTypes, newCells, faceLocations, faces);
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::SetCells(svtkUnsignedCharArray* cellTypes, svtkCellArray* cells,
  svtkIdTypeArray* faceLocations, svtkIdTypeArray* faces)
{
  this->Connectivity = cells;
  this->Types = cellTypes;
  this->DistinctCellTypes = nullptr;
  this->DistinctCellTypesUpdateMTime = 0;
  this->Faces = faces;
  this->FaceLocations = faceLocations;
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::BuildLinks()
{
  // Create appropriate locator. Currently it's either a svtkCellLocator (when
  // the dataset is editable) or svtkStaticCellLocator (when the dataset is
  // not editable).
  svtkIdType numPts = this->GetNumberOfPoints();
  if (!this->Editable)
  {
    this->Links = svtkSmartPointer<svtkStaticCellLinks>::New();
  }
  else
  {
    svtkNew<svtkCellLinks> links;
    links->Allocate(numPts);
    this->Links = std::move(links);
  }

  this->Links->BuildLinks(this);
}

//----------------------------------------------------------------------------
svtkAbstractCellLinks* svtkUnstructuredGrid::GetCellLinks()
{
  return this->Links;
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::GetPointCells(svtkIdType ptId, svtkIdType& ncells, svtkIdType*& cells)
{
  if (!this->Editable)
  {
    svtkStaticCellLinks* links = static_cast<svtkStaticCellLinks*>(this->Links.Get());

    ncells = links->GetNcells(ptId);
    cells = links->GetCells(ptId);
  }
  else
  {
    svtkCellLinks* links = static_cast<svtkCellLinks*>(this->Links.Get());

    ncells = links->GetNcells(ptId);
    cells = links->GetCells(ptId);
  }
}

//----------------------------------------------------------------------------
#ifndef SVTK_LEGACY_REMOVE
void svtkUnstructuredGrid::GetPointCells(svtkIdType ptId, unsigned short& ncells, svtkIdType*& cells)
{
  SVTK_LEGACY_BODY(svtkUnstructuredGrid::GetPointCells, "SVTK 9.0");
  svtkIdType nc;
  this->GetPointCells(ptId, nc, cells);
  ncells = static_cast<unsigned short>(nc);
}
#endif

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::GetCellPoints(svtkIdType cellId, svtkIdList* ptIds)
{
  this->Connectivity->GetCellAtId(cellId, ptIds);
}

namespace
{
class DistinctCellTypesWorker
{
public:
  DistinctCellTypesWorker(svtkUnstructuredGrid* grid)
    : Grid(grid)
  {
  }

  svtkUnstructuredGrid* Grid;
  std::set<unsigned char> DistinctCellTypes;

  // Thread-local storage
  svtkSMPThreadLocal<std::set<unsigned char> > LocalDistinctCellTypes;

  void Initialize() {}

  void operator()(svtkIdType begin, svtkIdType end)
  {
    if (!this->Grid)
    {
      return;
    }

    for (svtkIdType idx = begin; idx < end; ++idx)
    {
      unsigned char cellType = static_cast<unsigned char>(this->Grid->GetCellType(idx));
      this->LocalDistinctCellTypes.Local().insert(cellType);
    }
  }

  void Reduce()
  {
    this->DistinctCellTypes.clear();
    for (svtkSMPThreadLocal<std::set<unsigned char> >::iterator iter =
           this->LocalDistinctCellTypes.begin();
         iter != this->LocalDistinctCellTypes.end(); ++iter)
    {
      this->DistinctCellTypes.insert(iter->begin(), iter->end());
    }
  }
};
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::GetCellTypes(svtkCellTypes* types)
{
  if (this->Types == nullptr)
  {
    // No cell types
    return;
  }

  if (this->DistinctCellTypes == nullptr ||
    this->Types->GetMTime() > this->DistinctCellTypesUpdateMTime)
  {
    // Update the list of cell types
    DistinctCellTypesWorker cellTypesWorker(this);
    svtkSMPTools::For(0, this->GetNumberOfCells(), cellTypesWorker);

    if (this->DistinctCellTypes)
    {
      this->DistinctCellTypes->Reset();
    }
    else
    {
      this->DistinctCellTypes = svtkSmartPointer<svtkCellTypes>::New();
      this->DistinctCellTypes->Register(this);
      this->DistinctCellTypes->Delete();
    }
    this->DistinctCellTypes->Allocate(static_cast<int>(cellTypesWorker.DistinctCellTypes.size()));

    for (auto cellType : cellTypesWorker.DistinctCellTypes)
    {
      this->DistinctCellTypes->InsertNextType(cellType);
    }

    this->DistinctCellTypesUpdateMTime = this->Types->GetMTime();
  }

  types->DeepCopy(this->DistinctCellTypes);
}

//----------------------------------------------------------------------------
svtkUnsignedCharArray* svtkUnstructuredGrid::GetCellTypesArray()
{
  return this->Types;
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::GetFaceStream(svtkIdType cellId, svtkIdList* ptIds)
{
  if (this->GetCellType(cellId) != SVTK_POLYHEDRON)
  {
    this->GetCellPoints(cellId, ptIds);
    return;
  }

  ptIds->Reset();

  if (!this->Faces || !this->FaceLocations)
  {
    return;
  }

  svtkIdType loc = this->FaceLocations->GetValue(cellId);
  svtkIdType* facePtr = this->Faces->GetPointer(loc);

  svtkIdType nfaces = *facePtr++;
  ptIds->InsertNextId(nfaces);
  for (svtkIdType i = 0; i < nfaces; i++)
  {
    svtkIdType npts = *facePtr++;
    ptIds->InsertNextId(npts);
    for (svtkIdType j = 0; j < npts; j++)
    {
      ptIds->InsertNextId(*facePtr++);
    }
  }
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::GetFaceStream(
  svtkIdType cellId, svtkIdType& nfaces, svtkIdType const*& ptIds)
{
  if (this->GetCellType(cellId) != SVTK_POLYHEDRON)
  {
    this->GetCellPoints(cellId, nfaces, ptIds);
    return;
  }

  if (!this->Faces || !this->FaceLocations)
  {
    return;
  }

  svtkIdType loc = this->FaceLocations->GetValue(cellId);
  const svtkIdType* facePtr = this->Faces->GetPointer(loc);

  nfaces = *facePtr;
  ptIds = facePtr + 1;
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::GetPointCells(svtkIdType ptId, svtkIdList* cellIds)
{
  if (!this->Links)
  {
    this->BuildLinks();
  }
  cellIds->Reset();

  svtkIdType numCells, *cells;
  if (!this->Editable)
  {
    svtkStaticCellLinks* links = static_cast<svtkStaticCellLinks*>(this->Links.Get());
    numCells = links->GetNcells(ptId);
    cells = links->GetCells(ptId);
  }
  else
  {
    svtkCellLinks* links = static_cast<svtkCellLinks*>(this->Links.Get());
    numCells = links->GetNcells(ptId);
    cells = links->GetCells(ptId);
  }

  cellIds->SetNumberOfIds(numCells);
  for (auto i = 0; i < numCells; i++)
  {
    cellIds->SetId(i, cells[i]);
  }
}

//----------------------------------------------------------------------------
svtkCellIterator* svtkUnstructuredGrid::NewCellIterator()
{
  svtkUnstructuredGridCellIterator* iter(svtkUnstructuredGridCellIterator::New());
  iter->SetUnstructuredGrid(this);
  return iter;
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::Reset()
{
  if (this->Connectivity)
  {
    this->Connectivity->Reset();
  }
  if (this->Links)
  {
    this->Links->Reset();
  }
  if (this->Types)
  {
    this->Types->Reset();
  }
  if (this->DistinctCellTypes)
  {
    this->DistinctCellTypes->Reset();
  }
  if (this->Faces)
  {
    this->Faces->Reset();
  }
  if (this->FaceLocations)
  {
    this->FaceLocations->Reset();
  }
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::Squeeze()
{
  if (this->Connectivity)
  {
    this->Connectivity->Squeeze();
  }
  if (this->Links)
  {
    this->Links->Squeeze();
  }
  if (this->Types)
  {
    this->Types->Squeeze();
  }
  if (this->Faces)
  {
    this->Faces->Squeeze();
  }
  if (this->FaceLocations)
  {
    this->FaceLocations->Squeeze();
  }

  svtkPointSet::Squeeze();
}

//----------------------------------------------------------------------------
// Remove a reference to a cell in a particular point's link list. You may
// also consider using RemoveCellReference() to remove the references from
// all the cell's points to the cell. This operator does not reallocate
// memory; use the operator ResizeCellList() to do this if necessary. Note that
// dataset should be set to "Editable".
void svtkUnstructuredGrid::RemoveReferenceToCell(svtkIdType ptId, svtkIdType cellId)
{
  static_cast<svtkCellLinks*>(this->Links.Get())->RemoveCellReference(cellId, ptId);
}

//----------------------------------------------------------------------------
// Add a reference to a cell in a particular point's link list. (You may also
// consider using AddCellReference() to add the references from all the
// cell's points to the cell.) This operator does not realloc memory; use the
// operator ResizeCellList() to do this if necessary. Note that dataset
// should be set to "Editable".
void svtkUnstructuredGrid::AddReferenceToCell(svtkIdType ptId, svtkIdType cellId)
{
  static_cast<svtkCellLinks*>(this->Links.Get())->AddCellReference(cellId, ptId);
}

//----------------------------------------------------------------------------
// Resize the list of cells using a particular point. (This operator assumes
// that BuildLinks() has been called.) Note that dataset should be set to
// "Editable".
void svtkUnstructuredGrid::ResizeCellList(svtkIdType ptId, int size)
{
  static_cast<svtkCellLinks*>(this->Links.Get())->ResizeCellList(ptId, size);
}

//----------------------------------------------------------------------------
// Replace the points defining cell "cellId" with a new set of points. This
// operator is (typically) used when links from points to cells have not been
// built (i.e., BuildLinks() has not been executed). Use the operator
// ReplaceLinkedCell() to replace a cell when cell structure has been built.
void svtkUnstructuredGrid::InternalReplaceCell(svtkIdType cellId, int npts, const svtkIdType pts[])
{
  this->Connectivity->ReplaceCellAtId(cellId, npts, pts);
}

//----------------------------------------------------------------------------
// Add a new cell to the cell data structure (after cell links have been
// built). This method adds the cell and then updates the links from the points
// to the cells. (Memory is allocated as necessary.) Note that the dataset must
// be in "Editable" mode.
svtkIdType svtkUnstructuredGrid::InsertNextLinkedCell(int type, int npts, const svtkIdType pts[])
{
  svtkIdType i, id;

  id = this->InsertNextCell(type, npts, pts);

  svtkCellLinks* clinks = static_cast<svtkCellLinks*>(this->Links.Get());
  for (i = 0; i < npts; i++)
  {
    clinks->ResizeCellList(pts[i], 1);
    clinks->AddCellReference(id, pts[i]);
  }

  return id;
}

//----------------------------------------------------------------------------
unsigned long svtkUnstructuredGrid::GetActualMemorySize()
{
  unsigned long size = this->svtkPointSet::GetActualMemorySize();
  if (this->Connectivity)
  {
    size += this->Connectivity->GetActualMemorySize();
  }

  if (this->Links)
  {
    size += this->Links->GetActualMemorySize();
  }

  if (this->Types)
  {
    size += this->Types->GetActualMemorySize();
  }

  if (this->Faces)
  {
    size += this->Faces->GetActualMemorySize();
  }

  if (this->FaceLocations)
  {
    size += this->FaceLocations->GetActualMemorySize();
  }

  return size;
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::ShallowCopy(svtkDataObject* dataObject)
{
  if (svtkUnstructuredGrid* grid = svtkUnstructuredGrid::SafeDownCast(dataObject))
  {
    // I do not know if this is correct but.
    // ^ I really hope this comment lives for another 20 years.

    this->Connectivity = grid->Connectivity;
    this->Links = grid->Links;
    this->Types = grid->Types;
    this->DistinctCellTypes = nullptr;
    this->DistinctCellTypesUpdateMTime = 0;
    this->Faces = grid->Faces;
    this->FaceLocations = grid->FaceLocations;
  }
  else if (svtkUnstructuredGridBase* ugb = svtkUnstructuredGridBase::SafeDownCast(dataObject))
  {
    // The source object has svtkUnstructuredGrid topology, but a different
    // cell implementation. Deep copy the cells, and shallow copy the rest:
    svtkSmartPointer<svtkCellIterator> cellIter =
      svtkSmartPointer<svtkCellIterator>::Take(ugb->NewCellIterator());
    for (cellIter->InitTraversal(); !cellIter->IsDoneWithTraversal(); cellIter->GoToNextCell())
    {
      this->InsertNextCell(cellIter->GetCellType(), cellIter->GetNumberOfPoints(),
        cellIter->GetPointIds()->GetPointer(0), cellIter->GetNumberOfFaces(),
        cellIter->GetFaces()->GetPointer(1));
    }
  }

  this->Superclass::ShallowCopy(dataObject);
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::DeepCopy(svtkDataObject* dataObject)
{
  svtkUnstructuredGrid* grid = svtkUnstructuredGrid::SafeDownCast(dataObject);

  if (grid != nullptr)
  {
    if (grid->Connectivity)
    {
      this->Connectivity = svtkSmartPointer<svtkCellArray>::New();
      this->Connectivity->DeepCopy(grid->Connectivity);
    }
    else
    {
      this->Connectivity = nullptr;
    }

    if (grid->Types)
    {
      this->Types = svtkSmartPointer<svtkUnsignedCharArray>::New();
      this->Types->DeepCopy(grid->Types);
    }
    else
    {
      this->Types = nullptr;
    }

    if (grid->DistinctCellTypes)
    {
      this->DistinctCellTypes = svtkSmartPointer<svtkCellTypes>::New();
      this->DistinctCellTypes->DeepCopy(grid->DistinctCellTypes);
    }
    else
    {
      this->DistinctCellTypes = nullptr;
    }

    if (grid->Faces)
    {
      this->Faces = svtkSmartPointer<svtkIdTypeArray>::New();
      this->Faces->DeepCopy(grid->Faces);
    }
    else
    {
      this->Faces = nullptr;
    }

    if (grid->FaceLocations)
    {
      this->FaceLocations = svtkSmartPointer<svtkIdTypeArray>::New();
      this->FaceLocations->DeepCopy(grid->FaceLocations);
    }
    else
    {
      this->FaceLocations = nullptr;
    }

    // Skip the unstructured grid base implementation, as it uses a less
    // efficient method of copying cell data.
    this->svtkUnstructuredGridBase::Superclass::DeepCopy(grid);
  }
  else
  {
    // Use the svtkUnstructuredGridBase deep copy implementation.
    this->Superclass::DeepCopy(dataObject);
  }

  // Finally Build Links if we need to
  if (grid && grid->Links)
  {
    this->BuildLinks();
  }
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Number Of Pieces: " << this->GetNumberOfPieces() << endl;
  os << indent << "Piece: " << this->GetPiece() << endl;
  os << indent << "Ghost Level: " << this->GetGhostLevel() << endl;
}

//----------------------------------------------------------------------------
bool svtkUnstructuredGrid::AllocateExact(svtkIdType numCells, svtkIdType connectivitySize)
{
  if (numCells < 1)
  {
    numCells = 1024;
  }
  if (connectivitySize < 1)
  {
    connectivitySize = 1024;
  }

  this->DistinctCellTypesUpdateMTime = 0;
  this->DistinctCellTypes = svtkSmartPointer<svtkCellTypes>::New();
  this->Types = svtkSmartPointer<svtkUnsignedCharArray>::New();
  this->Connectivity = svtkSmartPointer<svtkCellArray>::New();

  bool result = this->Connectivity->AllocateExact(numCells, connectivitySize);
  if (result)
  {
    result = this->Types->Allocate(numCells) != 0;
  }
  if (result)
  {
    result = this->DistinctCellTypes->Allocate(SVTK_NUMBER_OF_CELL_TYPES) != 0;
  }

  return result;
}

//----------------------------------------------------------------------------
// Return the cells that use the ptIds provided. This is a set (intersection)
// operation - it can have significant performance impacts on certain filters
// like svtkGeometryFilter. It would be nice to make this faster.
void svtkUnstructuredGrid::GetCellNeighbors(svtkIdType cellId, svtkIdList* ptIds, svtkIdList* cellIds)
{
  // Ensure links are built.
  if (!this->Links)
  {
    this->BuildLinks();
  }
  cellIds->Reset();

  // Ensure that a proper neighborhood request is made.
  svtkIdType numPts = ptIds->GetNumberOfIds();
  if (numPts <= 0)
  {
    return;
  }

  svtkIdType numCells, ptId, *pts = ptIds->GetPointer(0);
  int minNumCells = SVTK_INT_MAX;
  svtkIdType* minCells = nullptr;
  svtkIdType minPtId = 0;

  // Find the point used by the fewest number of cells as a starting set.
  // Note the explicit cast to locator type. Several experiments were
  // undertaken using virtual methods, switch statements, etc. but there were
  // significant performance impacts. This includes using different
  // instantiations of svtkStaticCellLinksTemplate<> of various types (to
  // reduce memory footprint) - the lesson learned is that the code here is
  // very sensitive to compiler optimizations so if you make changes, make
  // sure to test the impacts on performance.
  if (!this->Editable)
  {
    svtkStaticCellLinks* links = static_cast<svtkStaticCellLinks*>(this->Links.Get());

    for (auto i = 0; i < numPts; ++i)
    {
      ptId = pts[i];
      numCells = links->GetNcells(ptId);
      if (numCells < minNumCells)
      {
        minNumCells = numCells;
        minPtId = ptId;
      }
    }
    minCells = links->GetCells(minPtId);
  }
  else
  {
    svtkCellLinks* links = static_cast<svtkCellLinks*>(this->Links.Get());

    for (auto i = 0; i < numPts; ++i)
    {
      ptId = pts[i];
      numCells = links->GetNcells(ptId);
      if (numCells < minNumCells)
      {
        minNumCells = numCells;
        minPtId = ptId;
      }
    }
    minCells = links->GetCells(minPtId);
  }

  // Now for each cell, see if it contains all the points
  // in the ptIds list. If so, add the cellId to the neighbor list.
  bool match;
  for (auto i = 0; i < minNumCells; ++i)
  {
    if (minCells[i] != cellId) // don't include current cell
    {
      const svtkIdType* cellPts;
      svtkIdType npts;
      this->GetCellPoints(minCells[i], npts, cellPts);
      match = true;
      for (auto j = 0; j < numPts && match; ++j) // for all pts in input cell
      {
        if (pts[j] != minPtId) // of course minPtId is contained by cell
        {
          match = false;
          for (auto k = 0; k < npts; ++k) // for all points in candidate cell
          {
            if (pts[j] == cellPts[k])
            {
              match = true; // a match was found
              break;
            }
          } // for all points in current cell
        }   // if not guaranteed match
      }     // for all points in input cell
      if (match)
      {
        cellIds->InsertNextId(minCells[i]);
      }
    } // if not the reference cell
  }   // for all candidate cells attached to point
}

//----------------------------------------------------------------------------
int svtkUnstructuredGrid::IsHomogeneous()
{
  unsigned char type;
  if (this->Types && this->Types->GetMaxId() >= 0)
  {
    type = Types->GetValue(0);
    svtkIdType numCells = this->GetNumberOfCells();
    for (svtkIdType cellId = 0; cellId < numCells; ++cellId)
    {
      if (this->Types->GetValue(cellId) != type)
      {
        return 0;
      }
    }
    return 1;
  }
  return 0;
}

//----------------------------------------------------------------------------
// Fill container with indices of cells which match given type.
void svtkUnstructuredGrid::GetIdsOfCellsOfType(int type, svtkIdTypeArray* array)
{
  for (int cellId = 0; cellId < this->GetNumberOfCells(); cellId++)
  {
    if (static_cast<int>(Types->GetValue(cellId)) == type)
    {
      array->InsertNextValue(cellId);
    }
  }
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::RemoveGhostCells()
{
  svtkUnstructuredGrid* newGrid = svtkUnstructuredGrid::New();
  svtkUnsignedCharArray* temp;
  unsigned char* cellGhosts;

  svtkIdType cellId, newCellId;
  svtkIdList *cellPts, *pointMap;
  svtkIdList* newCellPts;
  svtkCell* cell;
  svtkPoints* newPoints;
  svtkIdType i, ptId, newId, numPts;
  svtkIdType numCellPts;
  double* x;
  svtkPointData* pd = this->GetPointData();
  svtkPointData* outPD = newGrid->GetPointData();
  svtkCellData* cd = this->GetCellData();
  svtkCellData* outCD = newGrid->GetCellData();

  // Get a pointer to the cell ghost array.
  temp = this->GetCellGhostArray();
  if (temp == nullptr)
  {
    svtkDebugMacro("Could not find cell ghost array.");
    newGrid->Delete();
    return;
  }
  if ((temp->GetNumberOfComponents() != 1) ||
    (temp->GetNumberOfTuples() < this->GetNumberOfCells()))
  {
    svtkErrorMacro("Poorly formed ghost array.");
    newGrid->Delete();
    return;
  }
  cellGhosts = temp->GetPointer(0);

  // Now threshold based on the cell ghost array.

  // ensure that all attributes are copied over, including global ids.
  outPD->CopyAllOn(svtkDataSetAttributes::COPYTUPLE);
  outCD->CopyAllOn(svtkDataSetAttributes::COPYTUPLE);

  outPD->CopyAllocate(pd);
  outCD->CopyAllocate(cd);

  numPts = this->GetNumberOfPoints();
  newGrid->Allocate(this->GetNumberOfCells());
  newPoints = svtkPoints::New();
  newPoints->SetDataType(this->GetPoints()->GetDataType());
  newPoints->Allocate(numPts);

  pointMap = svtkIdList::New(); // maps old point ids into new
  pointMap->SetNumberOfIds(numPts);
  for (i = 0; i < numPts; i++)
  {
    pointMap->SetId(i, -1);
  }

  newCellPts = svtkIdList::New();

  // Check that the scalars of each cell satisfy the threshold criterion
  for (cellId = 0; cellId < this->GetNumberOfCells(); cellId++)
  {
    cell = this->GetCell(cellId);
    cellPts = cell->GetPointIds();
    numCellPts = cell->GetNumberOfPoints();

    if ((cellGhosts[cellId] & svtkDataSetAttributes::DUPLICATECELL) == 0) // Keep the cell.
    {
      for (i = 0; i < numCellPts; i++)
      {
        ptId = cellPts->GetId(i);
        if ((newId = pointMap->GetId(ptId)) < 0)
        {
          x = this->GetPoint(ptId);
          newId = newPoints->InsertNextPoint(x);
          pointMap->SetId(ptId, newId);
          outPD->CopyData(pd, ptId, newId);
        }
        newCellPts->InsertId(i, newId);
      }
      newCellId = newGrid->InsertNextCell(cell->GetCellType(), newCellPts);
      outCD->CopyData(cd, cellId, newCellId);
      newCellPts->Reset();
    } // satisfied thresholding
  }   // for all cells

  // now clean up / update ourselves
  pointMap->Delete();
  newCellPts->Delete();

  newGrid->SetPoints(newPoints);
  newPoints->Delete();

  this->CopyStructure(newGrid);
  this->GetPointData()->ShallowCopy(newGrid->GetPointData());
  this->GetCellData()->ShallowCopy(newGrid->GetCellData());
  newGrid->Delete();
  newGrid = nullptr;

  this->Squeeze();
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::DecomposeAPolyhedronCell(svtkCellArray* polyhedronCell,
  svtkIdType& numCellPts, svtkIdType& nCellfaces, svtkCellArray* cellArray, svtkIdTypeArray* faces)
{
  const svtkIdType* cellStream = nullptr;
  svtkIdType cellLength = 0;

  polyhedronCell->InitTraversal();
  polyhedronCell->GetNextCell(cellLength, cellStream);

  svtkUnstructuredGrid::DecomposeAPolyhedronCell(
    cellStream, numCellPts, nCellfaces, cellArray, faces);
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::DecomposeAPolyhedronCell(const svtkIdType* cellStream,
  svtkIdType& numCellPts, svtkIdType& nCellFaces, svtkCellArray* cellArray, svtkIdTypeArray* faces)
{
  nCellFaces = cellStream[0];
  if (nCellFaces <= 0)
  {
    return;
  }

  svtkUnstructuredGrid::DecomposeAPolyhedronCell(
    nCellFaces, cellStream + 1, numCellPts, cellArray, faces);
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::DecomposeAPolyhedronCell(svtkIdType nCellFaces,
  const svtkIdType cellStream[], svtkIdType& numCellPts, svtkCellArray* cellArray,
  svtkIdTypeArray* faces)
{
  std::set<svtkIdType> cellPointSet;
  std::set<svtkIdType>::iterator it;

  // insert number of faces into the face array
  faces->InsertNextValue(nCellFaces);

  // for each face
  for (svtkIdType fid = 0; fid < nCellFaces; fid++)
  {
    // extract all points on the same face, store them into a set
    svtkIdType npts = *cellStream++;
    faces->InsertNextValue(npts);
    for (svtkIdType i = 0; i < npts; i++)
    {
      svtkIdType pid = *cellStream++;
      faces->InsertNextValue(pid);
      cellPointSet.insert(pid);
    }
  }

  // standard cell connectivity array that stores the number of points plus
  // a list of point ids.
  cellArray->InsertNextCell(static_cast<int>(cellPointSet.size()));
  for (it = cellPointSet.begin(); it != cellPointSet.end(); ++it)
  {
    cellArray->InsertCellPoint(*it);
  }

  // the real number of points in the polyhedron cell.
  numCellPts = static_cast<svtkIdType>(cellPointSet.size());
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::ConvertFaceStreamPointIds(svtkIdList* faceStream, svtkIdType* idMap)
{
  svtkIdType* idPtr = faceStream->GetPointer(0);
  svtkIdType nfaces = *idPtr++;
  for (svtkIdType i = 0; i < nfaces; i++)
  {
    svtkIdType npts = *idPtr++;
    for (svtkIdType j = 0; j < npts; j++)
    {
      *idPtr = idMap[*idPtr];
      idPtr++;
    }
  }
}

//----------------------------------------------------------------------------
void svtkUnstructuredGrid::ConvertFaceStreamPointIds(
  svtkIdType nfaces, svtkIdType* faceStream, svtkIdType* idMap)
{
  svtkIdType* idPtr = faceStream;
  for (svtkIdType i = 0; i < nfaces; i++)
  {
    svtkIdType npts = *idPtr++;
    for (svtkIdType j = 0; j < npts; j++)
    {
      *idPtr = idMap[*idPtr];
      idPtr++;
    }
  }
}

//----------------------------------------------------------------------------
svtkUnstructuredGrid* svtkUnstructuredGrid::GetData(svtkInformation* info)
{
  return info ? svtkUnstructuredGrid::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkUnstructuredGrid* svtkUnstructuredGrid::GetData(svtkInformationVector* v, int i)
{
  return svtkUnstructuredGrid::GetData(v->GetInformationObject(i));
}
