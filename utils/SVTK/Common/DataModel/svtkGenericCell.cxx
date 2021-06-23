/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericCell.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkGenericCell.h"

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
#include "svtkConvexPointSet.h"
#include "svtkCubicLine.h"
#include "svtkEmptyCell.h"
#include "svtkHexagonalPrism.h"
#include "svtkHexahedron.h"
#include "svtkIncrementalPointLocator.h"
#include "svtkLagrangeCurve.h"
#include "svtkLagrangeHexahedron.h"
#include "svtkLagrangeQuadrilateral.h"
#include "svtkLagrangeTetra.h"
#include "svtkLagrangeTriangle.h"
#include "svtkLagrangeWedge.h"
#include "svtkLine.h"
#include "svtkObjectFactory.h"
#include "svtkPentagonalPrism.h"
#include "svtkPixel.h"
#include "svtkPoints.h"
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
#include "svtkTetra.h"
#include "svtkTriQuadraticHexahedron.h"
#include "svtkTriangle.h"
#include "svtkTriangleStrip.h"
#include "svtkVertex.h"
#include "svtkVoxel.h"
#include "svtkWedge.h"

svtkStandardNewMacro(svtkGenericCell);

//----------------------------------------------------------------------------
// Construct cell.
svtkGenericCell::svtkGenericCell()
{
  for (int i = 0; i < SVTK_NUMBER_OF_CELL_TYPES; ++i)
  {
    this->CellStore[i] = nullptr;
  }
  this->CellStore[SVTK_EMPTY_CELL] = svtkEmptyCell::New();
  this->Cell = this->CellStore[SVTK_EMPTY_CELL];
  this->Points->Delete();
  this->Points = this->Cell->Points;
  this->Points->Register(this);
  this->PointIds->Delete();
  this->PointIds = this->Cell->PointIds;
  this->PointIds->Register(this);
}

//----------------------------------------------------------------------------
svtkGenericCell::~svtkGenericCell()
{
  for (int i = 0; i < SVTK_NUMBER_OF_CELL_TYPES; ++i)
  {
    if (this->CellStore[i] != nullptr)
    {
      this->CellStore[i]->Delete();
    }
  }
}

//----------------------------------------------------------------------------
void svtkGenericCell::ShallowCopy(svtkCell* c)
{
  this->Cell->ShallowCopy(c);
}

//----------------------------------------------------------------------------
void svtkGenericCell::DeepCopy(svtkCell* c)
{
  this->Cell->DeepCopy(c);
}

//----------------------------------------------------------------------------
int svtkGenericCell::GetCellType()
{
  return this->Cell->GetCellType();
}

//----------------------------------------------------------------------------
int svtkGenericCell::GetCellDimension()
{
  return this->Cell->GetCellDimension();
}

//----------------------------------------------------------------------------
int svtkGenericCell::IsLinear()
{
  return this->Cell->IsLinear();
}

//----------------------------------------------------------------------------
int svtkGenericCell::RequiresInitialization()
{
  return this->Cell->RequiresInitialization();
}

//----------------------------------------------------------------------------
int svtkGenericCell::RequiresExplicitFaceRepresentation()
{
  return this->Cell->RequiresExplicitFaceRepresentation();
}

//----------------------------------------------------------------------------
void svtkGenericCell::SetFaces(svtkIdType* faces)
{
  this->Cell->SetFaces(faces);
}

//----------------------------------------------------------------------------
svtkIdType* svtkGenericCell::GetFaces()
{
  return this->Cell->GetFaces();
}

//----------------------------------------------------------------------------
void svtkGenericCell::Initialize()
{
  this->Cell->Initialize();
}

//----------------------------------------------------------------------------
int svtkGenericCell::GetNumberOfEdges()
{
  return this->Cell->GetNumberOfEdges();
}

//----------------------------------------------------------------------------
int svtkGenericCell::GetNumberOfFaces()
{
  return this->Cell->GetNumberOfFaces();
}

//----------------------------------------------------------------------------
svtkCell* svtkGenericCell::GetEdge(int edgeId)
{
  return this->Cell->GetEdge(edgeId);
}

//----------------------------------------------------------------------------
svtkCell* svtkGenericCell::GetFace(int faceId)
{
  return this->Cell->GetFace(faceId);
}

//----------------------------------------------------------------------------
int svtkGenericCell::CellBoundary(int subId, const double pcoords[3], svtkIdList* pts)
{
  return this->Cell->CellBoundary(subId, pcoords, pts);
}

//----------------------------------------------------------------------------
int svtkGenericCell::EvaluatePosition(const double x[3], double closestPoint[3], int& subId,
  double pcoords[3], double& dist2, double weights[])
{
  return this->Cell->EvaluatePosition(x, closestPoint, subId, pcoords, dist2, weights);
}

//----------------------------------------------------------------------------
void svtkGenericCell::EvaluateLocation(
  int& subId, const double pcoords[3], double x[3], double* weights)
{
  this->Cell->EvaluateLocation(subId, pcoords, x, weights);
}

//----------------------------------------------------------------------------
void svtkGenericCell::Contour(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* verts, svtkCellArray* lines,
  svtkCellArray* polys, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId,
  svtkCellData* outCd)
{
  this->Cell->Contour(
    value, cellScalars, locator, verts, lines, polys, inPd, outPd, inCd, cellId, outCd);
}

//----------------------------------------------------------------------------
void svtkGenericCell::Clip(double value, svtkDataArray* cellScalars,
  svtkIncrementalPointLocator* locator, svtkCellArray* connectivity, svtkPointData* inPd,
  svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd, int insideOut)
{
  this->Cell->Clip(
    value, cellScalars, locator, connectivity, inPd, outPd, inCd, cellId, outCd, insideOut);
}

//----------------------------------------------------------------------------
int svtkGenericCell::IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t,
  double x[3], double pcoords[3], int& subId)
{
  return this->Cell->IntersectWithLine(p1, p2, tol, t, x, pcoords, subId);
}

//----------------------------------------------------------------------------
int svtkGenericCell::Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts)
{
  return this->Cell->Triangulate(index, ptIds, pts);
}

//----------------------------------------------------------------------------
void svtkGenericCell::Derivatives(
  int subId, const double pcoords[3], const double* values, int dim, double* derivs)
{
  this->Cell->Derivatives(subId, pcoords, values, dim, derivs);
}

//----------------------------------------------------------------------------
int svtkGenericCell::GetParametricCenter(double pcoords[3])
{
  return this->Cell->GetParametricCenter(pcoords);
}

//----------------------------------------------------------------------------
double* svtkGenericCell::GetParametricCoords()
{
  return this->Cell->GetParametricCoords();
}

//----------------------------------------------------------------------------
int svtkGenericCell::IsPrimaryCell()
{
  return this->Cell->IsPrimaryCell();
}

//----------------------------------------------------------------------------
svtkCell* svtkGenericCell::InstantiateCell(int cellType)
{
  svtkCell* cell = nullptr;
  switch (cellType)
  {
    case SVTK_EMPTY_CELL:
      cell = svtkEmptyCell::New();
      break;
    case SVTK_VERTEX:
      cell = svtkVertex::New();
      break;
    case SVTK_POLY_VERTEX:
      cell = svtkPolyVertex::New();
      break;
    case SVTK_LINE:
      cell = svtkLine::New();
      break;
    case SVTK_POLY_LINE:
      cell = svtkPolyLine::New();
      break;
    case SVTK_TRIANGLE:
      cell = svtkTriangle::New();
      break;
    case SVTK_TRIANGLE_STRIP:
      cell = svtkTriangleStrip::New();
      break;
    case SVTK_POLYGON:
      cell = svtkPolygon::New();
      break;
    case SVTK_PIXEL:
      cell = svtkPixel::New();
      break;
    case SVTK_QUAD:
      cell = svtkQuad::New();
      break;
    case SVTK_TETRA:
      cell = svtkTetra::New();
      break;
    case SVTK_VOXEL:
      cell = svtkVoxel::New();
      break;
    case SVTK_HEXAHEDRON:
      cell = svtkHexahedron::New();
      break;
    case SVTK_WEDGE:
      cell = svtkWedge::New();
      break;
    case SVTK_PYRAMID:
      cell = svtkPyramid::New();
      break;
    case SVTK_PENTAGONAL_PRISM:
      cell = svtkPentagonalPrism::New();
      break;
    case SVTK_HEXAGONAL_PRISM:
      cell = svtkHexagonalPrism::New();
      break;
    case SVTK_QUADRATIC_EDGE:
      cell = svtkQuadraticEdge::New();
      break;
    case SVTK_QUADRATIC_TRIANGLE:
      cell = svtkQuadraticTriangle::New();
      break;
    case SVTK_QUADRATIC_QUAD:
      cell = svtkQuadraticQuad::New();
      break;
    case SVTK_QUADRATIC_POLYGON:
      cell = svtkQuadraticPolygon::New();
      break;
    case SVTK_QUADRATIC_TETRA:
      cell = svtkQuadraticTetra::New();
      break;
    case SVTK_QUADRATIC_HEXAHEDRON:
      cell = svtkQuadraticHexahedron::New();
      break;
    case SVTK_QUADRATIC_WEDGE:
      cell = svtkQuadraticWedge::New();
      break;
    case SVTK_QUADRATIC_PYRAMID:
      cell = svtkQuadraticPyramid::New();
      break;
    case SVTK_QUADRATIC_LINEAR_QUAD:
      cell = svtkQuadraticLinearQuad::New();
      break;
    case SVTK_BIQUADRATIC_QUAD:
      cell = svtkBiQuadraticQuad::New();
      break;
    case SVTK_TRIQUADRATIC_HEXAHEDRON:
      cell = svtkTriQuadraticHexahedron::New();
      break;
    case SVTK_QUADRATIC_LINEAR_WEDGE:
      cell = svtkQuadraticLinearWedge::New();
      break;
    case SVTK_BIQUADRATIC_QUADRATIC_WEDGE:
      cell = svtkBiQuadraticQuadraticWedge::New();
      break;
    case SVTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON:
      cell = svtkBiQuadraticQuadraticHexahedron::New();
      break;
    case SVTK_BIQUADRATIC_TRIANGLE:
      cell = svtkBiQuadraticTriangle::New();
      break;
    case SVTK_CUBIC_LINE:
      cell = svtkCubicLine::New();
      break;
    case SVTK_CONVEX_POINT_SET:
      cell = svtkConvexPointSet::New();
      break;
    case SVTK_POLYHEDRON:
      cell = svtkPolyhedron::New();
      break;
    case SVTK_LAGRANGE_TRIANGLE:
      cell = svtkLagrangeTriangle::New();
      break;
    case SVTK_LAGRANGE_TETRAHEDRON:
      cell = svtkLagrangeTetra::New();
      break;
    case SVTK_LAGRANGE_CURVE:
      cell = svtkLagrangeCurve::New();
      break;
    case SVTK_LAGRANGE_QUADRILATERAL:
      cell = svtkLagrangeQuadrilateral::New();
      break;
    case SVTK_LAGRANGE_HEXAHEDRON:
      cell = svtkLagrangeHexahedron::New();
      break;
    case SVTK_LAGRANGE_WEDGE:
      cell = svtkLagrangeWedge::New();
      break;
    case SVTK_BEZIER_TRIANGLE:
      cell = svtkBezierTriangle::New();
      break;
    case SVTK_BEZIER_TETRAHEDRON:
      cell = svtkBezierTetra::New();
      break;
    case SVTK_BEZIER_CURVE:
      cell = svtkBezierCurve::New();
      break;
    case SVTK_BEZIER_QUADRILATERAL:
      cell = svtkBezierQuadrilateral::New();
      break;
    case SVTK_BEZIER_HEXAHEDRON:
      cell = svtkBezierHexahedron::New();
      break;
    case SVTK_BEZIER_WEDGE:
      cell = svtkBezierWedge::New();
      break;
  }
  return cell;
}

//----------------------------------------------------------------------------
// Set the type of dereferenced cell. Checks to see whether cell type
// has changed and creates a new cell only if necessary.
void svtkGenericCell::SetCellType(int cellType)
{
  if (this->Cell->GetCellType() != cellType)
  {
    if (cellType < 0 || cellType >= SVTK_NUMBER_OF_CELL_TYPES)
    {
      this->Cell = nullptr;
    }
    else if (this->CellStore[cellType] == nullptr)
    {
      this->CellStore[cellType] = svtkGenericCell::InstantiateCell(cellType);
      this->Cell = this->CellStore[cellType];
    }
    else
    {
      this->Cell = this->CellStore[cellType];
    }
    if (this->Cell == nullptr)
    {
      svtkErrorMacro(<< "Unsupported cell type: " << cellType << " Setting to svtkEmptyCell");
      this->Cell = this->CellStore[SVTK_EMPTY_CELL];
    }

    this->Points->UnRegister(this);
    this->Points = this->Cell->Points;
    this->Points->Register(this);
    this->PointIds->UnRegister(this);
    this->PointIds = this->Cell->PointIds;
    this->PointIds->Register(this);
  }
}

//----------------------------------------------------------------------------
void svtkGenericCell::InterpolateFunctions(const double pcoords[3], double* weights)
{
  this->Cell->InterpolateFunctions(pcoords, weights);
}

//----------------------------------------------------------------------------
void svtkGenericCell::InterpolateDerivs(const double pcoords[3], double* derivs)
{
  this->Cell->InterpolateDerivs(pcoords, derivs);
}

//----------------------------------------------------------------------------
void svtkGenericCell::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Cell:\n";
  this->Cell->PrintSelf(os, indent.GetNextIndent());
}

//----------------------------------------------------------------------------
void svtkGenericCell::SetPoints(svtkPoints* points)
{
  if (points != this->Points)
  {
    this->Points->Delete();
    this->Points = points;
    this->Points->Register(this);
    this->Cell->Points->Delete();
    this->Cell->Points = points;
    this->Cell->Points->Register(this);
  }
}

//----------------------------------------------------------------------------
void svtkGenericCell::SetPointIds(svtkIdList* pointIds)
{
  if (pointIds != this->PointIds)
  {
    this->PointIds->Delete();
    this->PointIds = pointIds;
    this->PointIds->Register(this);
    this->Cell->PointIds->Delete();
    this->Cell->PointIds = pointIds;
    this->Cell->PointIds->Register(this);
  }
}
