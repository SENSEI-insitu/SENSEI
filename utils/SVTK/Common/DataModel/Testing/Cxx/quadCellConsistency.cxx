/*=========================================================================

  Program:   Visualization Toolkit
  Module:    quadCellConsistency.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// .NAME
// .SECTION Description
// this program tests the consistency of face/edge ids between linear and quadratic cells

#include "svtkBiQuadraticQuad.h"
#include "svtkBiQuadraticQuadraticHexahedron.h"
#include "svtkBiQuadraticQuadraticWedge.h"
#include "svtkBiQuadraticTriangle.h"
#include "svtkCubicLine.h"
#include "svtkHexahedron.h"
#include "svtkLine.h"
#include "svtkPyramid.h"
#include "svtkQuad.h"
#include "svtkQuadraticEdge.h"
#include "svtkQuadraticHexahedron.h"
#include "svtkQuadraticLinearQuad.h"
#include "svtkQuadraticLinearWedge.h"
#include "svtkQuadraticPyramid.h"
#include "svtkQuadraticQuad.h"
#include "svtkQuadraticTetra.h"
#include "svtkQuadraticTriangle.h"
#include "svtkQuadraticWedge.h"
#include "svtkTetra.h"
#include "svtkTriQuadraticHexahedron.h"
#include "svtkTriangle.h"
#include "svtkWedge.h"

void InitializeCell(svtkCell* cell)
{
  // Default initialize the cell ids to 0,1,2 ... n
  int n = cell->GetNumberOfPoints();
  for (int i = 0; i < n; i++)
  {
    cell->GetPointIds()->SetId(i, i);
  }
}

// Check that corner points id match quad ones for each edges
int CompareCellEdges(svtkCell* linear, svtkCell* quadratic)
{
  int dif;
  int sum = 0;
  int nEdges = linear->GetNumberOfEdges();
  for (int edge = 0; edge < nEdges; edge++)
  {
    svtkCell* lEdge = linear->GetEdge(edge);
    svtkCell* qEdge = quadratic->GetEdge(edge);

    int n = lEdge->GetNumberOfPoints();
    // Check that the points of the linear cell match the one from the quadratic one
    for (int i = 0; i < n; i++)
    {
      dif = lEdge->GetPointIds()->GetId(i) - qEdge->GetPointIds()->GetId(i);
      sum += dif;
    }
  }
  return sum;
}

// Check that corner points id match quad ones for each faces
int CompareCellFaces(svtkCell* linear, svtkCell* quadratic)
{
  int dif;
  int sum = 0;
  int nFaces = linear->GetNumberOfFaces();
  for (int face = 0; face < nFaces; face++)
  {
    svtkCell* lFace = linear->GetFace(face);
    svtkCell* qFace = quadratic->GetFace(face);

    int n = lFace->GetNumberOfPoints();
    // Check that linear Triangle match quad Tri
    if (lFace->GetCellType() == SVTK_TRIANGLE)
      sum += (qFace->GetCellType() != SVTK_QUADRATIC_TRIANGLE &&
        qFace->GetCellType() != SVTK_BIQUADRATIC_TRIANGLE);
    // Check that linear Quad match quad Quad
    if (lFace->GetCellType() == SVTK_QUAD &&
      (qFace->GetCellType() != SVTK_QUADRATIC_QUAD && qFace->GetCellType() != SVTK_BIQUADRATIC_QUAD &&
        qFace->GetCellType() != SVTK_QUADRATIC_LINEAR_QUAD))
      sum++;
    // Check that the points of the linear cell match the one from the quadratic one
    for (int i = 0; i < n; i++)
    {
      dif = lFace->GetPointIds()->GetId(i) - qFace->GetPointIds()->GetId(i);
      sum += dif;
    }
  }
  return sum;
}

int quadCellConsistency(int, char*[])
{
  int ret = 0;
  // Line / svtkQuadraticEdge / CubicLine:
  svtkLine* edge = svtkLine::New();
  svtkQuadraticEdge* qedge = svtkQuadraticEdge::New();
  svtkCubicLine* culine = svtkCubicLine::New();

  InitializeCell(edge);
  InitializeCell(qedge);
  ret += CompareCellEdges(edge, qedge);
  ret += CompareCellFaces(edge, qedge);

  qedge->Delete();

  InitializeCell(culine);
  ret += CompareCellEdges(edge, culine);
  ret += CompareCellFaces(edge, culine);

  edge->Delete();
  culine->Delete();

  // Triangles:
  svtkTriangle* tri = svtkTriangle::New();
  svtkQuadraticTriangle* qtri = svtkQuadraticTriangle::New();
  svtkBiQuadraticTriangle* bitri = svtkBiQuadraticTriangle::New();

  InitializeCell(tri);
  InitializeCell(qtri);
  ret += CompareCellEdges(tri, qtri);
  ret += CompareCellFaces(tri, qtri);

  qtri->Delete();

  InitializeCell(bitri);
  ret += CompareCellEdges(tri, bitri);
  ret += CompareCellFaces(tri, bitri);

  tri->Delete();
  bitri->Delete();

  // Quad
  svtkQuad* quad = svtkQuad::New();
  svtkQuadraticQuad* qquad = svtkQuadraticQuad::New();
  svtkBiQuadraticQuad* biqquad = svtkBiQuadraticQuad::New();
  svtkQuadraticLinearQuad* qlquad = svtkQuadraticLinearQuad::New();

  InitializeCell(quad);
  InitializeCell(qquad);
  InitializeCell(biqquad);
  InitializeCell(qlquad);
  ret += CompareCellEdges(quad, qquad);
  ret += CompareCellFaces(quad, qquad);
  ret += CompareCellEdges(quad, biqquad);
  ret += CompareCellFaces(quad, biqquad);
  ret += CompareCellEdges(quad, qlquad);
  ret += CompareCellFaces(quad, qlquad);

  quad->Delete();
  qquad->Delete();
  biqquad->Delete();
  qlquad->Delete();

  // Tetra
  svtkTetra* tetra = svtkTetra::New();
  svtkQuadraticTetra* qtetra = svtkQuadraticTetra::New();

  InitializeCell(tetra);
  InitializeCell(qtetra);
  ret += CompareCellEdges(tetra, qtetra);
  ret += CompareCellFaces(tetra, qtetra);

  tetra->Delete();
  qtetra->Delete();

  // Hexhedron
  svtkHexahedron* hex = svtkHexahedron::New();
  svtkQuadraticHexahedron* qhex = svtkQuadraticHexahedron::New();
  svtkTriQuadraticHexahedron* triqhex = svtkTriQuadraticHexahedron::New();
  svtkBiQuadraticQuadraticHexahedron* biqqhex = svtkBiQuadraticQuadraticHexahedron::New();

  InitializeCell(hex);
  InitializeCell(qhex);
  InitializeCell(triqhex);
  InitializeCell(biqqhex);
  ret += CompareCellEdges(hex, qhex);
  ret += CompareCellFaces(hex, qhex);
  ret += CompareCellEdges(hex, triqhex);
  ret += CompareCellFaces(hex, triqhex);
  ret += CompareCellEdges(hex, biqqhex);
  ret += CompareCellFaces(hex, biqqhex);

  hex->Delete();
  qhex->Delete();
  triqhex->Delete();
  biqqhex->Delete();

  // Pyramid
  svtkPyramid* pyr = svtkPyramid::New();
  svtkQuadraticPyramid* qpyr = svtkQuadraticPyramid::New();

  InitializeCell(pyr);
  InitializeCell(qpyr);
  ret += CompareCellEdges(pyr, qpyr);
  ret += CompareCellFaces(pyr, qpyr);

  pyr->Delete();
  qpyr->Delete();

  // Wedge cells
  svtkWedge* wedge = svtkWedge::New();
  svtkQuadraticWedge* qwedge = svtkQuadraticWedge::New();
  svtkBiQuadraticQuadraticWedge* biqwedge = svtkBiQuadraticQuadraticWedge::New();

  InitializeCell(wedge);
  InitializeCell(qwedge);
  InitializeCell(biqwedge);
  ret += CompareCellEdges(wedge, qwedge);
  ret += CompareCellFaces(wedge, qwedge);
  ret += CompareCellEdges(wedge, biqwedge);
  ret += CompareCellFaces(wedge, biqwedge);

  wedge->Delete();
  qwedge->Delete();
  biqwedge->Delete();

  return ret;
}
