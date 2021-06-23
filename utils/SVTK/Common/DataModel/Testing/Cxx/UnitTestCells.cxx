/*=========================================================================

  Program:   Visualization Toolkit
  Module:    UnitTestCells.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkCellType.h"
#include "svtkSmartPointer.h"

#include "svtkEmptyCell.h"
#include "svtkHexagonalPrism.h"
#include "svtkHexahedron.h"
#include "svtkLine.h"
#include "svtkPentagonalPrism.h"
#include "svtkPixel.h"
#include "svtkPolyLine.h"
#include "svtkPolyVertex.h"
#include "svtkPolygon.h"
#include "svtkPolyhedron.h"
#include "svtkPyramid.h"
#include "svtkQuad.h"
#include "svtkTetra.h"
#include "svtkTriangle.h"
#include "svtkTriangleStrip.h"
#include "svtkVertex.h"
#include "svtkVoxel.h"
#include "svtkWedge.h"

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

#include "svtkBiQuadraticQuad.h"
#include "svtkBiQuadraticQuadraticHexahedron.h"
#include "svtkBiQuadraticQuadraticWedge.h"
#include "svtkBiQuadraticTriangle.h"
#include "svtkTriQuadraticHexahedron.h"

#include "svtkCubicLine.h"

#include "svtkCellArray.h"
#include "svtkMath.h"
#include "svtkMathUtilities.h"
#include "svtkPoints.h"
#include <map>
#include <sstream>
#include <string>
#include <vector>

static svtkSmartPointer<svtkEmptyCell> MakeEmptyCell();
static svtkSmartPointer<svtkVertex> MakeVertex();
static svtkSmartPointer<svtkPolyVertex> MakePolyVertex();
static svtkSmartPointer<svtkLine> MakeLine();
static svtkSmartPointer<svtkPolyLine> MakePolyLine();
static svtkSmartPointer<svtkTriangle> MakeTriangle();
static svtkSmartPointer<svtkTriangleStrip> MakeTriangleStrip();
static svtkSmartPointer<svtkPolygon> MakePolygon();
static svtkSmartPointer<svtkQuad> MakeQuad();
static svtkSmartPointer<svtkPixel> MakePixel();
static svtkSmartPointer<svtkVoxel> MakeVoxel();
static svtkSmartPointer<svtkHexahedron> MakeHexahedron();
static svtkSmartPointer<svtkPyramid> MakePyramid();
static svtkSmartPointer<svtkTetra> MakeTetra();
static svtkSmartPointer<svtkWedge> MakeWedge();
static svtkSmartPointer<svtkPentagonalPrism> MakePentagonalPrism();
static svtkSmartPointer<svtkHexagonalPrism> MakeHexagonalPrism();
static svtkSmartPointer<svtkPolyhedron> MakeCube();
static svtkSmartPointer<svtkPolyhedron> MakeDodecahedron();

static svtkSmartPointer<svtkQuadraticEdge> MakeQuadraticEdge();
static svtkSmartPointer<svtkQuadraticHexahedron> MakeQuadraticHexahedron();
static svtkSmartPointer<svtkQuadraticPolygon> MakeQuadraticPolygon();
static svtkSmartPointer<svtkQuadraticLinearQuad> MakeQuadraticLinearQuad();
static svtkSmartPointer<svtkQuadraticLinearWedge> MakeQuadraticLinearWedge();
static svtkSmartPointer<svtkQuadraticPyramid> MakeQuadraticPyramid();
static svtkSmartPointer<svtkQuadraticQuad> MakeQuadraticQuad();
static svtkSmartPointer<svtkQuadraticTetra> MakeQuadraticTetra();
static svtkSmartPointer<svtkQuadraticTriangle> MakeQuadraticTriangle();
static svtkSmartPointer<svtkQuadraticWedge> MakeQuadraticWedge();

static svtkSmartPointer<svtkBiQuadraticQuad> MakeBiQuadraticQuad();
static svtkSmartPointer<svtkBiQuadraticQuadraticHexahedron> MakeBiQuadraticQuadraticHexahedron();
static svtkSmartPointer<svtkBiQuadraticQuadraticWedge> MakeBiQuadraticQuadraticWedge();
static svtkSmartPointer<svtkBiQuadraticTriangle> MakeBiQuadraticTriangle();
static svtkSmartPointer<svtkTriQuadraticHexahedron> MakeTriQuadraticHexahedron();
static svtkSmartPointer<svtkCubicLine> MakeCubicLine();

template <typename T>
int TestOneCell(const SVTKCellType cellType, svtkSmartPointer<T> cell, int linear = 1);
//----------------------------------------------------------------------------
int UnitTestCells(int, char*[])
{
  std::map<std::string, int> results;

  results["EmptyCell"] = TestOneCell<svtkEmptyCell>(SVTK_EMPTY_CELL, MakeEmptyCell());
  results["Vertex"] = TestOneCell<svtkVertex>(SVTK_VERTEX, MakeVertex());
  results["PolyVertex"] = TestOneCell<svtkPolyVertex>(SVTK_POLY_VERTEX, MakePolyVertex());
  results["Line"] = TestOneCell<svtkLine>(SVTK_LINE, MakeLine());
  results["PolyLine"] = TestOneCell<svtkPolyLine>(SVTK_POLY_LINE, MakePolyLine());
  results["Triangle"] = TestOneCell<svtkTriangle>(SVTK_TRIANGLE, MakeTriangle());
  results["TriangleStrip"] = TestOneCell<svtkTriangleStrip>(SVTK_TRIANGLE_STRIP, MakeTriangleStrip());
  results["Polygon"] = TestOneCell<svtkPolygon>(SVTK_POLYGON, MakePolygon());
  results["Pixel"] = TestOneCell<svtkPixel>(SVTK_PIXEL, MakePixel());
  results["Quad"] = TestOneCell<svtkQuad>(SVTK_QUAD, MakeQuad());
  results["Tetra"] = TestOneCell<svtkTetra>(SVTK_TETRA, MakeTetra());
  results["Voxel"] = TestOneCell<svtkVoxel>(SVTK_VOXEL, MakeVoxel());
  results["Hexahedron"] = TestOneCell<svtkHexahedron>(SVTK_HEXAHEDRON, MakeHexahedron());
  results["Wedge"] = TestOneCell<svtkWedge>(SVTK_WEDGE, MakeWedge());
  results["Pyramid"] = TestOneCell<svtkPyramid>(SVTK_PYRAMID, MakePyramid());
  results["PentagonalPrism"] =
    TestOneCell<svtkPentagonalPrism>(SVTK_PENTAGONAL_PRISM, MakePentagonalPrism());
  results["HexagonalPrism"] =
    TestOneCell<svtkHexagonalPrism>(SVTK_HEXAGONAL_PRISM, MakeHexagonalPrism());
  results["Polyhedron(Cube)"] = TestOneCell<svtkPolyhedron>(SVTK_POLYHEDRON, MakeCube());
  results["Polyhedron(Dodecahedron)"] =
    TestOneCell<svtkPolyhedron>(SVTK_POLYHEDRON, MakeDodecahedron());

  results["QuadraticEdge"] =
    TestOneCell<svtkQuadraticEdge>(SVTK_QUADRATIC_EDGE, MakeQuadraticEdge(), 0);
  results["QuadraticHexahedron"] =
    TestOneCell<svtkQuadraticHexahedron>(SVTK_QUADRATIC_HEXAHEDRON, MakeQuadraticHexahedron(), 0);
  results["QuadraticPolygon"] =
    TestOneCell<svtkQuadraticPolygon>(SVTK_QUADRATIC_POLYGON, MakeQuadraticPolygon(), 0);
  results["QuadraticLinearQuad"] =
    TestOneCell<svtkQuadraticLinearQuad>(SVTK_QUADRATIC_LINEAR_QUAD, MakeQuadraticLinearQuad(), 0);
  results["QuadraticLinearWedge"] =
    TestOneCell<svtkQuadraticLinearWedge>(SVTK_QUADRATIC_LINEAR_WEDGE, MakeQuadraticLinearWedge(), 0);
  results["QuadraticPyramid"] =
    TestOneCell<svtkQuadraticPyramid>(SVTK_QUADRATIC_PYRAMID, MakeQuadraticPyramid(), 0);
  results["QuadraticQuad"] =
    TestOneCell<svtkQuadraticQuad>(SVTK_QUADRATIC_QUAD, MakeQuadraticQuad(), 0);
  results["QuadraticTetra"] =
    TestOneCell<svtkQuadraticTetra>(SVTK_QUADRATIC_TETRA, MakeQuadraticTetra(), 0);
  results["QuadraticTrangle"] =
    TestOneCell<svtkQuadraticTriangle>(SVTK_QUADRATIC_TRIANGLE, MakeQuadraticTriangle(), 0);
  results["QuadraticWedge"] =
    TestOneCell<svtkQuadraticWedge>(SVTK_QUADRATIC_WEDGE, MakeQuadraticWedge(), 0);

  results["BiQuadraticQuad"] =
    TestOneCell<svtkBiQuadraticQuad>(SVTK_BIQUADRATIC_QUAD, MakeBiQuadraticQuad(), 0);
  results["BiQuadraticQuadraticHexahedron"] = TestOneCell<svtkBiQuadraticQuadraticHexahedron>(
    SVTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON, MakeBiQuadraticQuadraticHexahedron(), 0);
  results["BiQuadraticQuadraticWedge"] = TestOneCell<svtkBiQuadraticQuadraticWedge>(
    SVTK_BIQUADRATIC_QUADRATIC_WEDGE, MakeBiQuadraticQuadraticWedge(), 0);
  results["BiQuadraticTrangle"] =
    TestOneCell<svtkBiQuadraticTriangle>(SVTK_BIQUADRATIC_TRIANGLE, MakeBiQuadraticTriangle(), 0);
  results["CubicLine"] = TestOneCell<svtkCubicLine>(SVTK_CUBIC_LINE, MakeCubicLine(), 0);

  results["TriQuadraticHexahedron"] = TestOneCell<svtkTriQuadraticHexahedron>(
    SVTK_TRIQUADRATIC_HEXAHEDRON, MakeTriQuadraticHexahedron(), 0);

  int status = 0;
  std::cout << "----- Unit Test Summary -----" << std::endl;
  std::map<std::string, int>::iterator it;
  for (it = results.begin(); it != results.end(); ++it)
  {
    std::cout << std::setw(25) << it->first << " " << (it->second ? " FAILED" : " OK") << std::endl;
    if (it->second != 0)
    {
      ++status;
    }
  }
  if (status)
  {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

svtkSmartPointer<svtkEmptyCell> MakeEmptyCell()
{
  svtkSmartPointer<svtkEmptyCell> anEmptyCell = svtkSmartPointer<svtkEmptyCell>::New();
  return anEmptyCell;
}

svtkSmartPointer<svtkVertex> MakeVertex()
{
  svtkSmartPointer<svtkVertex> aVertex = svtkSmartPointer<svtkVertex>::New();
  aVertex->GetPointIds()->SetId(0, 0);
  aVertex->GetPoints()->SetPoint(0, 10.0, 20.0, 30.0);

  return aVertex;
}

svtkSmartPointer<svtkPolyVertex> MakePolyVertex()
{
  svtkSmartPointer<svtkPolyVertex> aPolyVertex = svtkSmartPointer<svtkPolyVertex>::New();
  aPolyVertex->GetPointIds()->SetNumberOfIds(2);
  aPolyVertex->GetPointIds()->SetId(0, 0);
  aPolyVertex->GetPointIds()->SetId(1, 1);

  aPolyVertex->GetPoints()->SetNumberOfPoints(2);
  aPolyVertex->GetPoints()->SetPoint(0, 10.0, 20.0, 30.0);
  aPolyVertex->GetPoints()->SetPoint(1, 30.0, 20.0, 10.0);

  return aPolyVertex;
}

svtkSmartPointer<svtkLine> MakeLine()
{
  svtkSmartPointer<svtkLine> aLine = svtkSmartPointer<svtkLine>::New();
  aLine->GetPointIds()->SetId(0, 0);
  aLine->GetPointIds()->SetId(1, 1);
  aLine->GetPoints()->SetPoint(0, 10.0, 20.0, 30.0);
  aLine->GetPoints()->SetPoint(1, 30.0, 20.0, 10.0);
  return aLine;
}

svtkSmartPointer<svtkPolyLine> MakePolyLine()
{
  svtkSmartPointer<svtkPolyLine> aPolyLine = svtkSmartPointer<svtkPolyLine>::New();
  aPolyLine->GetPointIds()->SetNumberOfIds(3);
  aPolyLine->GetPointIds()->SetId(0, 0);
  aPolyLine->GetPointIds()->SetId(1, 1);
  aPolyLine->GetPointIds()->SetId(2, 2);

  aPolyLine->GetPoints()->SetNumberOfPoints(3);
  aPolyLine->GetPoints()->SetPoint(0, 10.0, 20.0, 30.0);
  aPolyLine->GetPoints()->SetPoint(1, 10.0, 30.0, 30.0);
  aPolyLine->GetPoints()->SetPoint(2, 10.0, 30.0, 40.0);

  return aPolyLine;
}

svtkSmartPointer<svtkTriangle> MakeTriangle()
{
  svtkSmartPointer<svtkTriangle> aTriangle = svtkSmartPointer<svtkTriangle>::New();
  aTriangle->GetPoints()->SetPoint(0, -10.0, -10.0, 0.0);
  aTriangle->GetPoints()->SetPoint(1, 10.0, -10.0, 0.0);
  aTriangle->GetPoints()->SetPoint(2, 10.0, 10.0, 0.0);
  aTriangle->GetPointIds()->SetId(0, 0);
  aTriangle->GetPointIds()->SetId(1, 1);
  aTriangle->GetPointIds()->SetId(2, 2);
  return aTriangle;
}

svtkSmartPointer<svtkTriangleStrip> MakeTriangleStrip()
{
  svtkSmartPointer<svtkTriangleStrip> aTriangleStrip = svtkSmartPointer<svtkTriangleStrip>::New();
  aTriangleStrip->GetPointIds()->SetNumberOfIds(4);
  aTriangleStrip->GetPointIds()->SetId(0, 0);
  aTriangleStrip->GetPointIds()->SetId(1, 1);
  aTriangleStrip->GetPointIds()->SetId(2, 2);
  aTriangleStrip->GetPointIds()->SetId(3, 3);

  aTriangleStrip->GetPoints()->SetNumberOfPoints(4);
  aTriangleStrip->GetPoints()->SetPoint(0, 10.0, 10.0, 10.0);
  aTriangleStrip->GetPoints()->SetPoint(1, 12.0, 10.0, 10.0);
  aTriangleStrip->GetPoints()->SetPoint(2, 11.0, 12.0, 10.0);
  aTriangleStrip->GetPoints()->SetPoint(3, 13.0, 10.0, 10.0);

  return aTriangleStrip;
}

svtkSmartPointer<svtkPolygon> MakePolygon()
{
  svtkSmartPointer<svtkPolygon> aPolygon = svtkSmartPointer<svtkPolygon>::New();
  aPolygon->GetPointIds()->SetNumberOfIds(4);
  aPolygon->GetPointIds()->SetId(0, 0);
  aPolygon->GetPointIds()->SetId(1, 1);
  aPolygon->GetPointIds()->SetId(2, 2);
  aPolygon->GetPointIds()->SetId(3, 3);

  aPolygon->GetPoints()->SetNumberOfPoints(4);
  aPolygon->GetPoints()->SetPoint(0, 0.0, 0.0, 0.0);
  aPolygon->GetPoints()->SetPoint(1, 10.0, 0.0, 0.0);
  aPolygon->GetPoints()->SetPoint(2, 10.0, 10.0, 0.0);
  aPolygon->GetPoints()->SetPoint(3, 0.0, 10.0, 0.0);

  return aPolygon;
}

svtkSmartPointer<svtkQuad> MakeQuad()
{
  svtkSmartPointer<svtkQuad> aQuad = svtkSmartPointer<svtkQuad>::New();
  aQuad->GetPoints()->SetPoint(0, -10.0, -10.0, 0.0);
  aQuad->GetPoints()->SetPoint(1, 10.0, -10.0, 0.0);
  aQuad->GetPoints()->SetPoint(2, 10.0, 10.0, 0.0);
  aQuad->GetPoints()->SetPoint(3, -10.0, 10.0, 0.0);
  aQuad->GetPointIds()->SetId(0, 0);
  aQuad->GetPointIds()->SetId(1, 1);
  aQuad->GetPointIds()->SetId(2, 2);
  aQuad->GetPointIds()->SetId(2, 3);
  return aQuad;
}

svtkSmartPointer<svtkPixel> MakePixel()
{
  svtkSmartPointer<svtkPixel> aPixel = svtkSmartPointer<svtkPixel>::New();
  aPixel->GetPointIds()->SetId(0, 0);
  aPixel->GetPointIds()->SetId(1, 1);
  aPixel->GetPointIds()->SetId(2, 3);
  aPixel->GetPointIds()->SetId(3, 2);

  aPixel->GetPoints()->SetPoint(0, 10.0, 10.0, 10.0);
  aPixel->GetPoints()->SetPoint(1, 12.0, 10.0, 10.0);
  aPixel->GetPoints()->SetPoint(3, 12.0, 12.0, 10.0);
  aPixel->GetPoints()->SetPoint(2, 10.0, 12.0, 10.0);
  return aPixel;
}

svtkSmartPointer<svtkVoxel> MakeVoxel()
{
  svtkSmartPointer<svtkVoxel> aVoxel = svtkSmartPointer<svtkVoxel>::New();
  aVoxel->GetPointIds()->SetId(0, 0);
  aVoxel->GetPointIds()->SetId(1, 1);
  aVoxel->GetPointIds()->SetId(2, 3);
  aVoxel->GetPointIds()->SetId(3, 2);
  aVoxel->GetPointIds()->SetId(4, 4);
  aVoxel->GetPointIds()->SetId(5, 5);
  aVoxel->GetPointIds()->SetId(6, 7);
  aVoxel->GetPointIds()->SetId(7, 6);

  aVoxel->GetPoints()->SetPoint(0, 10, 10, 10);
  aVoxel->GetPoints()->SetPoint(1, 12, 10, 10);
  aVoxel->GetPoints()->SetPoint(3, 12, 12, 10);
  aVoxel->GetPoints()->SetPoint(2, 10, 12, 10);
  aVoxel->GetPoints()->SetPoint(4, 10, 10, 12);
  aVoxel->GetPoints()->SetPoint(5, 12, 10, 12);
  aVoxel->GetPoints()->SetPoint(7, 12, 12, 12);
  aVoxel->GetPoints()->SetPoint(6, 10, 12, 12);
  return aVoxel;
}

svtkSmartPointer<svtkHexahedron> MakeHexahedron()
{
  svtkSmartPointer<svtkHexahedron> aHexahedron = svtkSmartPointer<svtkHexahedron>::New();
  aHexahedron->GetPointIds()->SetId(0, 0);
  aHexahedron->GetPointIds()->SetId(1, 1);
  aHexahedron->GetPointIds()->SetId(2, 2);
  aHexahedron->GetPointIds()->SetId(3, 3);
  aHexahedron->GetPointIds()->SetId(4, 4);
  aHexahedron->GetPointIds()->SetId(5, 5);
  aHexahedron->GetPointIds()->SetId(6, 6);
  aHexahedron->GetPointIds()->SetId(7, 7);

  aHexahedron->GetPoints()->SetPoint(0, 10, 10, 10);
  aHexahedron->GetPoints()->SetPoint(1, 12, 10, 10);
  aHexahedron->GetPoints()->SetPoint(2, 12, 12, 10);
  aHexahedron->GetPoints()->SetPoint(3, 10, 12, 10);
  aHexahedron->GetPoints()->SetPoint(4, 10, 10, 12);
  aHexahedron->GetPoints()->SetPoint(5, 12, 10, 12);
  aHexahedron->GetPoints()->SetPoint(6, 12, 12, 12);
  aHexahedron->GetPoints()->SetPoint(7, 10, 12, 12);

  return aHexahedron;
}

svtkSmartPointer<svtkPyramid> MakePyramid()
{
  svtkSmartPointer<svtkPyramid> aPyramid = svtkSmartPointer<svtkPyramid>::New();
  aPyramid->GetPointIds()->SetId(0, 0);
  aPyramid->GetPointIds()->SetId(1, 1);
  aPyramid->GetPointIds()->SetId(2, 2);
  aPyramid->GetPointIds()->SetId(3, 3);
  aPyramid->GetPointIds()->SetId(4, 4);

  aPyramid->GetPoints()->SetPoint(0, 0, 0, 0);
  aPyramid->GetPoints()->SetPoint(1, 1, 0, 0);
  aPyramid->GetPoints()->SetPoint(2, 1, 1, 0);
  aPyramid->GetPoints()->SetPoint(3, 0, 1, 0);
  aPyramid->GetPoints()->SetPoint(4, .5, .5, 1);

  return aPyramid;
}

svtkSmartPointer<svtkQuadraticPyramid> MakeQuadraticPyramid()
{
  svtkSmartPointer<svtkQuadraticPyramid> aPyramid = svtkSmartPointer<svtkQuadraticPyramid>::New();
  for (int i = 0; i < 13; ++i)
  {
    aPyramid->GetPointIds()->SetId(i, i);
  }

  aPyramid->GetPoints()->SetPoint(0, 0, 0, 0);
  aPyramid->GetPoints()->SetPoint(1, 1, 0, 0);
  aPyramid->GetPoints()->SetPoint(2, 1, 1, 0);
  aPyramid->GetPoints()->SetPoint(3, 0, 1, 0);
  aPyramid->GetPoints()->SetPoint(4, .5, .5, 1);

  aPyramid->GetPoints()->SetPoint(5, 0.5, 0.0, 0.0);
  aPyramid->GetPoints()->SetPoint(6, 1.0, 0.5, 0.0);
  aPyramid->GetPoints()->SetPoint(7, 0.5, 1.0, 0.0);
  aPyramid->GetPoints()->SetPoint(8, 0.0, 0.5, 0.0);

  aPyramid->GetPoints()->SetPoint(9, 0.5, 0.5, 0.5);
  aPyramid->GetPoints()->SetPoint(10, 0.75, 0.5, 0.5);
  aPyramid->GetPoints()->SetPoint(11, 0.75, 0.75, 0.5);
  aPyramid->GetPoints()->SetPoint(12, 0.5, 0.75, 0.5);

  return aPyramid;
}

svtkSmartPointer<svtkQuadraticEdge> MakeQuadraticEdge()
{
  svtkSmartPointer<svtkQuadraticEdge> anEdge = svtkSmartPointer<svtkQuadraticEdge>::New();
  for (int i = 0; i < 3; ++i)
  {
    anEdge->GetPointIds()->SetId(i, i);
  }

  anEdge->GetPoints()->SetPoint(0, 0, 0, 0);
  anEdge->GetPoints()->SetPoint(1, 1, 0, 0);
  anEdge->GetPoints()->SetPoint(2, .5, 0, 0);

  return anEdge;
}

svtkSmartPointer<svtkQuadraticHexahedron> MakeQuadraticHexahedron()
{
  svtkSmartPointer<svtkQuadraticHexahedron> aHexahedron =
    svtkSmartPointer<svtkQuadraticHexahedron>::New();
  double* pcoords = aHexahedron->GetParametricCoords();
  for (int i = 0; i < aHexahedron->GetNumberOfPoints(); ++i)
  {
    aHexahedron->GetPointIds()->SetId(i, i);
    aHexahedron->GetPoints()->SetPoint(i, *(pcoords + 3 * i) + svtkMath::Random(-.1, .1),
      *(pcoords + 3 * i + 1) + svtkMath::Random(-.1, .1),
      *(pcoords + 3 * i + 2) + svtkMath::Random(-.1, .1));
  }
  return aHexahedron;
}

svtkSmartPointer<svtkBiQuadraticQuadraticHexahedron> MakeBiQuadraticQuadraticHexahedron()
{
  svtkSmartPointer<svtkBiQuadraticQuadraticHexahedron> aHexahedron =
    svtkSmartPointer<svtkBiQuadraticQuadraticHexahedron>::New();
  double* pcoords = aHexahedron->GetParametricCoords();
  for (int i = 0; i < aHexahedron->GetNumberOfPoints(); ++i)
  {
    aHexahedron->GetPointIds()->SetId(i, i);
    aHexahedron->GetPoints()->SetPoint(i, *(pcoords + 3 * i) + svtkMath::Random(-.1, .1),
      *(pcoords + 3 * i + 1) + svtkMath::Random(-.1, .1),
      *(pcoords + 3 * i + 2) + svtkMath::Random(-.1, .1));
  }
  return aHexahedron;
}

svtkSmartPointer<svtkTriQuadraticHexahedron> MakeTriQuadraticHexahedron()
{
  svtkSmartPointer<svtkTriQuadraticHexahedron> aHexahedron =
    svtkSmartPointer<svtkTriQuadraticHexahedron>::New();
  double* pcoords = aHexahedron->GetParametricCoords();
  for (int i = 0; i < aHexahedron->GetNumberOfPoints(); ++i)
  {
    aHexahedron->GetPointIds()->SetId(i, i);
    aHexahedron->GetPoints()->SetPoint(i, *(pcoords + 3 * i) + svtkMath::Random(-.1, .1),
      *(pcoords + 3 * i + 1) + svtkMath::Random(-.1, .1),
      *(pcoords + 3 * i + 2) + svtkMath::Random(-.1, .1));
  }
  return aHexahedron;
}

svtkSmartPointer<svtkQuadraticPolygon> MakeQuadraticPolygon()
{
  svtkSmartPointer<svtkQuadraticPolygon> aPolygon = svtkSmartPointer<svtkQuadraticPolygon>::New();

  aPolygon->GetPointIds()->SetNumberOfIds(8);
  aPolygon->GetPointIds()->SetId(0, 0);
  aPolygon->GetPointIds()->SetId(1, 1);
  aPolygon->GetPointIds()->SetId(2, 2);
  aPolygon->GetPointIds()->SetId(3, 3);
  aPolygon->GetPointIds()->SetId(4, 4);
  aPolygon->GetPointIds()->SetId(5, 5);
  aPolygon->GetPointIds()->SetId(6, 6);
  aPolygon->GetPointIds()->SetId(7, 7);

  aPolygon->GetPoints()->SetNumberOfPoints(8);
  aPolygon->GetPoints()->SetPoint(0, 0.0, 0.0, 0.0);
  aPolygon->GetPoints()->SetPoint(1, 2.0, 0.0, 0.0);
  aPolygon->GetPoints()->SetPoint(2, 2.0, 2.0, 0.0);
  aPolygon->GetPoints()->SetPoint(3, 0.0, 2.0, 0.0);
  aPolygon->GetPoints()->SetPoint(4, 1.0, 0.0, 0.0);
  aPolygon->GetPoints()->SetPoint(5, 2.0, 1.0, 0.0);
  aPolygon->GetPoints()->SetPoint(6, 1.0, 2.0, 0.0);
  aPolygon->GetPoints()->SetPoint(7, 0.0, 1.0, 0.0);
  aPolygon->GetPoints()->SetPoint(5, 3.0, 1.0, 0.0);
  return aPolygon;
}

svtkSmartPointer<svtkQuadraticLinearQuad> MakeQuadraticLinearQuad()
{
  svtkSmartPointer<svtkQuadraticLinearQuad> aLinearQuad =
    svtkSmartPointer<svtkQuadraticLinearQuad>::New();
  double* pcoords = aLinearQuad->GetParametricCoords();
  for (int i = 0; i < aLinearQuad->GetNumberOfPoints(); ++i)
  {
    aLinearQuad->GetPointIds()->SetId(i, i);
    aLinearQuad->GetPoints()->SetPoint(
      i, *(pcoords + 3 * i), *(pcoords + 3 * i + 1), *(pcoords + 3 * i + 2));
  }
  return aLinearQuad;
}

svtkSmartPointer<svtkQuadraticLinearWedge> MakeQuadraticLinearWedge()
{
  svtkSmartPointer<svtkQuadraticLinearWedge> aLinearWedge =
    svtkSmartPointer<svtkQuadraticLinearWedge>::New();
  double* pcoords = aLinearWedge->GetParametricCoords();
  for (int i = 0; i < 12; ++i)
  {
    aLinearWedge->GetPointIds()->SetId(i, i);
    aLinearWedge->GetPoints()->SetPoint(
      i, *(pcoords + 3 * i), *(pcoords + 3 * i + 1), *(pcoords + 3 * i + 2));
  }
  return aLinearWedge;
}

svtkSmartPointer<svtkQuadraticQuad> MakeQuadraticQuad()
{
  svtkSmartPointer<svtkQuadraticQuad> aQuad = svtkSmartPointer<svtkQuadraticQuad>::New();
  double* pcoords = aQuad->GetParametricCoords();
  for (int i = 0; i < 8; ++i)
  {
    aQuad->GetPointIds()->SetId(i, i);
    aQuad->GetPoints()->SetPoint(i, *(pcoords + 3 * i) + svtkMath::Random(-.1, .1),
      *(pcoords + 3 * i + 1) + svtkMath::Random(-.1, .1), *(pcoords + 3 * i + 2));
  }
  return aQuad;
}

svtkSmartPointer<svtkQuadraticTetra> MakeQuadraticTetra()
{
  svtkSmartPointer<svtkQuadraticTetra> aTetra = svtkSmartPointer<svtkQuadraticTetra>::New();
  double* pcoords = aTetra->GetParametricCoords();
  for (int i = 0; i < 10; ++i)
  {
    aTetra->GetPointIds()->SetId(i, i);
    aTetra->GetPoints()->SetPoint(i, *(pcoords + 3 * i) + svtkMath::Random(-.1, .1),
      *(pcoords + 3 * i + 1) + svtkMath::Random(-.1, .1),
      *(pcoords + 3 * i + 2) + svtkMath::Random(-.1, .1));
  }
  return aTetra;
}

svtkSmartPointer<svtkQuadraticTriangle> MakeQuadraticTriangle()
{
  svtkSmartPointer<svtkQuadraticTriangle> aTriangle = svtkSmartPointer<svtkQuadraticTriangle>::New();
  double* pcoords = aTriangle->GetParametricCoords();
  for (int i = 0; i < aTriangle->GetNumberOfPoints(); ++i)
  {
    aTriangle->GetPointIds()->SetId(i, i);
    aTriangle->GetPoints()->SetPoint(
      i, *(pcoords + 3 * i), *(pcoords + 3 * i + 1), *(pcoords + 3 * i + 2));
  }
  return aTriangle;
}

svtkSmartPointer<svtkBiQuadraticTriangle> MakeBiQuadraticTriangle()
{
  svtkSmartPointer<svtkBiQuadraticTriangle> aTriangle =
    svtkSmartPointer<svtkBiQuadraticTriangle>::New();
  double* pcoords = aTriangle->GetParametricCoords();
  for (int i = 0; i < aTriangle->GetNumberOfPoints(); ++i)
  {
    aTriangle->GetPointIds()->SetId(i, i);
    aTriangle->GetPoints()->SetPoint(
      i, *(pcoords + 3 * i), *(pcoords + 3 * i + 1), *(pcoords + 3 * i + 2));
  }
  return aTriangle;
}

svtkSmartPointer<svtkBiQuadraticQuad> MakeBiQuadraticQuad()
{
  svtkSmartPointer<svtkBiQuadraticQuad> aQuad = svtkSmartPointer<svtkBiQuadraticQuad>::New();
  double* pcoords = aQuad->GetParametricCoords();
  for (int i = 0; i < aQuad->GetNumberOfPoints(); ++i)
  {
    aQuad->GetPointIds()->SetId(i, i);
    aQuad->GetPoints()->SetPoint(i, *(pcoords + 3 * i) + svtkMath::Random(-.1, .1),
      *(pcoords + 3 * i + 1) + svtkMath::Random(-.1, .1), *(pcoords + 3 * i + 2));
  }
  return aQuad;
}

svtkSmartPointer<svtkCubicLine> MakeCubicLine()
{
  svtkSmartPointer<svtkCubicLine> aLine = svtkSmartPointer<svtkCubicLine>::New();
  double* pcoords = aLine->GetParametricCoords();
  for (int i = 0; i < aLine->GetNumberOfPoints(); ++i)
  {
    aLine->GetPointIds()->SetId(i, i);
    aLine->GetPoints()->SetPoint(
      i, *(pcoords + 3 * i), *(pcoords + 3 * i + 1), *(pcoords + 3 * i + 2));
  }
  return aLine;
}

svtkSmartPointer<svtkQuadraticWedge> MakeQuadraticWedge()
{
  svtkSmartPointer<svtkQuadraticWedge> aWedge = svtkSmartPointer<svtkQuadraticWedge>::New();
  double* pcoords = aWedge->GetParametricCoords();
  for (int i = 0; i < aWedge->GetNumberOfPoints(); ++i)
  {
    aWedge->GetPointIds()->SetId(i, i);
    aWedge->GetPoints()->SetPoint(
      i, *(pcoords + 3 * i), *(pcoords + 3 * i + 1), *(pcoords + 3 * i + 2));
  }
  return aWedge;
}

svtkSmartPointer<svtkBiQuadraticQuadraticWedge> MakeBiQuadraticQuadraticWedge()
{
  svtkSmartPointer<svtkBiQuadraticQuadraticWedge> aWedge =
    svtkSmartPointer<svtkBiQuadraticQuadraticWedge>::New();
  double* pcoords = aWedge->GetParametricCoords();
  for (int i = 0; i < aWedge->GetNumberOfPoints(); ++i)
  {
    aWedge->GetPointIds()->SetId(i, i);
    aWedge->GetPoints()->SetPoint(
      i, *(pcoords + 3 * i), *(pcoords + 3 * i + 1), *(pcoords + 3 * i + 2));
  }
  return aWedge;
}

svtkSmartPointer<svtkTetra> MakeTetra()
{
  svtkSmartPointer<svtkTetra> aTetra = svtkSmartPointer<svtkTetra>::New();
  aTetra->GetPointIds()->SetId(0, 0);
  aTetra->GetPointIds()->SetId(1, 1);
  aTetra->GetPointIds()->SetId(2, 2);
  aTetra->GetPointIds()->SetId(3, 3);
  aTetra->GetPoints()->SetPoint(0, 10.0, 10.0, 10.0);
  aTetra->GetPoints()->SetPoint(1, 12.0, 10.0, 10.0);
  aTetra->GetPoints()->SetPoint(2, 11.0, 12.0, 10.0);
  aTetra->GetPoints()->SetPoint(3, 11.0, 11.0, 12.0);
  return aTetra;
}

svtkSmartPointer<svtkWedge> MakeWedge()
{
  svtkSmartPointer<svtkWedge> aWedge = svtkSmartPointer<svtkWedge>::New();
  aWedge->GetPointIds()->SetId(0, 0);
  aWedge->GetPointIds()->SetId(1, 1);
  aWedge->GetPointIds()->SetId(2, 2);
  aWedge->GetPointIds()->SetId(3, 3);
  aWedge->GetPointIds()->SetId(4, 4);
  aWedge->GetPointIds()->SetId(5, 5);

  aWedge->GetPoints()->SetPoint(0, 10, 10, 10);
  aWedge->GetPoints()->SetPoint(1, 12, 10, 10);
  aWedge->GetPoints()->SetPoint(2, 11, 12, 10);
  aWedge->GetPoints()->SetPoint(3, 10, 10, 12);
  aWedge->GetPoints()->SetPoint(4, 12, 10, 12);
  aWedge->GetPoints()->SetPoint(5, 11, 12, 12);
  return aWedge;
}

svtkSmartPointer<svtkPolyhedron> MakeCube()
{
  svtkSmartPointer<svtkPolyhedron> aCube = svtkSmartPointer<svtkPolyhedron>::New();

  // create polyhedron (cube)
  svtkSmartPointer<svtkPoints> points = svtkSmartPointer<svtkPoints>::New();

  aCube->GetPointIds()->SetNumberOfIds(8);
  aCube->GetPointIds()->SetId(0, 0);
  aCube->GetPointIds()->SetId(1, 1);
  aCube->GetPointIds()->SetId(2, 2);
  aCube->GetPointIds()->SetId(3, 3);
  aCube->GetPointIds()->SetId(4, 4);
  aCube->GetPointIds()->SetId(5, 5);
  aCube->GetPointIds()->SetId(6, 6);
  aCube->GetPointIds()->SetId(7, 7);

  aCube->GetPoints()->SetNumberOfPoints(8);
  aCube->GetPoints()->SetPoint(0, -1.0, -1.0, -1.0);
  aCube->GetPoints()->SetPoint(1, 1.0, -1.0, -1.0);
  aCube->GetPoints()->SetPoint(2, 1.0, 1.0, -1.0);
  aCube->GetPoints()->SetPoint(3, -1.0, 1.0, -1.0);
  aCube->GetPoints()->SetPoint(4, -1.0, -1.0, 1.0);
  aCube->GetPoints()->SetPoint(5, 1.0, -1.0, 1.0);
  aCube->GetPoints()->SetPoint(6, 1.0, 1.0, 1.0);
  aCube->GetPoints()->SetPoint(7, -1.0, 1.0, 1.0);

  svtkIdType faces[31] = {
    6,             // number of faces
    4, 0, 3, 2, 1, //
    4, 0, 4, 7, 3, //
    4, 4, 5, 6, 7, //
    4, 5, 1, 2, 6, //
    4, 0, 1, 5, 4, //
    4, 2, 3, 7, 6  //
  };

  aCube->SetFaces(faces);
  aCube->Initialize();
  return aCube;
}

svtkSmartPointer<svtkPolyhedron> MakeDodecahedron()
{
  svtkSmartPointer<svtkPolyhedron> aDodecahedron = svtkSmartPointer<svtkPolyhedron>::New();

  // create polyhedron (dodecahedron)
  svtkSmartPointer<svtkPoints> points = svtkSmartPointer<svtkPoints>::New();

  for (int i = 0; i < 20; ++i)
  {
    aDodecahedron->GetPointIds()->InsertNextId(i);
  }

  aDodecahedron->GetPoints()->InsertNextPoint(1.21412, 0, 1.58931);
  aDodecahedron->GetPoints()->InsertNextPoint(0.375185, 1.1547, 1.58931);
  aDodecahedron->GetPoints()->InsertNextPoint(-0.982247, 0.713644, 1.58931);
  aDodecahedron->GetPoints()->InsertNextPoint(-0.982247, -0.713644, 1.58931);
  aDodecahedron->GetPoints()->InsertNextPoint(0.375185, -1.1547, 1.58931);
  aDodecahedron->GetPoints()->InsertNextPoint(1.96449, 0, 0.375185);
  aDodecahedron->GetPoints()->InsertNextPoint(0.607062, 1.86835, 0.375185);
  aDodecahedron->GetPoints()->InsertNextPoint(-1.58931, 1.1547, 0.375185);
  aDodecahedron->GetPoints()->InsertNextPoint(-1.58931, -1.1547, 0.375185);
  aDodecahedron->GetPoints()->InsertNextPoint(0.607062, -1.86835, 0.375185);
  aDodecahedron->GetPoints()->InsertNextPoint(1.58931, 1.1547, -0.375185);
  aDodecahedron->GetPoints()->InsertNextPoint(-0.607062, 1.86835, -0.375185);
  aDodecahedron->GetPoints()->InsertNextPoint(-1.96449, 0, -0.375185);
  aDodecahedron->GetPoints()->InsertNextPoint(-0.607062, -1.86835, -0.375185);
  aDodecahedron->GetPoints()->InsertNextPoint(1.58931, -1.1547, -0.375185);
  aDodecahedron->GetPoints()->InsertNextPoint(0.982247, 0.713644, -1.58931);
  aDodecahedron->GetPoints()->InsertNextPoint(-0.375185, 1.1547, -1.58931);
  aDodecahedron->GetPoints()->InsertNextPoint(-1.21412, 0, -1.58931);
  aDodecahedron->GetPoints()->InsertNextPoint(-0.375185, -1.1547, -1.58931);
  aDodecahedron->GetPoints()->InsertNextPoint(0.982247, -0.713644, -1.58931);

  svtkIdType faces[73] = {
    12,                   // number of faces
    5, 0, 1, 2, 3, 4,     // number of ids on face, ids
    5, 0, 5, 10, 6, 1,    //
    5, 1, 6, 11, 7, 2,    //
    5, 2, 7, 12, 8, 3,    //
    5, 3, 8, 13, 9, 4,    //
    5, 4, 9, 14, 5, 0,    //
    5, 15, 10, 5, 14, 19, //
    5, 16, 11, 6, 10, 15, //
    5, 17, 12, 7, 11, 16, //
    5, 18, 13, 8, 12, 17, //
    5, 19, 14, 9, 13, 18, //
    5, 19, 18, 17, 16, 15 //
  };

  aDodecahedron->SetFaces(faces);
  aDodecahedron->Initialize();

  return aDodecahedron;
}

svtkSmartPointer<svtkPentagonalPrism> MakePentagonalPrism()
{
  svtkSmartPointer<svtkPentagonalPrism> aPentagonalPrism = svtkSmartPointer<svtkPentagonalPrism>::New();

  aPentagonalPrism->GetPointIds()->SetId(0, 0);
  aPentagonalPrism->GetPointIds()->SetId(1, 1);
  aPentagonalPrism->GetPointIds()->SetId(2, 2);
  aPentagonalPrism->GetPointIds()->SetId(3, 3);
  aPentagonalPrism->GetPointIds()->SetId(4, 4);
  aPentagonalPrism->GetPointIds()->SetId(5, 5);
  aPentagonalPrism->GetPointIds()->SetId(6, 6);
  aPentagonalPrism->GetPointIds()->SetId(7, 7);
  aPentagonalPrism->GetPointIds()->SetId(8, 8);
  aPentagonalPrism->GetPointIds()->SetId(9, 9);

  aPentagonalPrism->GetPoints()->SetPoint(0, 11, 10, 10);
  aPentagonalPrism->GetPoints()->SetPoint(1, 13, 10, 10);
  aPentagonalPrism->GetPoints()->SetPoint(2, 14, 12, 10);
  aPentagonalPrism->GetPoints()->SetPoint(3, 12, 14, 10);
  aPentagonalPrism->GetPoints()->SetPoint(4, 10, 12, 10);
  aPentagonalPrism->GetPoints()->SetPoint(5, 11, 10, 14);
  aPentagonalPrism->GetPoints()->SetPoint(6, 13, 10, 14);
  aPentagonalPrism->GetPoints()->SetPoint(7, 14, 12, 14);
  aPentagonalPrism->GetPoints()->SetPoint(8, 12, 14, 14);
  aPentagonalPrism->GetPoints()->SetPoint(9, 10, 12, 14);

  return aPentagonalPrism;
}

svtkSmartPointer<svtkHexagonalPrism> MakeHexagonalPrism()
{
  svtkSmartPointer<svtkHexagonalPrism> aHexagonalPrism = svtkSmartPointer<svtkHexagonalPrism>::New();
  aHexagonalPrism->GetPointIds()->SetId(0, 0);
  aHexagonalPrism->GetPointIds()->SetId(1, 1);
  aHexagonalPrism->GetPointIds()->SetId(2, 2);
  aHexagonalPrism->GetPointIds()->SetId(3, 3);
  aHexagonalPrism->GetPointIds()->SetId(4, 4);
  aHexagonalPrism->GetPointIds()->SetId(5, 5);
  aHexagonalPrism->GetPointIds()->SetId(6, 6);
  aHexagonalPrism->GetPointIds()->SetId(7, 7);
  aHexagonalPrism->GetPointIds()->SetId(8, 8);
  aHexagonalPrism->GetPointIds()->SetId(9, 9);
  aHexagonalPrism->GetPointIds()->SetId(10, 10);
  aHexagonalPrism->GetPointIds()->SetId(11, 11);

  aHexagonalPrism->GetPoints()->SetPoint(0, 11, 10, 10);
  aHexagonalPrism->GetPoints()->SetPoint(1, 13, 10, 10);
  aHexagonalPrism->GetPoints()->SetPoint(2, 14, 12, 10);
  aHexagonalPrism->GetPoints()->SetPoint(3, 13, 14, 10);
  aHexagonalPrism->GetPoints()->SetPoint(4, 11, 14, 10);
  aHexagonalPrism->GetPoints()->SetPoint(5, 10, 12, 10);
  aHexagonalPrism->GetPoints()->SetPoint(6, 11, 10, 14);
  aHexagonalPrism->GetPoints()->SetPoint(7, 13, 10, 14);
  aHexagonalPrism->GetPoints()->SetPoint(8, 14, 12, 14);
  aHexagonalPrism->GetPoints()->SetPoint(9, 13, 14, 14);
  aHexagonalPrism->GetPoints()->SetPoint(10, 11, 14, 14);
  aHexagonalPrism->GetPoints()->SetPoint(11, 10, 12, 14);

  return aHexagonalPrism;
}

template <typename T>
int TestOneCell(const SVTKCellType cellType, svtkSmartPointer<T> aCell, int linear)
{
  int status = 0;
  std::cout << "Testing " << aCell->GetClassName() << std::endl;

  std::cout << "  Testing Print of an uninitialized cell...";
  std::ostringstream cellPrint;
  aCell->Print(cellPrint);
  std::cout << "PASSED" << std::endl;

  std::cout << "  Testing GetCellType...";
  if (cellType != aCell->GetCellType())
  {
    std::cout << "Expected " << cellType << " but got " << aCell->GetCellType() << " FAILED"
              << std::endl;
    ++status;
  }
  else
  {
    std::cout << "PASSED" << std::endl;
  }

  std::cout << "  Testing GetCellDimension...";
  std::cout << aCell->GetCellDimension();
  std::cout << "...PASSED" << std::endl;

  std::cout << "  Testing IsLinear...";
  if (aCell->IsLinear() != 1 && linear)
  {
    ++status;
    std::cout << "...FAILED" << std::endl;
  }
  else
  {
    std::cout << "...PASSED" << std::endl;
  }

  std::cout << "  Testing IsPrimaryCell...";
  std::cout << aCell->IsPrimaryCell();
  std::cout << "...PASSED" << std::endl;

  std::cout << "  Testing IsExplicitCell...";
  std::cout << aCell->IsExplicitCell();
  std::cout << "...PASSED" << std::endl;

  std::cout << "  Testing RequiresInitialization...";
  std::cout << aCell->RequiresInitialization();
  std::cout << "...PASSED" << std::endl;

  std::cout << "  Testing RequiresExplicitFaceRepresentation...";
  std::cout << aCell->RequiresExplicitFaceRepresentation();
  std::cout << "...PASSED" << std::endl;

  if (aCell->RequiresInitialization())
  {
    aCell->Initialize();
  }
  std::cout << "  Testing GetNumberOfPoints...";
  std::cout << aCell->GetNumberOfPoints();
  std::cout << "...PASSED" << std::endl;

  std::cout << "  Testing GetNumberOfEdges...";
  std::cout << aCell->GetNumberOfEdges();
  std::cout << "...PASSED" << std::endl;

  std::cout << "  Testing GetNumberOfFaces...";
  std::cout << aCell->GetNumberOfFaces();
  std::cout << "...PASSED" << std::endl;

  if (std::string(aCell->GetClassName()) != "svtkEmptyCell" &&
    std::string(aCell->GetClassName()) != "svtkVertex" &&
    std::string(aCell->GetClassName()) != "svtkPolyhedron")
  {
    std::cout << "  Testing GetParametricCoords...";
    double* parametricCoords = aCell->GetParametricCoords();
    if (aCell->IsPrimaryCell() && parametricCoords == nullptr)
    {
      ++status;
      std::cout << "...FAILED" << std::endl;
    }
    else if (parametricCoords)
    {
      std::vector<double> pweights(aCell->GetNumberOfPoints());
      // The pcoords should correspond to the cell points
      for (int p = 0; p < aCell->GetNumberOfPoints(); ++p)
      {
        double vertex[3];
        aCell->GetPoints()->GetPoint(p, vertex);
        int subId = 0;
        double x[3];
        aCell->EvaluateLocation(subId, parametricCoords + 3 * p, x, &(*pweights.begin()));
        if (!svtkMathUtilities::FuzzyCompare(x[0], vertex[0], 1.e-3) ||
          !svtkMathUtilities::FuzzyCompare(x[1], vertex[1], 1.e-3) ||
          !svtkMathUtilities::FuzzyCompare(x[2], vertex[2], 1.e-3))
        {
          std::cout << "EvaluateLocation failed...";
          std::cout << "pcoords[" << p << "]: " << parametricCoords[3 * p] << " "
                    << parametricCoords[3 * p + 1] << " " << parametricCoords[3 * p + 2]
                    << std::endl;
          std::cout << "x[" << p << "]: " << x[0] << " " << x[1] << " " << x[2] << std::endl;
          std::cout << "...FAILED" << std::endl;
          ++status;
        }
      }
      std::cout << "...PASSED" << std::endl;
    }
  }
  std::cout << "  Testing GetBounds...";
  double bounds[6];
  aCell->GetBounds(bounds);
  std::cout << bounds[0] << "," << bounds[1] << " " << bounds[2] << "," << bounds[3] << " "
            << bounds[4] << "," << bounds[5];
  std::cout << "...PASSED" << std::endl;

  if (aCell->GetNumberOfPoints() > 0)
  {
    std::cout << "  Testing GetParametricCenter...";
    double pcenter[3], center[3];
    pcenter[0] = pcenter[1] = pcenter[2] = -12345.0;
    aCell->GetParametricCenter(pcenter);
    std::cout << pcenter[0] << ", " << pcenter[1] << ", " << pcenter[2];
    std::vector<double> cweights(aCell->GetNumberOfPoints());
    int pSubId = 0;
    aCell->EvaluateLocation(pSubId, pcenter, center, &(*cweights.begin()));
    std::cout << " -> " << center[0] << ", " << center[1] << ", " << center[2];
    if (center[0] < bounds[0] || center[0] > bounds[1] || center[1] < bounds[2] ||
      center[1] > bounds[3] || center[2] < bounds[4] || center[2] > bounds[5])
    {
      std::cout << " The computed center is not within the bounds of the cell" << std::endl;
      std::cout << "bounds: " << bounds[0] << "," << bounds[1] << " " << bounds[2] << ","
                << bounds[3] << " " << bounds[4] << "," << bounds[5] << std::endl;
      std::cout << "parametric center " << pcenter[0] << ", " << pcenter[1] << ", " << pcenter[2]
                << " "
                << "center: " << center[0] << ", " << center[1] << ", " << center[2] << std::endl;
      std::cout << "...FAILED" << std::endl;
    }
    else
    {
      std::cout << "...PASSED" << std::endl;
    }
  }

  std::cout << "  Testing GetParametricDistance...";
  double pcenter[3];
  aCell->GetParametricCenter(pcenter);
  double pd = aCell->GetParametricDistance(pcenter);
  if (pd == 0.0)
  {
    std::cout << "...PASSED" << std::endl;
  }
  else
  {
    ++status;
    std::cout << "...FAILED" << std::endl;
  }

  std::cout << "  Testing CellBoundaries...";
  svtkSmartPointer<svtkIdList> cellIds = svtkSmartPointer<svtkIdList>::New();
  int cellStatus = aCell->CellBoundary(0, pcenter, cellIds);
  if (aCell->GetCellDimension() > 0 && cellStatus != 1)
  {
    ++status;
    std::cout << "FAILED" << std::endl;
  }
  else
  {
    for (int c = 0; c < cellIds->GetNumberOfIds(); ++c)
    {
      std::cout << " " << cellIds->GetId(c) << ", ";
    }
    std::cout << "PASSED" << std::endl;
  }

  if (aCell->GetNumberOfPoints() > 0 && strcmp(aCell->GetClassName(), "svtkQuadraticEdge") != 0)
  {
    std::cout << "  Testing Derivatives...";
    // Create scalars and set first scalar to 1.0
    std::vector<double> scalars(aCell->GetNumberOfPoints());
    scalars[0] = 1.0;
    for (int s = 1; s < aCell->GetNumberOfPoints(); ++s)
    {
      scalars[s] = 0.0;
    }
    std::vector<double> derivs(3, -12345.0);
    aCell->Derivatives(0, pcenter, &(*scalars.begin()), 1, &(*derivs.begin()));
    if (derivs[0] == -12345. && derivs[1] == -12345. && derivs[2] == -12345.)
    {
      std::cout << " not computed";
    }
    else
    {
      std::cout << " " << derivs[0] << " " << derivs[1] << " " << derivs[2] << " ";
    }
    std::cout << "...PASSED" << std::endl;
  }

  std::cout << "  Testing EvaluateLocation vertex matches pcoord...";
  int status5 = 0;
  double* locations = aCell->GetParametricCoords();
  if (locations)
  {
    std::vector<double> lweights(aCell->GetNumberOfPoints());
    for (int l = 0; l < aCell->GetNumberOfPoints(); ++l)
    {
      double point[3];
      double vertex[3];
      aCell->GetPoints()->GetPoint(l, vertex);
      int subId = 0;
      aCell->EvaluateLocation(subId, locations + 3 * l, point, &(*lweights.begin()));
      for (int v = 0; v < 3; ++v)
      {
        if (!svtkMathUtilities::FuzzyCompare(point[v], vertex[v], 1.e-3))
        {
          std::cout << " " << point[0] << ", " << point[1] << ", " << point[2]
                    << " != " << vertex[0] << ", " << vertex[1] << ", " << vertex[2] << " ";
          std::cout << "eps ratio is: "
                    << (point[v] - vertex[v]) / std::numeric_limits<double>::epsilon() << std::endl;

          ++status5;
          break;
        }
      }
    }
  }
  if (status5)
  {
    std::cout << "...FAILED" << std::endl;
    ++status;
  }
  else
  {
    std::cout << "...PASSED" << std::endl;
  }

  std::cout << "  Testing EvaluatePosition pcoord matches vertex...";
  // Each vertex should corrrespond to a pcoord.
  int subId = 0;
  int status6 = 0;
  std::vector<double> weights(aCell->GetNumberOfPoints());
  double* vlocations = aCell->GetParametricCoords();
  if (vlocations)
  {
    for (int i = 0; i < aCell->GetNumberOfPoints(); ++i)
    {
      int status61 = 0;
      double closestPoint[3];
      double point[3];
      double pcoords[3];
      double dist2;
      aCell->GetPoints()->GetPoint(i, point);
      aCell->EvaluatePosition(point, closestPoint, subId, pcoords, dist2, &(*weights.begin()));
      for (int v = 0; v < 3; ++v)
      {
        if (!svtkMathUtilities::FuzzyCompare(*(vlocations + 3 * i + v), pcoords[v], 1.e-3))
        {
          ++status61;
        }
      }
      if (status61)
      {
        std::cout << std::endl
                  << *(vlocations + 3 * i + 0) << ", " << *(vlocations + 3 * i + 1) << ", "
                  << *(vlocations + 3 * i + 2) << " != " << pcoords[0] << ", " << pcoords[1] << ", "
                  << pcoords[2] << " ";
        ++status6;
      }
    }
  }
  if (status6)
  {
    ++status;
    std::cout << "...FAILED" << std::endl;
  }
  else
  {
    std::cout << "...PASSED" << std::endl;
  }

  std::cout << "  Testing EvaluatePosition in/out test...";

  int status2 = 0;
  std::vector<std::vector<double> > testPoints;
  std::vector<int> inOuts;
  std::vector<std::string> typePoint;

  // First test cell points
  for (int i = 0; i < aCell->GetNumberOfPoints(); ++i)
  {
    std::vector<double> point(3);
    aCell->GetPoints()->GetPoint(i, &(*point.begin()));
    testPoints.push_back(point);
    inOuts.push_back(1);
    typePoint.push_back("cell point");
  }
  // Then test center of cell
  if (aCell->GetNumberOfPoints() > 0)
  {
    std::vector<double> tCenter(3);
    aCell->EvaluateLocation(subId, pcenter, &(*tCenter.begin()), &(*weights.begin()));
    testPoints.push_back(tCenter);
    inOuts.push_back(1);
    typePoint.push_back("cell center");
    // Test a point above the cell
    if (aCell->GetCellDimension() == 2)
    {
      std::vector<double> above(3);
      above[0] = tCenter[0];
      above[1] = tCenter[1];
      above[2] = tCenter[2] + aCell->GetLength2();
      testPoints.push_back(above);
      inOuts.push_back(0);
      typePoint.push_back("point above cell");
    }
  }

  // Test points at the center of each edge
  for (int e = 0; e < aCell->GetNumberOfEdges(); ++e)
  {
    std::vector<double> eCenter(3);
    svtkCell* c = aCell->GetEdge(e);
    c->GetParametricCenter(pcenter);
    c->EvaluateLocation(subId, pcenter, &(*eCenter.begin()), &(*weights.begin()));
    testPoints.push_back(eCenter);
    typePoint.push_back("edge center");
    inOuts.push_back(1);
  }

  // Test points at the center of each face
  for (int f = 0; f < aCell->GetNumberOfFaces(); ++f)
  {
    std::vector<double> fCenter(3);
    svtkCell* c = aCell->GetFace(f);
    c->GetParametricCenter(pcenter);
    c->EvaluateLocation(subId, pcenter, &(*fCenter.begin()), &(*weights.begin()));
    testPoints.push_back(fCenter);
    inOuts.push_back(1);
    typePoint.push_back("face center");
  }

  // Test a point outside the cell
  if (aCell->GetNumberOfPoints() > 0)
  {
    std::vector<double> outside(3, -12345.0);
    testPoints.push_back(outside);
    inOuts.push_back(0);
    typePoint.push_back("outside point");
  }
  for (size_t p = 0; p < testPoints.size(); ++p)
  {
    double closestPoint[3], pcoords[3], dist2;
    int inOut = aCell->EvaluatePosition(
      &(*testPoints[p].begin()), closestPoint, subId, pcoords, dist2, &(*weights.begin()));
    if ((inOut == 0 || inOut == -1) && inOuts[p] == 0)
    {
      continue;
    }
    else if (inOut == 1 && dist2 == 0.0 && inOuts[p] == 1)
    {
      continue;
    }
    else if (inOut == 1 && dist2 != 0.0 && inOuts[p] == 0)
    {
      continue;
    }
    // inOut failed
    std::cout << typePoint[p] << " failed inOut: " << inOut << " "
              << "point: " << testPoints[p][0] << ", " << testPoints[p][1] << ", "
              << testPoints[p][2] << "-> "
              << "pcoords: " << pcoords[0] << ", " << pcoords[1] << ", " << pcoords[2] << ": "
              << "closestPoint: " << closestPoint[0] << ", " << closestPoint[1] << ", "
              << closestPoint[2] << " "
              << "dist2: " << dist2;
    std::cout << " weights: ";
    for (int w = 0; w < aCell->GetNumberOfPoints(); ++w)
    {
      std::cout << weights[w] << " ";
    }
    std::cout << std::endl;
    status2 += 1;
  }
  if (status2)
  {
    ++status;
    std::cout << "FAILED" << std::endl;
  }
  else
  {
    std::cout << "PASSED" << std::endl;
  }

  if (aCell->GetNumberOfPoints() > 0 && aCell->GetCellDimension() > 0)
  {
    std::cout << "  Testing IntersectWithLine...";
    double tol = 1.e-5;
    double t;
    double startPoint[3];
    double endPoint[3];
    double intersection[3], pintersection[3];
    aCell->GetParametricCenter(pcenter);
    aCell->EvaluateLocation(subId, pcenter, startPoint, &(*weights.begin()));
    endPoint[0] = startPoint[0];
    endPoint[1] = startPoint[1];
    endPoint[2] = startPoint[2] + aCell->GetLength2();
    startPoint[2] = startPoint[2] - aCell->GetLength2();
    int status3 = 0;
    int result =
      aCell->IntersectWithLine(startPoint, endPoint, tol, t, intersection, pintersection, subId);
    if (result == 0)
    {
      ++status3;
    }
    else
    {
      std::cout << " t: " << t << " ";
    }
    startPoint[2] = endPoint[2] + aCell->GetLength2();
    result =
      aCell->IntersectWithLine(startPoint, endPoint, tol, t, intersection, pintersection, subId);
    if (result == 1)
    {
      ++status3;
    }

    if (status3 != 0)
    {
      ++status;
      std::cout << "...FAILED" << std::endl;
    }
    else
    {
      std::cout << "...PASSED" << std::endl;
    }
  }

  // Triangulate
  std::cout << "  Testing Triangulate...";
  int index = 0;
  svtkSmartPointer<svtkIdList> ptIds = svtkSmartPointer<svtkIdList>::New();
  ptIds->SetNumberOfIds(100);
  svtkSmartPointer<svtkPoints> triPoints = svtkSmartPointer<svtkPoints>::New();
  aCell->Triangulate(index, ptIds, triPoints);
  int pts = ptIds->GetNumberOfIds();
  if (aCell->GetCellDimension() == 0)
  {
    std::cout << "Generated " << pts << " Points";
  }
  else if (aCell->GetCellDimension() == 1)
  {
    std::cout << "Generated " << pts / 2 << " Lines";
  }
  else if (aCell->GetCellDimension() == 2)
  {
    std::cout << "Generated " << pts / 3 << " Triangles";
  }
  else if (aCell->GetCellDimension() == 3)
  {
    std::cout << "Generated " << pts / 4 << " Tetra";
  }
  std::cout << "...PASSED" << std::endl;

  if (status)
  {
    std::cout << aCell->GetClassName() << " FAILED" << std::endl;
  }
  else
  {
    std::cout << aCell->GetClassName() << " PASSED" << std::endl;
  }
  return status;
}
