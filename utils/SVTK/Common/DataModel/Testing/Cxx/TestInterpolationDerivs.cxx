/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestInterpolationDerivs.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#define SVTK_EPSILON 1e-10

// Subclass of svtkCell
//#include "svtkEmptyCell.h"
#include "svtkGenericCell.h"
#include "svtkLine.h"
#include "svtkPixel.h"
//#include "svtkPolygon.h"
//#include "svtkPolyLine.h"
//#include "svtkPolyVertex.h"
#include "svtkQuad.h"
#include "svtkTriangle.h"
//#include "svtkTriangleStrip.h"
#include "svtkVertex.h"

// Subclass of svtkCell3D
//#include "svtkConvexPointSet.h"
#include "svtkHexagonalPrism.h"
#include "svtkHexahedron.h"
#include "svtkPentagonalPrism.h"
#include "svtkPyramid.h"
#include "svtkTetra.h"
#include "svtkVoxel.h"
#include "svtkWedge.h"

// Subclass of svtkNonLinearCell
#include "svtkQuadraticEdge.h"
#include "svtkQuadraticHexahedron.h"
#include "svtkQuadraticPyramid.h"
#include "svtkQuadraticQuad.h"
#include "svtkQuadraticTetra.h"
#include "svtkQuadraticTriangle.h"
#include "svtkQuadraticWedge.h"

// New bi-class from gebbert
#include "svtkBiQuadraticQuad.h"
#include "svtkBiQuadraticQuadraticHexahedron.h"
#include "svtkBiQuadraticQuadraticWedge.h"
#include "svtkQuadraticLinearQuad.h"
#include "svtkQuadraticLinearWedge.h"
#include "svtkTriQuadraticHexahedron.h"

// New Bi-Class
#include "svtkBiQuadraticTriangle.h"
#include "svtkCubicLine.h"

template <class TCell>
int TestOneInterpolationDerivs(double eps = SVTK_EPSILON)
{
  TCell* cell = TCell::New();
  int numPts = cell->GetNumberOfPoints();
  int dim = cell->GetCellDimension();
  double* derivs = new double[dim * numPts];
  double* coords = cell->GetParametricCoords();
  int r = 0;
  for (int i = 0; i < numPts; ++i)
  {
    double* point = coords + 3 * i;
    double sum = 0.;
    cell->InterpolateDerivs(point, derivs); // static function
    for (int j = 0; j < dim * numPts; j++)
    {
      sum += derivs[j];
    }
    if (fabs(sum) > eps)
    {
      ++r;
    }
  }

  // Let's test zero condition on the center point:
  double center[3];
  cell->GetParametricCenter(center);
  cell->InterpolateDerivs(center, derivs); // static function
  double sum = 0.;
  for (int j = 0; j < dim * numPts; j++)
  {
    sum += derivs[j];
  }
  if (fabs(sum) > eps)
  {
    ++r;
  }

  cell->Delete();
  delete[] derivs;
  return r;
}

int TestInterpolationDerivs(int, char*[])
{
  int r = 0;

  // Subclasses of svtkCell3D
  // r += TestOneInterpolationDerivs<svtkEmptyCell>(); // not implemented
  // r += TestOneInterpolationDerivs<svtkGenericCell>(); // not implemented
  // r += TestOneInterpolationDerivs<svtkLine>();
  r += TestOneInterpolationDerivs<svtkPixel>();
  // r += TestOneInterpolationDerivs<svtkPolygon>(); // not implemented
  // r += TestOneInterpolationDerivs<svtkPolyLine>(); // not implemented
  // r += TestOneInterpolationDerivs<svtkPolyVertex>(); // not implemented
  r += TestOneInterpolationDerivs<svtkQuad>();
  r += TestOneInterpolationDerivs<svtkTriangle>();
  // r += TestOneInterpolationDerivs<svtkTriangleStrip>(); // not implemented
  // r += TestOneInterpolationDerivs<svtkVertex>();

  // Subclasses of svtkCell3D
  // r += TestOneInterpolationDerivs<svtkConvexPointSet>(); // not implemented
  r += TestOneInterpolationDerivs<svtkHexagonalPrism>();
  r += TestOneInterpolationDerivs<svtkHexahedron>();
  r += TestOneInterpolationDerivs<svtkPentagonalPrism>(1.e-05);
  r += TestOneInterpolationDerivs<svtkPyramid>();
  // r += TestOneInterpolationDerivs<svtkTetra>();
  r += TestOneInterpolationDerivs<svtkVoxel>();
  r += TestOneInterpolationDerivs<svtkWedge>();

  // Subclasses of svtkNonLinearCell
  r += TestOneInterpolationDerivs<svtkQuadraticEdge>();
  r += TestOneInterpolationDerivs<svtkQuadraticHexahedron>();
  r += TestOneInterpolationDerivs<svtkQuadraticPyramid>();
  r += TestOneInterpolationDerivs<svtkQuadraticQuad>();
  r += TestOneInterpolationDerivs<svtkQuadraticTetra>();
  r += TestOneInterpolationDerivs<svtkQuadraticTriangle>();
  r += TestOneInterpolationDerivs<svtkQuadraticWedge>();

  // New bi-class
  r += TestOneInterpolationDerivs<svtkBiQuadraticQuad>();
  r += TestOneInterpolationDerivs<svtkBiQuadraticQuadraticHexahedron>();
  r += TestOneInterpolationDerivs<svtkBiQuadraticQuadraticWedge>();
  r += TestOneInterpolationDerivs<svtkQuadraticLinearQuad>();
  r += TestOneInterpolationDerivs<svtkQuadraticLinearWedge>();
  r += TestOneInterpolationDerivs<svtkTriQuadraticHexahedron>();
  r += TestOneInterpolationDerivs<svtkBiQuadraticTriangle>();
  r += TestOneInterpolationDerivs<svtkCubicLine>();

  return r;
}
