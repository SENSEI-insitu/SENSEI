/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestInterpolationFunctions.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#define SVTK_EPSILON 1e-10

// Subclass of svtkCell
#include "svtkEmptyCell.h"
#include "svtkGenericCell.h"
#include "svtkLine.h"
#include "svtkPixel.h"
#include "svtkPolyLine.h"
#include "svtkPolyVertex.h"
#include "svtkPolygon.h"
#include "svtkQuad.h"
#include "svtkTriangle.h"
#include "svtkTriangleStrip.h"
#include "svtkVertex.h"

// Subclass of svtkCell3D
#include "svtkConvexPointSet.h"
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

// Bi/Tri linear quadratic cells
#include "svtkBiQuadraticQuad.h"
#include "svtkBiQuadraticQuadraticHexahedron.h"
#include "svtkBiQuadraticQuadraticWedge.h"
#include "svtkBiQuadraticTriangle.h"
#include "svtkCubicLine.h"
#include "svtkQuadraticLinearQuad.h"
#include "svtkQuadraticLinearWedge.h"
#include "svtkTriQuadraticHexahedron.h"

#include <vector>

template <class TCell>
int TestOneInterpolationFunction(double eps = SVTK_EPSILON)
{
  TCell* cell = TCell::New();
  int numPts = cell->GetNumberOfPoints();
  std::vector<double> sf(numPts);
  double* coords = cell->GetParametricCoords();
  int r = 0;
  for (int i = 0; i < numPts; ++i)
  {
    double* point = coords + 3 * i;
    double sum = 0.;
    cell->InterpolateFunctions(point, sf.data()); // virtual function
    for (int j = 0; j < numPts; j++)
    {
      sum += sf[j];
      if (j == i)
      {
        if (fabs(sf[j] - 1) > eps)
        {
          std::cout << "fabs(sf[" << j << "] - 1): " << fabs(sf[j] - 1) << std::endl;
          ++r;
        }
      }
      else
      {
        if (fabs(sf[j] - 0) > eps)
        {
          std::cout << "fabs(sf[" << j << "] - 0): " << fabs(sf[j] - 0) << std::endl;
          ++r;
        }
      }
    }
    if (fabs(sum - 1) > eps)
    {
      ++r;
    }
  }

  // Let's test unity condition on the center point:
  double center[3];
  cell->GetParametricCenter(center);
  cell->InterpolateFunctions(center, sf.data()); // virtual function
  double sum = 0.;
  for (int j = 0; j < numPts; j++)
  {
    sum += sf[j];
  }
  if (fabs(sum - 1) > eps)
  {
    ++r;
  }

  cell->Delete();
  return r;
}

int TestInterpolationFunctions(int, char*[])
{
  int r = 0;
  // Subclass of svtkCell3D
  // r += TestOneInterpolationFunction<svtkEmptyCell>(); // not implemented
  // r += TestOneInterpolationFunction<svtkGenericCell>(); // not implemented
  r += TestOneInterpolationFunction<svtkLine>();
  r += TestOneInterpolationFunction<svtkPixel>();
  // r += TestOneInterpolationFunction<svtkPolygon>();
  // r += TestOneInterpolationFunction<svtkPolyLine>(); // not implemented
  // r += TestOneInterpolationFunction<svtkPolyVertex>(); // not implemented
  r += TestOneInterpolationFunction<svtkQuad>();
  r += TestOneInterpolationFunction<svtkTriangle>();
  // r += TestOneInterpolationFunction<svtkTriangleStrip>(); // not implemented
  r += TestOneInterpolationFunction<svtkVertex>();

  // Subclass of svtkCell3D
  // r += TestOneInterpolationFunction<svtkConvexPointSet>(); // not implemented
  r += TestOneInterpolationFunction<svtkHexagonalPrism>();
  r += TestOneInterpolationFunction<svtkHexahedron>();
  r += TestOneInterpolationFunction<svtkPentagonalPrism>(1.e-5);
  r += TestOneInterpolationFunction<svtkPyramid>();
  r += TestOneInterpolationFunction<svtkTetra>();
  r += TestOneInterpolationFunction<svtkVoxel>();
  r += TestOneInterpolationFunction<svtkWedge>();

  // Subclass of svtkNonLinearCell
  r += TestOneInterpolationFunction<svtkQuadraticEdge>();
  r += TestOneInterpolationFunction<svtkQuadraticHexahedron>();
  r += TestOneInterpolationFunction<svtkQuadraticPyramid>();
  r += TestOneInterpolationFunction<svtkQuadraticQuad>();
  r += TestOneInterpolationFunction<svtkQuadraticTetra>();
  r += TestOneInterpolationFunction<svtkQuadraticTriangle>();
  r += TestOneInterpolationFunction<svtkQuadraticWedge>();

  // Bi/Tri linear quadratic cells
  r += TestOneInterpolationFunction<svtkBiQuadraticQuad>();
  r += TestOneInterpolationFunction<svtkBiQuadraticQuadraticHexahedron>();
  r += TestOneInterpolationFunction<svtkBiQuadraticQuadraticWedge>();
  r += TestOneInterpolationFunction<svtkQuadraticLinearQuad>();
  r += TestOneInterpolationFunction<svtkQuadraticLinearWedge>();
  r += TestOneInterpolationFunction<svtkTriQuadraticHexahedron>();
  r += TestOneInterpolationFunction<svtkBiQuadraticTriangle>();
  r += TestOneInterpolationFunction<svtkCubicLine>();

  return r;
}
