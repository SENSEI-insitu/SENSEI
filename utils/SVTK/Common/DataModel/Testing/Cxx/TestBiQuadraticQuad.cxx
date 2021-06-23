/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestBiQuadraticQuad.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkBiQuadraticQuad.h"
#include "svtkCellArray.h"
#include "svtkDoubleArray.h"
#include "svtkMathUtilities.h"
#include "svtkNew.h"
#include "svtkPointData.h"
#include "svtkPolyData.h"
#include "svtkProbeFilter.h"
#include "svtkUnstructuredGrid.h"

//----------------------------------------------------------------------------
int TestBiQuadraticQuad(int, char*[])
{
  svtkNew<svtkPoints> points;
  points->InsertNextPoint(0.0, 0.0, 0.0);
  points->InsertNextPoint(1.0, 0.0, 0.0);
  points->InsertNextPoint(1.0, 1.0, 0.0);
  points->InsertNextPoint(0.0, 1.0, 0.0);
  points->InsertNextPoint(0.5, 0.0, 0.0);
  points->InsertNextPoint(1.0, 0.5, 0.0);
  points->InsertNextPoint(0.5, 1.0, 0.0);
  points->InsertNextPoint(0.0, 0.5, 0.0);
  points->InsertNextPoint(0.5, 0.5, 0.0);

  svtkNew<svtkBiQuadraticQuad> quad;
  for (int i = 0; i < 9; ++i)
  {
    quad->GetPointIds()->SetId(i, i);
  }

  svtkNew<svtkCellArray> cellArray;
  cellArray->InsertNextCell(quad);

  svtkNew<svtkDoubleArray> uArray;
  uArray->SetName("u");
  uArray->SetNumberOfComponents(1);
  uArray->SetNumberOfTuples(9);
  // set u(x, y) = x
  for (int i = 0; i < 9; i++)
  {
    uArray->SetValue(i, points->GetPoint(i)[0]);
  }

  svtkNew<svtkUnstructuredGrid> grid;
  grid->SetPoints(points);
  grid->SetCells(SVTK_BIQUADRATIC_QUAD, cellArray);
  grid->GetPointData()->SetScalars(uArray);

  double probeX = 2.0 / 3.0;
  double probeY = 0.25;
  svtkNew<svtkPoints> probePoints;
  probePoints->InsertNextPoint(probeX, probeY, 0.0);
  svtkNew<svtkPolyData> probePolyData;
  probePolyData->SetPoints(probePoints);

  svtkNew<svtkProbeFilter> prober;
  prober->SetSourceData(grid);
  prober->SetInputData(probePolyData);
  prober->Update();

  svtkDataArray* data = prober->GetOutput()->GetPointData()->GetScalars();
  svtkDoubleArray* doubleData = svtkArrayDownCast<svtkDoubleArray>(data);

  double interpolated(0.0);
  if (doubleData)
  {
    interpolated = doubleData->GetComponent(0, 0);
  }
  else
  {
    cout << "Failed to downcast prober scalars." << endl;
  }
  if (!svtkMathUtilities::FuzzyCompare(interpolated, probeX, 1.0e-6))
  {
    cout << "Interpolated value of " << interpolated << " with probe value " << probeX
         << " difference of " << (interpolated - probeX) << endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
