/*=========================================================================

Program:   Visualization Toolkit
Module:    TestPolyhedronCutter.cxx

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkCutter.h"
#include "svtkPlane.h"
#include "svtkPolygon.h"
#include "svtkUnstructuredGrid.h"
#include "svtkXMLUnstructuredGridReader.h"

#include <set>

using namespace std;

int TestPolyhedronCutter(int argc, char* argv[])
{
  svtkObject::GlobalWarningDisplayOff();
  svtkNew<svtkXMLUnstructuredGridReader> r;

  if (argc < 3)
  {
    cout << "Not enough arguments. Passing test nonetheless.";
    return EXIT_SUCCESS;
  }

  char* fname = argv[1];
  r->SetFileName(fname);
  r->Update();

  svtkUnstructuredGrid* grid = r->GetOutput();
  if (grid->GetNumberOfCells() != 1)
  {
    return EXIT_FAILURE;
  }

  svtkNew<svtkCutter> cut;
  svtkNew<svtkPlane> cutPlane;
  cutPlane->SetOrigin(0, 0, 350);
  cutPlane->SetNormal(1, 0, 0);
  cut->SetCutFunction(cutPlane);

  cut->AddInputConnection(r->GetOutputPort());
  cut->Update();

  svtkPolyData* result = cut->GetOutput();
  if (!result)
  {
    return EXIT_FAILURE;
  }

  if (result->GetNumberOfCells() != 1)
  {
    return EXIT_FAILURE;
  }

  svtkCell* aCell = result->GetCell(0);

  svtkPolygon* cell = svtkPolygon::SafeDownCast(aCell);

  if (!cell)
  {
    return EXIT_FAILURE;
  }

  if (cell->GetNumberOfEdges() != 5)
  {
    cerr << "The resulting polygon consists of " << cell->GetNumberOfEdges()
         << " edges instead of the expected 5 edges." << endl;
    return EXIT_FAILURE;
  }

  svtkIdList* ids = cell->GetPointIds();
  set<svtkIdType> uniqueIds;

  for (int i = 0; i < 5; ++i)
  {
    uniqueIds.insert(ids->GetId(i));
  }

  if (uniqueIds.size() != 5)
  {
    cerr << "The resulting polygon consists of invalid edges" << endl;
    return EXIT_FAILURE;
  }

  // For the second slice operation, the only requirement (currently) is that it returns *a result*
  fname = argv[2];
  r->SetFileName(fname);
  r->Update();

  grid = r->GetOutput();
  if (grid->GetNumberOfCells() <= 0)
  {
    return EXIT_FAILURE;
  }

  cut->Update();

  result = cut->GetOutput();
  if (!result || result->GetNumberOfCells() <= 0)
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
