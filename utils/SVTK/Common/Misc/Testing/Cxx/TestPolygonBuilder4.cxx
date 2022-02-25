/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestPolygonBuilder4.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkIdListCollection.h"
#include "svtkNew.h"
#include "svtkPoints.h"
#include "svtkPolygonBuilder.h"
#include "svtkSmartPointer.h"

int TestPolygonBuilder4(int, char*[])
{

  svtkPolygonBuilder builder;
  svtkNew<svtkIdListCollection> polys;

  svtkSmartPointer<svtkPoints> points = svtkSmartPointer<svtkPoints>::New();
  svtkIdType a = points->InsertNextPoint(0, 0, 0);
  svtkIdType b = points->InsertNextPoint(1, 0, 0);
  svtkIdType c = points->InsertNextPoint(0, 1, 0);
  svtkIdType d = points->InsertNextPoint(1, 1, 0);

  // two counter-rotated triangles give unexpected results
#define NTRIANGLES 4
  svtkIdType triangles[NTRIANGLES][3] = { { b, c, a }, { d, c, b }, { c, b, a }, { d, b, c } };

  svtkIdType p[3];
  for (size_t i = 0; i < NTRIANGLES; i++)
  {
    for (size_t j = 0; j < 3; j++)
    {
      p[j] = triangles[i][j];
    }
    builder.InsertTriangle(p);
  }

  builder.GetPolygons(polys);

  svtkIdType expected(1);
  if (polys->GetNumberOfItems() != 1) // and a-b-c-d expected
  {
    svtkGenericWarningMacro(<< "number of items is " << polys->GetNumberOfItems() << " but expected "
                           << expected << endl);
    return EXIT_FAILURE;
  }

  svtkIdList* poly = polys->GetItem(0);
  expected = 4;
  if (poly->GetNumberOfIds() != 4)
  {
    svtkGenericWarningMacro(<< "number of ids is " << poly->GetNumberOfIds() << " but expected "
                           << expected << endl);
    return EXIT_FAILURE;
  }
  poly->Delete();
  polys->RemoveAllItems();

  return EXIT_SUCCESS;
}
