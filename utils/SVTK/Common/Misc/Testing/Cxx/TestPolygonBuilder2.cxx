/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestPolygonBuilder2.cxx

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

int TestPolygonBuilder2(int, char*[])
{

  svtkPolygonBuilder builder;
  svtkNew<svtkIdListCollection> polys;

  svtkSmartPointer<svtkPoints> points = svtkSmartPointer<svtkPoints>::New();
  svtkIdType a = points->InsertNextPoint(0, 0, 0);
  svtkIdType b = points->InsertNextPoint(1, 0, 0);
  svtkIdType c = points->InsertNextPoint(0, 1, 0);
  svtkIdType d = points->InsertNextPoint(1, 1, 0);
  svtkIdType e = points->InsertNextPoint(0, 0, 1);
  svtkIdType f = points->InsertNextPoint(1, 0, 1);
  svtkIdType g = points->InsertNextPoint(0, 1, 1);
  svtkIdType h = points->InsertNextPoint(1, 1, 1);

  // the following ordering of triangles is from a case that gives a crash
  // note that ALL indices are required to give a naked, unconnected edge 3-->2
  // leaving out the first three triangles, even though completely unconnected
  // from the last 6 triangles, does not trigger the error. The way to mitigate
  // the error is to disallow collapsed triangles.
#define NTRIANGLES 9
  svtkIdType triangles[NTRIANGLES][3] = { { e, g, f }, { h, g, e }, { h, h, g }, { b, c, a },
    { d, c, b }, { d, d, c }, { c, b, a }, { d, b, c }, { d, d, b } };

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

  if (polys->GetNumberOfItems() != 2) // expect abcd and efgh
  {
    cout << "number of items is " << polys->GetNumberOfItems() << endl;
    return EXIT_FAILURE;
  }

  svtkIdList* poly = polys->GetItem(0);
  svtkIdType expected(4);
  if (poly->GetNumberOfIds() != expected)
  {
    svtkGenericWarningMacro(<< "number of ids is " << poly->GetNumberOfIds() << " but expected "
                           << expected << endl);
    return EXIT_FAILURE;
  }
  poly->Delete();
  poly = polys->GetItem(1);
  if (poly->GetNumberOfIds() != expected)
  {
    svtkGenericWarningMacro(<< "number of ids is " << poly->GetNumberOfIds() << " but expected "
                           << expected << endl);
    return EXIT_FAILURE;
  }
  poly->Delete();
  polys->RemoveAllItems();

  return EXIT_SUCCESS;
}
