/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestCutter.cxx

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

int TestPolygonBuilder(int, char*[])
{
  svtkSmartPointer<svtkPoints> points = svtkSmartPointer<svtkPoints>::New();
  svtkIdType a = points->InsertNextPoint(0, 0, 0);
  svtkIdType b = points->InsertNextPoint(1, 0, 0);
  svtkIdType c = points->InsertNextPoint(0, 1, 0);
  svtkIdType d = points->InsertNextPoint(1, 1, 0);
  svtkIdType e = points->InsertNextPoint(0.5, 0.5, 0);

  // The ordering of the vertices ensures that the normals of all of the
  // subtriangles are in the same direction (0,0,1)
  svtkIdType triangles[4][3] = { { e, c, a }, { e, a, b }, { e, b, d }, { e, d, c } };

  svtkPolygonBuilder builder;

  svtkIdType p[3];
  for (size_t i = 0; i < 4; i++)
  {
    for (size_t j = 0; j < 3; j++)
      p[j] = triangles[i][j];
    builder.InsertTriangle(p);
  }

  svtkNew<svtkIdListCollection> polys;
  builder.GetPolygons(polys);

  if (polys->GetNumberOfItems() != 1)
  {
    return EXIT_FAILURE;
  }

  svtkIdList* poly = polys->GetItem(0);
  if (poly->GetNumberOfIds() != 4)
  {
    return EXIT_FAILURE;
  }
  poly->Delete();
  polys->RemoveAllItems();

  return EXIT_SUCCESS;
}
