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

int TestPolygonBuilder3(int, char*[])
{

  svtkPolygonBuilder builder;
  builder.InsertTriangle(nullptr);
  svtkNew<svtkIdListCollection> polys;
  builder.GetPolygons(polys);
  if (polys->GetNumberOfItems() != 0)
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
