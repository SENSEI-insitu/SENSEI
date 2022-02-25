/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestPath.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkIntArray.h"
#include "svtkNew.h"
#include "svtkPath.h"
#include "svtkPoints.h"

int TestPath(int, char*[])
{
  svtkNew<svtkPath> path;

  path->Allocate(7);

  path->InsertNextPoint(0.0, 0.0, 0.0, svtkPath::MOVE_TO);
  path->InsertNextPoint(1.0, 0.0, 0.0, svtkPath::LINE_TO);
  path->InsertNextPoint(1.5, 2.0, 0.0, svtkPath::CONIC_CURVE);
  path->InsertNextPoint(1.0, 1.5, 0.0, svtkPath::CONIC_CURVE);
  path->InsertNextPoint(0.0, 3.0, 0.0, svtkPath::CUBIC_CURVE);
  path->InsertNextPoint(0.0, 1.0, 0.0, svtkPath::CUBIC_CURVE);
  path->InsertNextPoint(path->GetPoint(0), svtkPath::CUBIC_CURVE);

  if (path->GetCodes()->GetNumberOfTuples() != 7 || path->GetPoints()->GetNumberOfPoints() != 7)
  {
    return EXIT_FAILURE;
  }

  path->Reset();

  if (path->GetCodes()->GetNumberOfTuples() != 0 || path->GetPoints()->GetNumberOfPoints() != 0)
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
