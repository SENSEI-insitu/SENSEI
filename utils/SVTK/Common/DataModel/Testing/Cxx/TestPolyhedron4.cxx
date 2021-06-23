/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestPolyhedron4.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkCutter.h"
#include "svtkNew.h"
#include "svtkPlane.h"
#include "svtkPolyData.h"
#include "svtkTestUtilities.h"
#include "svtkXMLUnstructuredGridReader.h"

int TestPolyhedron4(int argc, char* argv[])
{
  // Test that a nonwatertight polyhedron does no make svtkPolyhedron segfault
  char* filename = svtkTestUtilities::ExpandDataFileName(argc, argv,
    "Data/nonWatertightPolyhedron.vtu"); // this is in fact a bad name; the grid *is* watertight

  svtkNew<svtkXMLUnstructuredGridReader> reader;
  reader->SetFileName(filename);
  delete[] filename;

  svtkNew<svtkCutter> cutter;
  svtkNew<svtkPlane> p;
  p->SetOrigin(0, 0, 0);
  p->SetNormal(0, 1, 0);

  cutter->SetCutFunction(p);
  cutter->GenerateTrianglesOn();
  cutter->SetInputConnection(0, reader->GetOutputPort());

  // We want to check this does not segfault. We cannot check the error message
  svtkObject::GlobalWarningDisplayOff();
  cutter->Update();
  return EXIT_SUCCESS; // success
}
