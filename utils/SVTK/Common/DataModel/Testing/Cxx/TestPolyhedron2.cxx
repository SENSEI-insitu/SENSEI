/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestPolyhedron2.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkPlane.h"
#include "svtkPolyhedron.h"
#include "svtkUnstructuredGrid.h"

#include "svtkCutter.h"
#include "svtkNew.h"
#include "svtkPlane.h"
#include "svtkTestUtilities.h"
#include "svtkXMLPolyDataWriter.h"
#include "svtkXMLUnstructuredGridReader.h"
#include "svtkXMLUnstructuredGridWriter.h"

// Test of contour/clip of svtkPolyhedron. uses input from
// https://gitlab.kitware.com/svtk/svtk/issues/14485
int TestPolyhedron2(int argc, char* argv[])
{
  if (argc < 3)
    return 1; // test not run with data on the command line

  svtkObject::GlobalWarningDisplayOff();

  const char* filename = argv[2];
  svtkNew<svtkXMLUnstructuredGridReader> reader;
  reader->SetFileName(filename);
  reader->Update();

  svtkUnstructuredGrid* pGrid = reader->GetOutput();

  svtkNew<svtkCutter> cutter;
  svtkNew<svtkPlane> p;
  p->SetOrigin(pGrid->GetCenter());
  p->SetNormal(1, 0, 0);

  cutter->SetCutFunction(p);
  cutter->SetGenerateTriangles(0);

  cutter->SetInputConnection(0, reader->GetOutputPort());
  cutter->Update();

  svtkPolyData* output = svtkPolyData::SafeDownCast(cutter->GetOutputDataObject(0));
  if (output->GetNumberOfCells() != 2)
  {
    std::cerr << "Expected 2 polygons but found " << output->GetNumberOfCells()
              << " polygons in sliced polyhedron." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
