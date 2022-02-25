/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestCellLocator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkExtractGeometry.h"
#include "svtkImageData.h"
#include "svtkPolyData.h"
#include "svtkSmartPointer.h"
#include "svtkSphere.h"
#include "svtkSphereSource.h"
#include "svtkStaticCellLinks.h"
#include "svtkStaticCellLinksTemplate.h"
#include "svtkTimerLog.h"
#include "svtkUnstructuredGrid.h"

// Test the building of static cell links in both unstructured and structured
// grids.
int TestStaticCellLinks(int, char*[])
{
  int dataDim = 3;

  // First create a volume which will be converted to an unstructured grid
  svtkSmartPointer<svtkImageData> volume = svtkSmartPointer<svtkImageData>::New();
  volume->SetDimensions(dataDim, dataDim, dataDim);
  volume->AllocateScalars(SVTK_INT, 1);

  //----------------------------------------------------------------------------
  // Build links on volume
  svtkSmartPointer<svtkStaticCellLinks> imlinks = svtkSmartPointer<svtkStaticCellLinks>::New();
  imlinks->BuildLinks(volume);

  svtkIdType ncells = imlinks->GetNumberOfCells(0);
  const svtkIdType* imcells = imlinks->GetCells(0);
  cout << "Volume:\n";
  cout << "   Lower Left corner (numCells, cells): " << ncells << " (";
  for (int i = 0; i < ncells; ++i)
  {
    cout << imcells[i];
    if (i < (ncells - 1))
      cout << ",";
  }
  cout << ")\n";
  if (ncells != 1 || imcells[0] != 0)
  {
    return EXIT_FAILURE;
  }

  ncells = imlinks->GetNumberOfCells(13);
  imcells = imlinks->GetCells(13);
  cout << "   Center (ncells, cells): " << ncells << " (";
  for (int i = 0; i < ncells; ++i)
  {
    cout << imcells[i];
    if (i < (ncells - 1))
      cout << ",";
  }
  cout << ")\n";
  if (ncells != 8)
  {
    return EXIT_FAILURE;
  }

  ncells = imlinks->GetNumberOfCells(26);
  imcells = imlinks->GetCells(26);
  cout << "   Upper Right corner (ncells, cells): " << ncells << " (";
  for (int i = 0; i < ncells; ++i)
  {
    cout << imcells[i];
    if (i < (ncells - 1))
      cout << ",";
  }
  cout << ")\n";
  if (ncells != 1 || imcells[0] != 7)
  {
    return EXIT_FAILURE;
  }

  //----------------------------------------------------------------------------
  // Unstructured grid
  svtkSmartPointer<svtkSphere> sphere = svtkSmartPointer<svtkSphere>::New();
  sphere->SetCenter(0, 0, 0);
  sphere->SetRadius(100000);

  // Side effect of this filter is conversion of volume to unstructured grid
  svtkSmartPointer<svtkExtractGeometry> extract = svtkSmartPointer<svtkExtractGeometry>::New();
  extract->SetInputData(volume);
  extract->SetImplicitFunction(sphere);
  extract->Update();

  // Grab the output, build links on unstructured grid
  svtkSmartPointer<svtkUnstructuredGrid> ugrid = extract->GetOutput();

  svtkStaticCellLinksTemplate<int> slinks;
  slinks.BuildLinks(ugrid);

  int numCells = slinks.GetNumberOfCells(0);
  const int* cells = slinks.GetCells(0);
  cout << "\nUnstructured Grid:\n";
  cout << "   Lower Left corner (numCells, cells): " << numCells << " (";
  for (int i = 0; i < numCells; ++i)
  {
    cout << cells[i];
    if (i < (numCells - 1))
      cout << ",";
  }
  cout << ")\n";
  if (numCells != 1 || cells[0] != 0)
  {
    return EXIT_FAILURE;
  }

  numCells = slinks.GetNumberOfCells(13);
  cells = slinks.GetCells(13);
  cout << "   Center (numCells, cells): " << numCells << " (";
  for (int i = 0; i < numCells; ++i)
  {
    cout << cells[i];
    if (i < (numCells - 1))
      cout << ",";
  }
  cout << ")\n";
  if (numCells != 8)
  {
    return EXIT_FAILURE;
  }

  numCells = slinks.GetNumberOfCells(26);
  cells = slinks.GetCells(26);
  cout << "   Upper Right corner (numCells, cells): " << numCells << " (";
  for (int i = 0; i < numCells; ++i)
  {
    cout << cells[i];
    if (i < (numCells - 1))
      cout << ",";
  }
  cout << ")\n";
  if (numCells != 1 || cells[0] != 7)
  {
    return EXIT_FAILURE;
  }

  //----------------------------------------------------------------------------
  // Polydata
  svtkSmartPointer<svtkSphereSource> ss = svtkSmartPointer<svtkSphereSource>::New();
  ss->SetThetaResolution(12);
  ss->SetPhiResolution(10);
  ss->Update();

  svtkSmartPointer<svtkPolyData> pdata = ss->GetOutput();

  slinks.Initialize(); // reuse
  slinks.BuildLinks(pdata);

  // The first point is at the pole
  numCells = slinks.GetNumberOfCells(0);
  cells = slinks.GetCells(0);
  cout << "\nPolydata:\n";
  cout << "   Pole: (numCells, cells): " << numCells << " (";
  for (int i = 0; i < numCells; ++i)
  {
    cout << cells[i];
    if (i < (numCells - 1))
      cout << ",";
  }
  cout << ")\n";
  if (numCells != 12)
  {
    return EXIT_FAILURE;
  }

  // The next point is at near the equator
  numCells = slinks.GetNumberOfCells(5);
  cells = slinks.GetCells(5);
  cout << "   Equator: (numCells, cells): " << numCells << " (";
  for (int i = 0; i < numCells; ++i)
  {
    cout << cells[i];
    if (i < (numCells - 1))
      cout << ",";
  }
  cout << ")\n";
  if (numCells != 6)
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
