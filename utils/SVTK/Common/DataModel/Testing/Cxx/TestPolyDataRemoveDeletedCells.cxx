/*=========================================================================

  Program:   Visualization Toolkit
  Module:  TestPolyDataRemoveDeletedCells.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

   This software is distributed WITHOUT ANY WARRANTY; without even
   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
   PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkCell.h"
#include "svtkCellData.h"
#include "svtkDataArray.h"
#include "svtkIdList.h"
#include "svtkIntArray.h"
#include "svtkPoints.h"
#include "svtkPolyData.h"
#include "svtkSmartPointer.h"

#define SVTK_SUCCESS 0
#define SVTK_FAILURE 1

int CheckDeletedCells()
{
  svtkPoints* points = svtkPoints::New();
  for (svtkIdType i = 0; i < 10; i++)
  {
    points->InsertNextPoint(i, i, i);
  }
  svtkSmartPointer<svtkPolyData> poly = svtkSmartPointer<svtkPolyData>::New();
  poly->SetPoints(points);
  points->Delete();
  poly->AllocateEstimate(10, 1);
  for (svtkIdType i = 0; i < 10; i++)
  {
    poly->InsertNextCell(SVTK_VERTEX, 1, &i);
  }
  poly->BuildCells();
  if (poly->GetNumberOfPoints() != 10 || poly->GetNumberOfCells() != 10)
  {
    std::cout << "Wrong number of input points or cells!" << std::endl;
    return SVTK_FAILURE;
  }

  for (svtkIdType i = 0; i < 5; i++)
  {
    poly->DeleteCell(i * 2 + 1);
  }
  poly->RemoveDeletedCells();

  if (poly->GetNumberOfCells() != 5)
  {
    std::cout << "Wrong number of removed cells!" << std::endl;
    return SVTK_FAILURE;
  }

  for (svtkIdType i = 0; i < 5; i++)
  {
    svtkCell* cell = poly->GetCell(i);
    if (cell->GetPointId(0) != i * 2)
    {
      std::cout << "Wrong point of cell " << i << ", should be point " << i * 2 << " but is "
                << cell->GetPointId(0) << std::endl;
      return SVTK_FAILURE;
    }
  }

  return SVTK_SUCCESS;
}

int TestPolyDataRemoveDeletedCells(int, char*[])
{
  int numCells = 8;

  svtkPoints* pts = svtkPoints::New();

  pts->InsertNextPoint(1, 0, 0); // 0
  pts->InsertNextPoint(3, 0, 0); // 1
  pts->InsertNextPoint(5, 0, 0); // 2
  pts->InsertNextPoint(7, 0, 0); // 3

  pts->InsertNextPoint(0, 2, 0); // 4
  pts->InsertNextPoint(2, 2, 0); // 5
  pts->InsertNextPoint(4, 2, 0); // 6
  pts->InsertNextPoint(6, 2, 0); // 7

  pts->InsertNextPoint(9, 0, 0);  // 8
  pts->InsertNextPoint(11, 0, 0); // 9
  pts->InsertNextPoint(8, 2, 0);  // 10
  pts->InsertNextPoint(10, 2, 0); // 11

  svtkPolyData* pd = svtkPolyData::New();
  pd->AllocateEstimate(numCells, 1);
  pd->SetPoints(pts);

  pts->Delete();

  svtkIdList* cell;
  cell = svtkIdList::New();

  // adding different cells in arbitrary order

  // 1

  cell->InsertNextId(4);
  cell->InsertNextId(0);
  cell->InsertNextId(5);
  cell->InsertNextId(1);
  cell->InsertNextId(6);

  pd->InsertNextCell(SVTK_TRIANGLE_STRIP, cell);

  cell->Reset();

  // 2

  cell->InsertNextId(1);
  cell->InsertNextId(6);
  cell->InsertNextId(2);
  cell->InsertNextId(7);
  cell->InsertNextId(3);

  pd->InsertNextCell(SVTK_TRIANGLE_STRIP, cell);

  cell->Reset();

  // 3

  cell->InsertNextId(0);

  pd->InsertNextCell(SVTK_VERTEX, cell);

  cell->Reset();

  // 4

  cell->InsertNextId(0);
  cell->InsertNextId(1);
  cell->InsertNextId(2);
  cell->InsertNextId(3);
  cell->InsertNextId(8);
  cell->InsertNextId(11);
  cell->InsertNextId(10);
  cell->InsertNextId(7);
  cell->InsertNextId(6);
  cell->InsertNextId(5);
  cell->InsertNextId(4);

  pd->InsertNextCell(SVTK_POLY_LINE, cell);

  cell->Reset();

  // 5

  cell->InsertNextId(3);
  cell->InsertNextId(8);
  cell->InsertNextId(9);
  cell->InsertNextId(11);
  cell->InsertNextId(10);
  cell->InsertNextId(7);

  pd->InsertNextCell(SVTK_POLYGON, cell);

  cell->Reset();

  // 6

  cell->InsertNextId(1);

  pd->InsertNextCell(SVTK_VERTEX, cell);

  cell->Reset();

  // 7

  cell->InsertNextId(3);
  cell->InsertNextId(10);

  pd->InsertNextCell(SVTK_LINE, cell);

  cell->Reset();

  // 8

  cell->InsertNextId(8);
  cell->InsertNextId(9);
  cell->InsertNextId(11);

  pd->InsertNextCell(SVTK_TRIANGLE, cell);

  cell->Delete();

  svtkIntArray* scalars = svtkIntArray::New();

  scalars->InsertNextValue(SVTK_TRIANGLE_STRIP);
  scalars->InsertNextValue(SVTK_TRIANGLE_STRIP);
  scalars->InsertNextValue(SVTK_VERTEX);
  scalars->InsertNextValue(SVTK_POLY_LINE);
  scalars->InsertNextValue(SVTK_POLYGON);
  scalars->InsertNextValue(SVTK_VERTEX);
  scalars->InsertNextValue(SVTK_LINE);
  scalars->InsertNextValue(SVTK_TRIANGLE);

  pd->GetCellData()->SetScalars(scalars);

  scalars->Delete();

  // delete some cells

  pd->DeleteCell(0);
  pd->DeleteCell(3);
  pd->DeleteCell(5);
  pd->DeleteCell(7);

  int types[] = {
    // SVTK_TRIANGLE_STRIP,
    SVTK_TRIANGLE_STRIP, SVTK_VERTEX,
    // SVTK_POLY_LINE,
    SVTK_POLYGON,
    // SVTK_VERTEX,
    SVTK_LINE,
    // SVTK_TRIANGLE
  };

  pd->RemoveDeletedCells();

  svtkIntArray* newScalars = svtkArrayDownCast<svtkIntArray>(pd->GetCellData()->GetScalars());

  int retVal = SVTK_SUCCESS;

  if (pd->GetNumberOfCells() != numCells - 4)
  {
    std::cout << "Number of cells does not match!" << std::endl;
    retVal = SVTK_FAILURE;
  }

  for (svtkIdType i = 0; i < pd->GetNumberOfCells(); i++)
  {
    if (pd->GetCellType(i) != newScalars->GetValue(i))
    {
      std::cout << "Cell " << i << " has wrong data. Value should be " << pd->GetCellType(i)
                << " but is " << newScalars->GetValue(i) << std::endl;
      retVal = SVTK_FAILURE;
    }

    if (pd->GetCellType(i) != types[i])
    {
      std::cout << "Cell " << i << " has wrong type. Value should be " << pd->GetCellType(i)
                << " but is " << newScalars->GetValue(i) << std::endl;
      retVal = SVTK_FAILURE;
    }
  }

  pd->Delete();
  if (retVal != SVTK_SUCCESS)
  {
    return retVal;
  }

  return CheckDeletedCells();
}
