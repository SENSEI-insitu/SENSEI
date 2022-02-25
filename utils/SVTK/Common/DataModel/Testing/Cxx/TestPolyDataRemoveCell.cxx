/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestPolyDataRemoveCell.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkCellData.h"
#include "svtkIdTypeArray.h"
#include "svtkIntArray.h"
#include "svtkPoints.h"
#include "svtkPolyData.h"
#include "svtkSmartPointer.h"

int TestPolyDataRemoveCell(int, char*[])
{
  int rval = 0;

  svtkIdType numPoints = 20;
  svtkIdType numVerts = 5;
  svtkIdType numLines = 8;
  svtkIdType numTriangles = 3;
  svtkIdType numStrips = 2;
  svtkIdType numCells = numVerts + numLines + numTriangles + numStrips;
  svtkIdType i;

  svtkPoints* points = svtkPoints::New();
  points->SetNumberOfPoints(numPoints);
  for (i = 0; i < numPoints; i++)
  {
    double loc[3] = { static_cast<double>(i), static_cast<double>(i * i), 0.0 };
    points->InsertPoint(i, loc);
  }
  svtkSmartPointer<svtkPolyData> poly = svtkSmartPointer<svtkPolyData>::New();
  poly->AllocateExact(numCells, numCells);
  poly->SetPoints(points);
  points->Delete();

  for (i = 0; i < numVerts; i++)
  {
    poly->InsertNextCell(SVTK_VERTEX, 1, &i);
  }

  for (i = 0; i < numLines; i++)
  {
    svtkIdType pts[2] = { i, i + 1 };
    poly->InsertNextCell(SVTK_LINE, 2, pts);
  }

  for (i = 0; i < numTriangles; i++)
  {
    svtkIdType pts[3] = { 0, i + 1, i + 2 };
    poly->InsertNextCell(SVTK_TRIANGLE, 3, pts);
  }

  for (i = 0; i < numStrips; i++)
  {
    svtkIdType pts[3] = { 0, i + 1, i + 2 };
    poly->InsertNextCell(SVTK_TRIANGLE_STRIP, 3, pts);
  }

  svtkIntArray* cellTypes = svtkIntArray::New();
  const char ctName[] = "cell types";
  cellTypes->SetName(ctName);
  cellTypes->SetNumberOfComponents(1);
  cellTypes->SetNumberOfTuples(numCells);
  for (i = 0; i < numCells; i++)
  {
    cellTypes->SetValue(i, poly->GetCellType(i));
  }
  poly->GetCellData()->AddArray(cellTypes);
  cellTypes->Delete();

  svtkIdTypeArray* cellPoints = svtkIdTypeArray::New();
  const char cpName[] = "cell points";
  cellPoints->SetName(cpName);
  cellPoints->SetNumberOfComponents(4); // num points + point ids
  cellPoints->SetNumberOfTuples(numCells);
  for (i = 0; i < numCells; i++)
  {
    svtkIdType npts;
    const svtkIdType* pts;
    poly->GetCellPoints(i, npts, pts);
    svtkIdType data[4] = { npts, pts[0], 0, 0 };
    for (svtkIdType j = 1; j < npts; j++)
    {
      data[j + 1] = pts[j];
    }
    cellPoints->SetTypedTuple(i, data);
  }
  poly->GetCellData()->AddArray(cellPoints);
  cellPoints->Delete();

  poly->BuildCells();
  // now that we're all set up, try deleting one of each object
  poly->DeleteCell(numVerts - 1);                           // vertex
  poly->DeleteCell(numVerts + numLines - 1);                // line
  poly->DeleteCell(numVerts + numLines + numTriangles - 1); // triangle
  poly->DeleteCell(numCells - 1);                           // strip

  poly->RemoveDeletedCells();

  if (poly->GetNumberOfCells() != numCells - 4)
  {
    cout << "Wrong number of cells after removal.\n";
    return 1;
  }

  // the arrays should have been changed so get them again...
  cellTypes = svtkArrayDownCast<svtkIntArray>(poly->GetCellData()->GetArray(ctName));
  cellPoints = svtkArrayDownCast<svtkIdTypeArray>(poly->GetCellData()->GetArray(cpName));

  // check the cell types and arrays
  for (i = 0; i < poly->GetNumberOfCells(); i++)
  {
    if (cellTypes->GetValue(i) != poly->GetCellType(i))
    {
      cout << "Problem with cell type for cell " << i << endl;
      return 1;
    }
  }

  // check the cell's points
  for (i = 0; i < poly->GetNumberOfCells(); i++)
  {
    svtkIdType npts;
    const svtkIdType* pts;
    poly->GetCellPoints(i, npts, pts);
    svtkIdType data[4];
    cellPoints->GetTypedTuple(i, data);
    if (data[0] != npts)
    {
      cout << "Problem with the number of points for cell " << i << endl;
      return 1;
    }
    for (svtkIdType j = 0; j < npts; j++)
    {
      if (pts[j] != data[j + 1])
      {
        cout << "Problem with point " << j << " for cell " << i << endl;
        return 1;
      }
    }
  }

  return rval;
}
