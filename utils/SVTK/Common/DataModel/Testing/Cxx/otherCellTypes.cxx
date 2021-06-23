/*=========================================================================

  Program:   Visualization Toolkit
  Module:    otherCellTypes.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// .NAME
// .SECTION Description
// this program tests the CellTypes

#include "svtkDebugLeaks.h"

#include "svtkCellType.h"
#include "svtkCellTypes.h"

void TestOCT()
{
  // actual test
  svtkCellTypes* ct = svtkCellTypes::New();
  ct->Allocate();

  ct->InsertCell(0, SVTK_QUAD, 0);
  ct->InsertNextCell(SVTK_PIXEL, 1);

  svtkUnsignedCharArray* cellTypes = svtkUnsignedCharArray::New();
  svtkIntArray* cellLocations = svtkIntArray::New();

  cellLocations->InsertNextValue(0);
  cellTypes->InsertNextValue(SVTK_QUAD);

  cellLocations->InsertNextValue(1);
  cellTypes->InsertNextValue(SVTK_PIXEL);

  cellLocations->InsertNextValue(2);
  cellTypes->InsertNextValue(SVTK_TETRA);

  ct->SetCellTypes(3, cellTypes, cellLocations);

  ct->GetCellLocation(1);
  ct->DeleteCell(1);

  ct->GetNumberOfTypes();

  ct->IsType(SVTK_QUAD);
  ct->IsType(SVTK_WEDGE);

  ct->InsertNextType(SVTK_WEDGE);
  ct->IsType(SVTK_WEDGE);

  ct->GetCellType(2);

  ct->GetActualMemorySize();

  svtkCellTypes* ct1 = svtkCellTypes::New();
  ct1->DeepCopy(ct);

  ct->Reset();
  ct->Squeeze();

  ct1->Delete();
  ct->Delete();
  cellLocations->Delete();
  cellTypes->Delete();
}

int otherCellTypes(int, char*[])
{
  TestOCT();

  // Might need to be adjusted if svtkCellTypes changes
  bool fail1 = (SVTK_NUMBER_OF_CELL_TYPES <= SVTK_HIGHER_ORDER_HEXAHEDRON);

  // svtkUnstructuredGrid uses uchar to store cellId
  bool fail2 = (SVTK_NUMBER_OF_CELL_TYPES > 255);

  return (fail1 || fail2);
}
