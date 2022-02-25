/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestCompositeDataSets.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkCompositeDataIterator.h"
#include "svtkDataObjectTree.h"
#include "svtkDataObjectTreeIterator.h"
#include "svtkInformation.h"
#include "svtkMultiBlockDataSet.h"
#include "svtkNew.h"
#include "svtkSmartPointer.h"
#include "svtkStdString.h"
#include "svtkUniformGrid.h"
#include "svtkUniformGridAMR.h"

#include <iostream>
#include <vector>

//------------------------------------------------------------------------------
int TestCompositeDataSets(int, char*[])
{
  int errors = 0;

  return (errors);
}
