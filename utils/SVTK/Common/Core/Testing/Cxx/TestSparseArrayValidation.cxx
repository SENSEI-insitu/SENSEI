/*=========================================================================

  Program:   Visualization Toolkit
  Module:    SparseArrayValidation.cxx

-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <svtkArrayPrint.h>
#include <svtkSmartPointer.h>
#include <svtkSparseArray.h>
#include <svtkTestErrorObserver.h>

#include <iostream>
#include <stdexcept>

#define test_expression(expression)                                                                \
  {                                                                                                \
    if (!(expression))                                                                             \
      throw std::runtime_error("Expression failed: " #expression);                                 \
  }

int TestSparseArrayValidation(int svtkNotUsed(argc), char* svtkNotUsed(argv)[])
{
  try
  {
    // Create an array ...
    svtkSmartPointer<svtkSparseArray<double> > array =
      svtkSmartPointer<svtkSparseArray<double> >::New();
    test_expression(array->Validate());

    array->Resize(svtkArrayExtents::Uniform(2, 3));
    test_expression(array->Validate());

    array->Clear();
    array->AddValue(0, 0, 1);
    array->AddValue(1, 2, 2);
    array->AddValue(0, 1, 3);
    test_expression(array->Validate());

    svtkSmartPointer<svtkTest::ErrorObserver> errorObserver =
      svtkSmartPointer<svtkTest::ErrorObserver>::New();
    array->AddObserver(svtkCommand::ErrorEvent, errorObserver);
    array->Clear();
    array->AddValue(0, 0, 1);
    array->AddValue(1, 2, 2);
    array->AddValue(0, 0, 4);
    test_expression(!array->Validate());
    int status = 0;
    status += errorObserver->CheckErrorMessage("Array contains 1 duplicate coordinates");

    array->Clear();
    array->AddValue(0, 0, 1);
    array->AddValue(3, 3, 2);
    test_expression(!array->Validate());

    return 0;
  }
  catch (std::exception& e)
  {
    cerr << e.what() << endl;
    return 1;
  }
}
