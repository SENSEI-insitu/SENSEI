/*=========================================================================

  Program:   Visualization Toolkit
  Module:    ArrayBool.cxx

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

#include <svtkDenseArray.h>
#include <svtkSmartPointer.h>
#include <svtkSparseArray.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

#define test_expression(expression)                                                                \
  {                                                                                                \
    if (!(expression))                                                                             \
    {                                                                                              \
      std::ostringstream buffer;                                                                   \
      buffer << "Expression failed at line " << __LINE__ << ": " << #expression;                   \
      throw std::runtime_error(buffer.str());                                                      \
    }                                                                                              \
  }

int TestArrayBool(int svtkNotUsed(argc), char* svtkNotUsed(argv)[])
{
  try
  {
    // Confirm that we can work with dense arrays of bool values
    svtkSmartPointer<svtkDenseArray<char> > dense = svtkSmartPointer<svtkDenseArray<char> >::New();
    svtkDenseArray<char>& dense_ref = *dense;
    dense->Resize(2, 2);
    dense->Fill(0);

    test_expression(dense->GetValue(1, 1) == 0);
    dense->SetValue(1, 1, 1);
    test_expression(dense->GetValue(1, 1) == 1);

    test_expression(dense->GetValue(0, 1) == 0);
    test_expression(dense_ref[svtkArrayCoordinates(0, 1)] == 0);
    dense_ref[svtkArrayCoordinates(0, 1)] = 1;
    test_expression(dense_ref[svtkArrayCoordinates(0, 1)] == 1);
    test_expression(dense->GetValue(0, 1) == 1);

    // Confirm that we can work with sparse arrays of bool values
    svtkSmartPointer<svtkSparseArray<char> > sparse = svtkSmartPointer<svtkSparseArray<char> >::New();
    sparse->Resize(2, 2);

    test_expression(sparse->GetValue(1, 1) == 0);
    sparse->SetValue(1, 1, 1);
    test_expression(sparse->GetValue(1, 1) == 1);

    return 0;
  }
  catch (std::exception& e)
  {
    cerr << e.what() << endl;
    return 1;
  }
}
