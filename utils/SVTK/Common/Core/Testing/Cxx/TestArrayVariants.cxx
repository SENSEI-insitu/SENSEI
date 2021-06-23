/*=========================================================================

  Program:   Visualization Toolkit
  Module:    ArrayVariants.cxx

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

int TestArrayVariants(int svtkNotUsed(argc), char* svtkNotUsed(argv)[])
{
  try
  {
    // Exercise the API that gets/sets variants ...
    svtkSmartPointer<svtkDenseArray<double> > concrete =
      svtkSmartPointer<svtkDenseArray<double> >::New();
    concrete->Resize(3, 2);
    svtkTypedArray<double>* const typed = concrete;
    svtkArray* const abstract = concrete;

    abstract->SetVariantValue(0, 0, 1.0);
    abstract->SetVariantValue(svtkArrayCoordinates(0, 1), 2.0);
    typed->SetVariantValue(1, 0, 3.0);
    typed->SetVariantValue(svtkArrayCoordinates(1, 1), 4.0);
    concrete->SetVariantValue(2, 0, 5.0);
    concrete->SetVariantValue(svtkArrayCoordinates(2, 1), 6.0);

    test_expression(abstract->GetVariantValue(0, 0) == 1.0);
    test_expression(abstract->GetVariantValue(svtkArrayCoordinates(0, 1)) == 2.0);
    test_expression(typed->GetVariantValue(1, 0) == 3.0);
    test_expression(typed->GetVariantValue(svtkArrayCoordinates(1, 1)) == 4.0);
    test_expression(concrete->GetVariantValue(2, 0) == 5.0);
    test_expression(concrete->GetVariantValue(svtkArrayCoordinates(2, 1)) == 6.0);

    abstract->SetVariantValueN(0, 7.0);
    test_expression(abstract->GetVariantValueN(0) == 7.0);
    typed->SetVariantValueN(0, 8.0);
    test_expression(typed->GetVariantValueN(0) == 8.0);
    concrete->SetVariantValueN(0, 9.0);
    test_expression(concrete->GetVariantValueN(0) == 9.0);

    return 0;
  }
  catch (std::exception& e)
  {
    cerr << e.what() << endl;
    return 1;
  }
}
