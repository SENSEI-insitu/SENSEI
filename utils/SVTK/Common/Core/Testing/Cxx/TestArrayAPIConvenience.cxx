/*=========================================================================

  Program:   Visualization Toolkit
  Module:    ArrayAPIConvenience.cxx

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

int TestArrayAPIConvenience(int svtkNotUsed(argc), char* svtkNotUsed(argv)[])
{
  try
  {
    svtkSmartPointer<svtkDenseArray<double> > a = svtkSmartPointer<svtkDenseArray<double> >::New();
    svtkSmartPointer<svtkDenseArray<double> > b = svtkSmartPointer<svtkDenseArray<double> >::New();

    a->Resize(5);
    b->Resize(svtkArrayExtents(5));
    test_expression(a->GetExtents() == b->GetExtents());

    a->SetValue(2, 3);
    b->SetValue(svtkArrayCoordinates(2), 3);
    test_expression(a->GetValue(2) == b->GetValue(svtkArrayCoordinates(2)));

    a->Resize(5, 6);
    b->Resize(svtkArrayExtents(5, 6));
    test_expression(a->GetExtents() == b->GetExtents());

    a->SetValue(2, 3, 4);
    b->SetValue(svtkArrayCoordinates(2, 3), 4);
    test_expression(a->GetValue(2, 3) == b->GetValue(svtkArrayCoordinates(2, 3)));

    a->Resize(5, 6, 7);
    b->Resize(svtkArrayExtents(5, 6, 7));
    test_expression(a->GetExtents() == b->GetExtents());

    a->SetValue(2, 3, 4, 5);
    b->SetValue(svtkArrayCoordinates(2, 3, 4), 5);
    test_expression(a->GetValue(2, 3, 4) == b->GetValue(svtkArrayCoordinates(2, 3, 4)));

    return 0;
  }
  catch (std::exception& e)
  {
    cerr << e.what() << endl;
    return 1;
  }
}
