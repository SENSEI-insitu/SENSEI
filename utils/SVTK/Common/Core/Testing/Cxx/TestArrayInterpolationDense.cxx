/*=========================================================================

  Program:   Visualization Toolkit
  Module:    ArrayInterpolationDense.cxx

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

#include <svtkArrayInterpolate.h>
#include <svtkDenseArray.h>
#include <svtkSmartPointer.h>

#include <iostream>
#include <stdexcept>

void test_expression(const bool expression, const std::string& message)
{
  if (!expression)
    throw std::runtime_error(message);
}

int TestArrayInterpolationDense(int svtkNotUsed(argc), char* svtkNotUsed(argv)[])
{
  try
  {
    svtkSmartPointer<svtkDenseArray<double> > a = svtkSmartPointer<svtkDenseArray<double> >::New();
    a->Resize(4);
    a->SetValue(0, 0);
    a->SetValue(1, 1);
    a->SetValue(2, 2);
    a->SetValue(3, 3);

    svtkSmartPointer<svtkDenseArray<double> > b = svtkSmartPointer<svtkDenseArray<double> >::New();
    b->Resize(svtkArrayExtents(2));

    svtkInterpolate(a.GetPointer(),
      svtkArrayExtentsList(
        svtkArrayExtents(svtkArrayRange(0, 1)), svtkArrayExtents(svtkArrayRange(1, 2))),
      svtkArrayWeights(0.5, 0.5), svtkArrayExtents(svtkArrayRange(0, 1)), b.GetPointer());
    svtkInterpolate(a.GetPointer(),
      svtkArrayExtentsList(
        svtkArrayExtents(svtkArrayRange(2, 3)), svtkArrayExtents(svtkArrayRange(3, 4))),
      svtkArrayWeights(0.5, 0.5), svtkArrayExtents(svtkArrayRange(1, 2)), b.GetPointer());

    test_expression(b->GetValue(0) == 0.5, "expected 0.5");
    test_expression(b->GetValue(1) == 2.5, "expected 2.5");

    svtkSmartPointer<svtkDenseArray<double> > c = svtkSmartPointer<svtkDenseArray<double> >::New();
    c->Resize(4, 2);
    c->SetValue(0, 0, 0);
    c->SetValue(0, 1, 1);
    c->SetValue(1, 0, 2);
    c->SetValue(1, 1, 3);
    c->SetValue(2, 0, 4);
    c->SetValue(2, 1, 5);
    c->SetValue(3, 0, 6);
    c->SetValue(3, 1, 7);

    svtkSmartPointer<svtkDenseArray<double> > d = svtkSmartPointer<svtkDenseArray<double> >::New();
    d->Resize(svtkArrayExtents(2, 2));

    svtkInterpolate(c.GetPointer(),
      svtkArrayExtentsList(svtkArrayExtents(svtkArrayRange(0, 1), svtkArrayRange(0, 2)),
        svtkArrayExtents(svtkArrayRange(1, 2), svtkArrayRange(0, 2))),
      svtkArrayWeights(0.5, 0.5), svtkArrayExtents(svtkArrayRange(0, 1), svtkArrayRange(0, 2)),
      d.GetPointer());
    svtkInterpolate(c.GetPointer(),
      svtkArrayExtentsList(svtkArrayExtents(svtkArrayRange(2, 3), svtkArrayRange(0, 2)),
        svtkArrayExtents(svtkArrayRange(3, 4), svtkArrayRange(0, 2))),
      svtkArrayWeights(0.5, 0.5), svtkArrayExtents(svtkArrayRange(1, 2), svtkArrayRange(0, 2)),
      d.GetPointer());

    test_expression(d->GetValue(0, 0) == 1, "expected 1");
    test_expression(d->GetValue(0, 1) == 2, "expected 2");
    test_expression(d->GetValue(1, 0) == 5, "expected 5");
    test_expression(d->GetValue(1, 1) == 6, "expected 6");

    return 0;
  }
  catch (std::exception& e)
  {
    cerr << e.what() << endl;
    return 1;
  }
}
