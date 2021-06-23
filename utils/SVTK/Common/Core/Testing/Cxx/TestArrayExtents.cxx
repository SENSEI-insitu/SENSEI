/*=========================================================================

  Program:   Visualization Toolkit
  Module:    ArrayExtents.cxx

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

#include <svtkArrayCoordinates.h>
#include <svtkArrayExtents.h>

#include "svtkSetGet.h"
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

int TestArrayExtents(int svtkNotUsed(argc), char* svtkNotUsed(argv)[])
{
  try
  {
    svtkArrayExtents slice(svtkArrayRange(2, 4), svtkArrayRange(6, 9));

    test_expression(slice.GetDimensions() == 2);
    test_expression(slice[0].GetSize() == 2);
    test_expression(slice[1].GetSize() == 3);
    test_expression(slice.GetSize() == 6);

    svtkArrayCoordinates coordinates;
    for (svtkArrayExtents::SizeT n = 0; n != slice.GetSize(); ++n)
    {
      slice.GetLeftToRightCoordinatesN(n, coordinates);
      cerr << coordinates << endl;
    }

    slice.GetLeftToRightCoordinatesN(0, coordinates);
    test_expression(coordinates == svtkArrayCoordinates(2, 6));
    slice.GetLeftToRightCoordinatesN(1, coordinates);
    test_expression(coordinates == svtkArrayCoordinates(3, 6));
    slice.GetLeftToRightCoordinatesN(2, coordinates);
    test_expression(coordinates == svtkArrayCoordinates(2, 7));
    slice.GetLeftToRightCoordinatesN(3, coordinates);
    test_expression(coordinates == svtkArrayCoordinates(3, 7));
    slice.GetLeftToRightCoordinatesN(4, coordinates);
    test_expression(coordinates == svtkArrayCoordinates(2, 8));
    slice.GetLeftToRightCoordinatesN(5, coordinates);
    test_expression(coordinates == svtkArrayCoordinates(3, 8));

    test_expression(slice.Contains(svtkArrayCoordinates(3, 7)));
    test_expression(!slice.Contains(svtkArrayCoordinates(1, 7)));

    return 0;
  }
  catch (std::exception& e)
  {
    cerr << e.what() << endl;
    return 1;
  }
}
