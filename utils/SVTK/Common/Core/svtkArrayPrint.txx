/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayPrint.txx

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

#ifndef svtkArrayPrint_txx
#define svtkArrayPrint_txx

#include "svtkArrayCoordinates.h"
#include <algorithm>
#include <iterator>

template <typename T>
void svtkPrintCoordinateFormat(ostream& stream, svtkTypedArray<T>* array)
{
  if (!array)
  {
    svtkGenericWarningMacro(<< "svtkPrintCoordinateFormat() requires a non-nullptr array as input.");
    return;
  }

  const svtkArrayExtents extents = array->GetExtents();
  const svtkIdType dimensions = array->GetDimensions();
  const svtkIdType non_null_size = array->GetNonNullSize();

  for (svtkIdType i = 0; i != dimensions; ++i)
    stream << extents[i] << " ";
  stream << array->GetNonNullSize() << "\n";

  svtkArrayCoordinates coordinates;
  for (svtkIdType n = 0; n != non_null_size; ++n)
  {
    array->GetCoordinatesN(n, coordinates);
    for (svtkIdType i = 0; i != dimensions; ++i)
      stream << coordinates[i] << " ";
    stream << array->GetValueN(n) << "\n";
  }
}

template <typename T>
void svtkPrintMatrixFormat(ostream& stream, svtkTypedArray<T>* matrix)
{
  if (!matrix)
  {
    svtkGenericWarningMacro(<< "svtkPrintMatrixFormat() requires a non-nullptr array as input.");
    return;
  }

  if (matrix->GetDimensions() != 2)
  {
    svtkGenericWarningMacro(<< "svtkPrintMatrixFormat() requires a matrix (2-way array) as input.");
    return;
  }

  const svtkArrayRange rows = matrix->GetExtent(0);
  const svtkArrayRange columns = matrix->GetExtent(1);

  for (svtkIdType row = rows.GetBegin(); row != rows.GetEnd(); ++row)
  {
    for (svtkIdType column = columns.GetBegin(); column != columns.GetEnd(); ++column)
    {
      stream << matrix->GetValue(svtkArrayCoordinates(row, column)) << " ";
    }
    stream << "\n";
  }
}

template <typename T>
void svtkPrintVectorFormat(ostream& stream, svtkTypedArray<T>* vector)
{
  if (!vector)
  {
    svtkGenericWarningMacro(<< "svtkPrintVectorFormat() requires a non-nullptr array as input.");
    return;
  }

  if (vector->GetDimensions() != 1)
  {
    svtkGenericWarningMacro(<< "svtkPrintVectorFormat() requires a vector (1-way array) as input.");
    return;
  }

  const svtkArrayRange rows = vector->GetExtent(0);

  for (svtkIdType row = rows.GetBegin(); row != rows.GetEnd(); ++row)
  {
    stream << vector->GetValue(svtkArrayCoordinates(row)) << "\n";
  }
}

#endif
