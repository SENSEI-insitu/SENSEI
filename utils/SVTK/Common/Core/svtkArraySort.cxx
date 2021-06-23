/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArraySort.cxx

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

#include "svtkArraySort.h"

svtkArraySort::svtkArraySort() = default;

svtkArraySort::svtkArraySort(DimensionT i)
  : Storage(1)
{
  this->Storage[0] = i;
}

svtkArraySort::svtkArraySort(DimensionT i, DimensionT j)
  : Storage(2)
{
  this->Storage[0] = i;
  this->Storage[1] = j;
}

svtkArraySort::svtkArraySort(DimensionT i, DimensionT j, DimensionT k)
  : Storage(3)
{
  this->Storage[0] = i;
  this->Storage[1] = j;
  this->Storage[2] = k;
}

svtkArraySort::DimensionT svtkArraySort::GetDimensions() const
{
  return static_cast<svtkArraySort::DimensionT>(this->Storage.size());
}

void svtkArraySort::SetDimensions(DimensionT dimensions)
{
  this->Storage.assign(dimensions, 0);
}

svtkArraySort::DimensionT& svtkArraySort::operator[](DimensionT i)
{
  return this->Storage[i];
}

const svtkArraySort::DimensionT& svtkArraySort::operator[](DimensionT i) const
{
  return this->Storage[i];
}

bool svtkArraySort::operator==(const svtkArraySort& rhs) const
{
  return this->Storage == rhs.Storage;
}

bool svtkArraySort::operator!=(const svtkArraySort& rhs) const
{
  return !(*this == rhs);
}

ostream& operator<<(ostream& stream, const svtkArraySort& rhs)
{
  for (svtkArraySort::DimensionT i = 0; i != rhs.GetDimensions(); ++i)
  {
    if (i)
      stream << ",";
    stream << rhs[i];
  }

  return stream;
}
