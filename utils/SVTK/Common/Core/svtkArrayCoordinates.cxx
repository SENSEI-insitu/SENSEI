/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayCoordinates.cxx

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

#include "svtkArrayCoordinates.h"

svtkArrayCoordinates::svtkArrayCoordinates() = default;

svtkArrayCoordinates::svtkArrayCoordinates(CoordinateT i)
  : Storage(1)
{
  this->Storage[0] = i;
}

svtkArrayCoordinates::svtkArrayCoordinates(CoordinateT i, CoordinateT j)
  : Storage(2)
{
  this->Storage[0] = i;
  this->Storage[1] = j;
}

svtkArrayCoordinates::svtkArrayCoordinates(CoordinateT i, CoordinateT j, CoordinateT k)
  : Storage(3)
{
  this->Storage[0] = i;
  this->Storage[1] = j;
  this->Storage[2] = k;
}

svtkArrayCoordinates::DimensionT svtkArrayCoordinates::GetDimensions() const
{
  return static_cast<svtkArrayCoordinates::DimensionT>(this->Storage.size());
}

void svtkArrayCoordinates::SetDimensions(DimensionT dimensions)
{
  this->Storage.assign(dimensions, 0);
}

svtkArrayCoordinates::CoordinateT& svtkArrayCoordinates::operator[](DimensionT i)
{
  return this->Storage[i];
}

const svtkArrayCoordinates::CoordinateT& svtkArrayCoordinates::operator[](DimensionT i) const
{
  return this->Storage[i];
}

svtkArrayCoordinates::CoordinateT svtkArrayCoordinates::GetCoordinate(DimensionT i) const
{
  return this->Storage[i];
}

void svtkArrayCoordinates::SetCoordinate(DimensionT i, const CoordinateT& coordinate)
{
  this->Storage[i] = coordinate;
}

bool svtkArrayCoordinates::operator==(const svtkArrayCoordinates& rhs) const
{
  return this->Storage == rhs.Storage;
}

bool svtkArrayCoordinates::operator!=(const svtkArrayCoordinates& rhs) const
{
  return !(*this == rhs);
}

ostream& operator<<(ostream& stream, const svtkArrayCoordinates& rhs)
{
  for (svtkArrayCoordinates::DimensionT i = 0; i != rhs.GetDimensions(); ++i)
  {
    if (i)
      stream << ",";
    stream << rhs[i];
  }

  return stream;
}
