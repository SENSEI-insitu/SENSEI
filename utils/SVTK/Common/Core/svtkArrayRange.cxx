/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayRange.cxx

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

#include "svtkArrayRange.h"

#include <algorithm> // for std::max()

svtkArrayRange::svtkArrayRange()
  : Begin(0)
  , End(0)
{
}

svtkArrayRange::svtkArrayRange(CoordinateT begin, CoordinateT end)
  : Begin(begin)
  , End(std::max(begin, end))
{
}

svtkArrayRange::CoordinateT svtkArrayRange::GetBegin() const
{
  return this->Begin;
}

svtkArrayRange::CoordinateT svtkArrayRange::GetEnd() const
{
  return this->End;
}

svtkArrayRange::CoordinateT svtkArrayRange::GetSize() const
{
  return this->End - this->Begin;
}

bool svtkArrayRange::Contains(const svtkArrayRange& range) const
{
  return this->Begin <= range.Begin && range.End <= this->End;
}

bool svtkArrayRange::Contains(const CoordinateT coordinate) const
{
  return this->Begin <= coordinate && coordinate < this->End;
}

bool operator==(const svtkArrayRange& lhs, const svtkArrayRange& rhs)
{
  return lhs.Begin == rhs.Begin && lhs.End == rhs.End;
}

bool operator!=(const svtkArrayRange& lhs, const svtkArrayRange& rhs)
{
  return !(lhs == rhs);
}

ostream& operator<<(ostream& stream, const svtkArrayRange& rhs)
{
  stream << "[" << rhs.Begin << ", " << rhs.End << ")";
  return stream;
}
