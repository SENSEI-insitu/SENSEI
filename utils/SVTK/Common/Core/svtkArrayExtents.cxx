/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayExtents.cxx

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

#include "svtkArrayExtents.h"
#include "svtkArrayCoordinates.h"

#include <functional>
#include <numeric>

svtkArrayExtents::svtkArrayExtents() = default;

svtkArrayExtents::svtkArrayExtents(const CoordinateT i)
  : Storage(1)
{
  this->Storage[0] = svtkArrayRange(0, i);
}

svtkArrayExtents::svtkArrayExtents(const svtkArrayRange& i)
  : Storage(1)
{
  this->Storage[0] = i;
}

svtkArrayExtents::svtkArrayExtents(const CoordinateT i, const CoordinateT j)
  : Storage(2)
{
  this->Storage[0] = svtkArrayRange(0, i);
  this->Storage[1] = svtkArrayRange(0, j);
}

svtkArrayExtents::svtkArrayExtents(const svtkArrayRange& i, const svtkArrayRange& j)
  : Storage(2)
{
  this->Storage[0] = i;
  this->Storage[1] = j;
}

svtkArrayExtents::svtkArrayExtents(const CoordinateT i, const CoordinateT j, const CoordinateT k)
  : Storage(3)
{
  this->Storage[0] = svtkArrayRange(0, i);
  this->Storage[1] = svtkArrayRange(0, j);
  this->Storage[2] = svtkArrayRange(0, k);
}

svtkArrayExtents::svtkArrayExtents(
  const svtkArrayRange& i, const svtkArrayRange& j, const svtkArrayRange& k)
  : Storage(3)
{
  this->Storage[0] = i;
  this->Storage[1] = j;
  this->Storage[2] = k;
}

svtkArrayExtents svtkArrayExtents::Uniform(DimensionT n, CoordinateT m)
{
  svtkArrayExtents result;
  // IA64 HP-UX doesn't seem to have the vector<T> vector1(n, value)
  // overload nor the assign(n, value) method, so we use the single
  // argument constructor and initialize the values manually.
  // result.Storage = std::vector<svtkIdType>(n, m);

  result.Storage = std::vector<svtkArrayRange>(n);
  for (DimensionT i = 0; i < n; i++)
  {
    result.Storage[i] = svtkArrayRange(0, m);
  }
  return result;
}

void svtkArrayExtents::Append(const svtkArrayRange& extent)
{
  this->Storage.push_back(extent);
}

svtkArrayExtents::DimensionT svtkArrayExtents::GetDimensions() const
{
  return static_cast<svtkArrayExtents::DimensionT>(this->Storage.size());
}

svtkTypeUInt64 svtkArrayExtents::GetSize() const
{
  if (this->Storage.empty())
    return 0;

  svtkTypeUInt64 size = 1;
  for (size_t i = 0; i != this->Storage.size(); ++i)
    size *= this->Storage[i].GetSize();

  return size;
}

void svtkArrayExtents::SetDimensions(DimensionT dimensions)
{
  this->Storage.assign(dimensions, svtkArrayRange());
}

svtkArrayRange& svtkArrayExtents::operator[](DimensionT i)
{
  return this->Storage[i];
}

const svtkArrayRange& svtkArrayExtents::operator[](DimensionT i) const
{
  return this->Storage[i];
}

svtkArrayRange svtkArrayExtents::GetExtent(DimensionT i) const
{
  return this->Storage[i];
}

void svtkArrayExtents::SetExtent(DimensionT i, const svtkArrayRange& extent)
{
  this->Storage[i] = extent;
}

bool svtkArrayExtents::operator==(const svtkArrayExtents& rhs) const
{
  return this->Storage == rhs.Storage;
}

bool svtkArrayExtents::operator!=(const svtkArrayExtents& rhs) const
{
  return !(*this == rhs);
}

bool svtkArrayExtents::ZeroBased() const
{
  for (DimensionT i = 0; i != this->GetDimensions(); ++i)
  {
    if (this->Storage[i].GetBegin() != 0)
      return false;
  }

  return true;
}

bool svtkArrayExtents::SameShape(const svtkArrayExtents& rhs) const
{
  if (this->GetDimensions() != rhs.GetDimensions())
    return false;

  for (DimensionT i = 0; i != this->GetDimensions(); ++i)
  {
    if (this->Storage[i].GetSize() != rhs.Storage[i].GetSize())
      return false;
  }

  return true;
}

void svtkArrayExtents::GetLeftToRightCoordinatesN(SizeT n, svtkArrayCoordinates& coordinates) const
{
  coordinates.SetDimensions(this->GetDimensions());

  svtkIdType divisor = 1;
  for (svtkIdType i = 0; i < this->GetDimensions(); ++i)
  {
    coordinates[i] = ((n / divisor) % this->Storage[i].GetSize()) + this->Storage[i].GetBegin();
    divisor *= this->Storage[i].GetSize();
  }
}

void svtkArrayExtents::GetRightToLeftCoordinatesN(SizeT n, svtkArrayCoordinates& coordinates) const
{
  coordinates.SetDimensions(this->GetDimensions());

  svtkIdType divisor = 1;
  for (svtkIdType i = this->GetDimensions() - 1; i >= 0; --i)
  {
    coordinates[i] = ((n / divisor) % this->Storage[i].GetSize()) + this->Storage[i].GetBegin();
    divisor *= this->Storage[i].GetSize();
  }
}

bool svtkArrayExtents::Contains(const svtkArrayExtents& other) const
{
  if (this->GetDimensions() != other.GetDimensions())
    return false;

  for (DimensionT i = 0; i != this->GetDimensions(); ++i)
  {
    if (!this->Storage[i].Contains(other[i]))
      return false;
  }

  return true;
}

bool svtkArrayExtents::Contains(const svtkArrayCoordinates& coordinates) const
{
  if (coordinates.GetDimensions() != this->GetDimensions())
    return false;

  for (DimensionT i = 0; i != this->GetDimensions(); ++i)
  {
    if (!this->Storage[i].Contains(coordinates[i]))
      return false;
  }

  return true;
}

ostream& operator<<(ostream& stream, const svtkArrayExtents& rhs)
{
  for (size_t i = 0; i != rhs.Storage.size(); ++i)
  {
    if (i)
      stream << "x";
    stream << "[" << rhs.Storage[i].GetBegin() << "," << rhs.Storage[i].GetEnd() << ")";
  }

  return stream;
}
