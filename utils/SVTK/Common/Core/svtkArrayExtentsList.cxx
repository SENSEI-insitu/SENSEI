/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayExtentsList.cxx

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

#include "svtkArrayExtentsList.h"

svtkArrayExtentsList::svtkArrayExtentsList() = default;

svtkArrayExtentsList::svtkArrayExtentsList(const svtkArrayExtents& i)
  : Storage(1)
{
  this->Storage[0] = i;
}

svtkArrayExtentsList::svtkArrayExtentsList(const svtkArrayExtents& i, const svtkArrayExtents& j)
  : Storage(2)
{
  this->Storage[0] = i;
  this->Storage[1] = j;
}

svtkArrayExtentsList::svtkArrayExtentsList(
  const svtkArrayExtents& i, const svtkArrayExtents& j, const svtkArrayExtents& k)
  : Storage(3)
{
  this->Storage[0] = i;
  this->Storage[1] = j;
  this->Storage[2] = k;
}

svtkArrayExtentsList::svtkArrayExtentsList(const svtkArrayExtents& i, const svtkArrayExtents& j,
  const svtkArrayExtents& k, const svtkArrayExtents& l)
  : Storage(4)
{
  this->Storage[0] = i;
  this->Storage[1] = j;
  this->Storage[2] = k;
  this->Storage[3] = l;
}

svtkIdType svtkArrayExtentsList::GetCount() const
{
  return static_cast<svtkIdType>(this->Storage.size());
}

void svtkArrayExtentsList::SetCount(svtkIdType count)
{
  this->Storage.assign(count, svtkArrayExtents());
}

svtkArrayExtents& svtkArrayExtentsList::operator[](svtkIdType i)
{
  return this->Storage[i];
}

const svtkArrayExtents& svtkArrayExtentsList::operator[](svtkIdType i) const
{
  return this->Storage[i];
}
