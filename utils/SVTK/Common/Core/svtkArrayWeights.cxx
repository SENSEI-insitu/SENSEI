/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayWeights.cxx

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

#include "svtkArrayWeights.h"
#include <vector>

class svtkArrayWeightsStorage
{
public:
  svtkArrayWeightsStorage(size_t size)
    : Storage(size)
  {
  }
  std::vector<double> Storage;
};

svtkArrayWeights::svtkArrayWeights()
{
  this->Storage = new svtkArrayWeightsStorage(0);
}

svtkArrayWeights::svtkArrayWeights(double i)
{
  this->Storage = new svtkArrayWeightsStorage(1);
  this->Storage->Storage[0] = i;
}

svtkArrayWeights::svtkArrayWeights(double i, double j)
{
  this->Storage = new svtkArrayWeightsStorage(2);
  this->Storage->Storage[0] = i;
  this->Storage->Storage[1] = j;
}

svtkArrayWeights::svtkArrayWeights(double i, double j, double k)
{
  this->Storage = new svtkArrayWeightsStorage(3);
  this->Storage->Storage[0] = i;
  this->Storage->Storage[1] = j;
  this->Storage->Storage[2] = k;
}

svtkArrayWeights::svtkArrayWeights(double i, double j, double k, double l)
{
  this->Storage = new svtkArrayWeightsStorage(4);
  this->Storage->Storage[0] = i;
  this->Storage->Storage[1] = j;
  this->Storage->Storage[2] = k;
  this->Storage->Storage[3] = l;
}

svtkArrayWeights::svtkArrayWeights(const svtkArrayWeights& other)
{
  this->Storage = new svtkArrayWeightsStorage(*other.Storage);
}

// ----------------------------------------------------------------------------
svtkArrayWeights::~svtkArrayWeights()
{
  delete this->Storage;
}

// ----------------------------------------------------------------------------
svtkIdType svtkArrayWeights::GetCount() const
{
  return static_cast<svtkIdType>(this->Storage->Storage.size());
}

void svtkArrayWeights::SetCount(svtkIdType count)
{
  this->Storage->Storage.assign(static_cast<size_t>(count), 0.0);
}

double& svtkArrayWeights::operator[](svtkIdType i)
{
  return this->Storage->Storage[static_cast<size_t>(i)];
}

const double& svtkArrayWeights::operator[](svtkIdType i) const
{
  return this->Storage->Storage[static_cast<size_t>(i)];
}

svtkArrayWeights& svtkArrayWeights::operator=(const svtkArrayWeights& other)
{
  *this->Storage = *other.Storage;
  return *this;
}
