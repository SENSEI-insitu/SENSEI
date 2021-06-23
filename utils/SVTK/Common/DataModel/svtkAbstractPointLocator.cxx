/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAbstractPointLocator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkAbstractPointLocator.h"

#include "svtkDataSet.h"
#include "svtkIdList.h"

//-----------------------------------------------------------------------------
svtkAbstractPointLocator::svtkAbstractPointLocator()
{
  for (int i = 0; i < 6; i++)
  {
    this->Bounds[i] = 0;
  }
  this->NumberOfBuckets = 0;
}

//-----------------------------------------------------------------------------
svtkAbstractPointLocator::~svtkAbstractPointLocator() = default;

//-----------------------------------------------------------------------------
// Given a position x-y-z, return the id of the point closest to it.
svtkIdType svtkAbstractPointLocator::FindClosestPoint(double x, double y, double z)
{
  double xyz[3];

  xyz[0] = x;
  xyz[1] = y;
  xyz[2] = z;
  return this->FindClosestPoint(xyz);
}

//-----------------------------------------------------------------------------
void svtkAbstractPointLocator::FindClosestNPoints(
  int N, double x, double y, double z, svtkIdList* result)
{
  double p[3];
  p[0] = x;
  p[1] = y;
  p[2] = z;
  this->FindClosestNPoints(N, p, result);
}

//-----------------------------------------------------------------------------
void svtkAbstractPointLocator::FindPointsWithinRadius(
  double R, double x, double y, double z, svtkIdList* result)
{
  double p[3];
  p[0] = x;
  p[1] = y;
  p[2] = z;
  this->FindPointsWithinRadius(R, p, result);
}

//-----------------------------------------------------------------------------
void svtkAbstractPointLocator::GetBounds(double* bnds)
{
  for (int i = 0; i < 6; i++)
  {
    bnds[i] = this->Bounds[i];
  }
}

//-----------------------------------------------------------------------------
void svtkAbstractPointLocator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  for (int i = 0; i < 6; i++)
  {
    os << indent << "Bounds[" << i << "]: " << this->Bounds[i] << "\n";
  }

  os << indent << "Number of Buckets: " << this->NumberOfBuckets << "\n";
}
