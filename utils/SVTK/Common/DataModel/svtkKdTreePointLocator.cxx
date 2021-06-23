/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkKdTreePointLocator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkKdTreePointLocator.h"

#include "svtkKdTree.h"
#include "svtkObjectFactory.h"
#include "svtkPointSet.h"

svtkStandardNewMacro(svtkKdTreePointLocator);

svtkKdTreePointLocator::svtkKdTreePointLocator()
{
  this->KdTree = nullptr;
}

svtkKdTreePointLocator::~svtkKdTreePointLocator()
{
  if (this->KdTree)
  {
    this->KdTree->Delete();
  }
}

svtkIdType svtkKdTreePointLocator::FindClosestPoint(const double x[3])
{
  this->BuildLocator();
  double dist2;

  return this->KdTree->FindClosestPoint(x[0], x[1], x[2], dist2);
}

svtkIdType svtkKdTreePointLocator::FindClosestPointWithinRadius(
  double radius, const double x[3], double& dist2)
{
  this->BuildLocator();
  return this->KdTree->FindClosestPointWithinRadius(radius, x, dist2);
}

void svtkKdTreePointLocator::FindClosestNPoints(int N, const double x[3], svtkIdList* result)
{
  this->BuildLocator();
  this->KdTree->FindClosestNPoints(N, x, result);
}

void svtkKdTreePointLocator::FindPointsWithinRadius(double R, const double x[3], svtkIdList* result)
{
  this->BuildLocator();
  this->KdTree->FindPointsWithinRadius(R, x, result);
}

void svtkKdTreePointLocator::FreeSearchStructure()
{
  if (this->KdTree)
  {
    this->KdTree->Delete();
    this->KdTree = nullptr;
  }
}

void svtkKdTreePointLocator::BuildLocator()
{
  if (!this->KdTree)
  {
    svtkPointSet* pointSet = svtkPointSet::SafeDownCast(this->GetDataSet());
    if (!pointSet)
    {
      svtkErrorMacro("svtkKdTreePointLocator requires a PointSet to build locator.");
      return;
    }
    this->KdTree = svtkKdTree::New();
    this->KdTree->BuildLocatorFromPoints(pointSet);
    this->KdTree->GetBounds(this->Bounds);
    this->Modified();
  }
}

void svtkKdTreePointLocator::GenerateRepresentation(int level, svtkPolyData* pd)
{
  this->BuildLocator();
  this->KdTree->GenerateRepresentation(level, pd);
}

void svtkKdTreePointLocator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "KdTree " << this->KdTree << "\n";
}
