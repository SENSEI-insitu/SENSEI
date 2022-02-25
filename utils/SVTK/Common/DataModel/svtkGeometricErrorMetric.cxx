/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGeometricErrorMetric.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkGeometricErrorMetric.h"

#include "svtkGenericAdaptorCell.h"
#include "svtkGenericAttribute.h"
#include "svtkGenericAttributeCollection.h"
#include "svtkGenericDataSet.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include <cassert>

svtkStandardNewMacro(svtkGeometricErrorMetric);

//-----------------------------------------------------------------------------
svtkGeometricErrorMetric::svtkGeometricErrorMetric()
{
  this->AbsoluteGeometricTolerance = 1.0; // arbitrary positive value
  this->Relative = 0;                     // GetError() will return the square absolute error.
  this->SmallestSize = 1;
}

//-----------------------------------------------------------------------------
svtkGeometricErrorMetric::~svtkGeometricErrorMetric() = default;

//-----------------------------------------------------------------------------
// Description :
// Set the geometric accuracy with an absolute value.
// Subdivision will be required if the square distance is greater than
// value. For instance 0.01 will give better result than 0.1.
// \pre positive_value: value>0
void svtkGeometricErrorMetric::SetAbsoluteGeometricTolerance(double value)
{
  assert("pre: positive_value" && value > 0);
  this->Relative = 0;
  if (this->AbsoluteGeometricTolerance != value)
  {
    this->AbsoluteGeometricTolerance = value;
    this->Modified();
  }
}

//-----------------------------------------------------------------------------
// Description :
// Set the geometric accuracy with a value relative to the bounding box of
// the dataset. Internally compute the absolute tolerance.
// For instance 0.01 will give better result than 0.1.
// \pre valid_range_value: value>0 && value<1
// \pre ds_exists: ds!=0
void svtkGeometricErrorMetric::SetRelativeGeometricTolerance(double value, svtkGenericDataSet* ds)
{
  assert("pre: valid_range_value" && value > 0 && value < 1);
  assert("pre: ds_exists" && ds != nullptr);

  double bounds[6];
  ds->GetBounds(bounds);
  double smallest;
  double length;
  smallest = bounds[1] - bounds[0];
  length = bounds[3] - bounds[2];
  if (length < smallest || smallest == 0.0)
  {
    smallest = length;
  }
  length = bounds[5] - bounds[4];
  if (length < smallest || smallest == 0.0)
  {
    smallest = length;
  }
  length = ds->GetLength();
  if (length < smallest || smallest == 0.0)
  {
    smallest = length;
  }
  if (smallest == 0)
  {
    smallest = 1;
  }
  double tmp = value * smallest;
  this->SmallestSize = smallest;
  cout << "this->SmallestSize=" << this->SmallestSize << endl;
  this->Relative = 1;
  tmp = tmp * tmp;

  if (this->AbsoluteGeometricTolerance != tmp)
  {
    this->AbsoluteGeometricTolerance = tmp;
    this->Modified();
  }
}

#define SVTK_DISTANCE_LINE_POINT

//-----------------------------------------------------------------------------
int svtkGeometricErrorMetric::RequiresEdgeSubdivision(
  double* leftPoint, double* midPoint, double* rightPoint,
#ifdef SVTK_DISTANCE_LINE_POINT
  double svtkNotUsed(alpha)
#else
  double alpha
#endif
)
{
  assert("pre: leftPoint_exists" && leftPoint != nullptr);
  assert("pre: midPoint_exists" && midPoint != nullptr);
  assert("pre: rightPoint_exists" && rightPoint != nullptr);
  //  assert("pre: clamped_alpha" && alpha>0 && alpha<1); // or else true
  if (this->GenericCell->IsGeometryLinear())
  {
    // don't need to do anything:
    return 0;
  }
  // distance between the line (leftPoint,rightPoint) and the point midPoint.
#ifdef SVTK_DISTANCE_LINE_POINT
  return this->Distance2LinePoint(leftPoint, rightPoint, midPoint) >
    this->AbsoluteGeometricTolerance;
#else
  // Interpolated point
  double interpolatedPoint[3];
  int i = 0;
  while (i < 3)
  {
    interpolatedPoint[i] = leftPoint[i] + alpha * (rightPoint[i] - leftPoint[i]);
    ++i;
  }
  return svtkMath::Distance2BetweenPoints(midPoint, interpolatedPoint) >
    this->AbsoluteGeometricTolerance;
#endif
}

//-----------------------------------------------------------------------------
// Description:
// Return the error at the mid-point. The type of error depends on the state
// of the concrete error metric. For instance, it can return an absolute
// or relative error metric.
// See RequiresEdgeSubdivision() for a description of the arguments.
// \post positive_result: result>=0
double svtkGeometricErrorMetric::GetError(double* leftPoint, double* midPoint, double* rightPoint,
#ifdef SVTK_DISTANCE_LINE_POINT
  double svtkNotUsed(alpha)
#else
  double alpha
#endif
)
{
  assert("pre: leftPoint_exists" && leftPoint != nullptr);
  assert("pre: midPoint_exists" && midPoint != nullptr);
  assert("pre: rightPoint_exists" && rightPoint != nullptr);
  //  assert("pre: clamped_alpha" && alpha>0 && alpha<1); // or else true
  if (this->GenericCell->IsGeometryLinear())
  {
    // don't need to do anything:
    return 0;
  }
  // distance between the line (leftPoint,rightPoint) and the point midPoint.
#ifdef SVTK_DISTANCE_LINE_POINT
  double squareAbsoluteError = this->Distance2LinePoint(leftPoint, rightPoint, midPoint);
#else
  // Interpolated point
  double interpolatedPoint[3];
  int i = 0;
  while (i < 3)
  {
    interpolatedPoint[i] = leftPoint[i] + alpha * (rightPoint[i] - leftPoint[i]);
    ++i;
  }
  double squareAbsoluteError = svtkMath::Distance2BetweenPoints(midPoint, interpolatedPoint);
#endif
  if (this->Relative)
  {
    return sqrt(squareAbsoluteError) / this->SmallestSize;
  }
  else
  {
    return squareAbsoluteError;
  }
}

//-----------------------------------------------------------------------------
// Description:
// Return the type of output of GetError()
int svtkGeometricErrorMetric::GetRelative()
{
  return this->Relative;
}

//-----------------------------------------------------------------------------
// Description:
// Square distance between a straight line (defined by points x and y)
// and a point z. Property: if x and y are equal, the line is a point and
// the result is the square distance between points x and z.
double svtkGeometricErrorMetric::Distance2LinePoint(double x[3], double y[3], double z[3])
{
  double u[3];
  double v[3];
  double w[3];

  u[0] = y[0] - x[0];
  u[1] = y[1] - x[1];
  u[2] = y[2] - x[2];

  svtkMath::Normalize(u);

  v[0] = z[0] - x[0];
  v[1] = z[1] - x[1];
  v[2] = z[2] - x[2];

  double dot = svtkMath::Dot(u, v);

  w[0] = v[0] - dot * u[0];
  w[1] = v[1] - dot * u[1];
  w[2] = v[2] - dot * u[2];

  return svtkMath::Dot(w, w);
}

//-----------------------------------------------------------------------------
void svtkGeometricErrorMetric::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "AbsoluteGeometricTolerance: " << this->AbsoluteGeometricTolerance << endl;
}
