/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSphericalTransform.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkSphericalTransform.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include <cmath>
#include <cstdlib>

svtkStandardNewMacro(svtkSphericalTransform);

//----------------------------------------------------------------------------
svtkSphericalTransform::svtkSphericalTransform() = default;

svtkSphericalTransform::~svtkSphericalTransform() = default;

void svtkSphericalTransform::PrintSelf(ostream& os, svtkIndent indent)
{
  svtkWarpTransform::PrintSelf(os, indent);
}

void svtkSphericalTransform::InternalDeepCopy(svtkAbstractTransform* transform)
{
  svtkSphericalTransform* sphericalTransform = static_cast<svtkSphericalTransform*>(transform);

  // copy these even though they aren't used
  this->SetInverseTolerance(sphericalTransform->InverseTolerance);
  this->SetInverseIterations(sphericalTransform->InverseIterations);

  // copy the inverse flag, which is used
  if (this->InverseFlag != sphericalTransform->InverseFlag)
  {
    this->InverseFlag = sphericalTransform->InverseFlag;
    this->Modified();
  }
}

svtkAbstractTransform* svtkSphericalTransform::MakeTransform()
{
  return svtkSphericalTransform::New();
}

template <class T>
void svtkSphericalToRectangular(const T inPoint[3], T outPoint[3], T derivative[3][3])
{
  T r = inPoint[0];
  T sinphi = sin(inPoint[1]);
  T cosphi = cos(inPoint[1]);
  T sintheta = sin(inPoint[2]);
  T costheta = cos(inPoint[2]);

  outPoint[0] = r * sinphi * costheta;
  outPoint[1] = r * sinphi * sintheta;
  outPoint[2] = r * cosphi;

  if (derivative)
  {
    derivative[0][0] = sinphi * costheta;
    derivative[0][1] = r * cosphi * costheta;
    derivative[0][2] = -r * sinphi * sintheta;

    derivative[1][0] = sinphi * sintheta;
    derivative[1][1] = r * cosphi * sintheta;
    derivative[1][2] = r * sinphi * costheta;

    derivative[2][0] = cosphi;
    derivative[2][1] = -r * sinphi;
    derivative[2][2] = 0;
  }
}

template <class T>
void svtkRectangularToSpherical(const T inPoint[3], T outPoint[3])
{
  T x = inPoint[0];
  T y = inPoint[1];
  T z = inPoint[2];

  T RR = x * x + y * y;
  T r = sqrt(RR + z * z);

  outPoint[0] = r;
  if (r == 0)
  {
    outPoint[1] = 0;
  }
  else
  {
    outPoint[1] = acos(z / r);
  }
  if (RR == 0)
  {
    outPoint[2] = 0;
  }
  else
  {
    // Change range to [0, 2*Pi], otherwise the same as atan2(y, x)
    outPoint[2] = T(svtkMath::Pi()) + atan2(-y, -x);
  }
}

void svtkSphericalTransform::ForwardTransformPoint(const float inPoint[3], float outPoint[3])
{
  svtkSphericalToRectangular(inPoint, outPoint, static_cast<float(*)[3]>(nullptr));
}

void svtkSphericalTransform::ForwardTransformPoint(const double inPoint[3], double outPoint[3])
{
  svtkSphericalToRectangular(inPoint, outPoint, static_cast<double(*)[3]>(nullptr));
}

void svtkSphericalTransform::ForwardTransformDerivative(
  const float inPoint[3], float outPoint[3], float derivative[3][3])
{
  svtkSphericalToRectangular(inPoint, outPoint, derivative);
}

void svtkSphericalTransform::ForwardTransformDerivative(
  const double inPoint[3], double outPoint[3], double derivative[3][3])
{
  svtkSphericalToRectangular(inPoint, outPoint, derivative);
}

void svtkSphericalTransform::InverseTransformPoint(const float inPoint[3], float outPoint[3])
{
  svtkRectangularToSpherical(inPoint, outPoint);
}

void svtkSphericalTransform::InverseTransformPoint(const double inPoint[3], double outPoint[3])
{
  svtkRectangularToSpherical(inPoint, outPoint);
}

void svtkSphericalTransform::InverseTransformDerivative(
  const float inPoint[3], float outPoint[3], float derivative[3][3])
{
  float tmp[3];
  svtkRectangularToSpherical(inPoint, outPoint);
  svtkSphericalToRectangular(outPoint, tmp, derivative);
}

void svtkSphericalTransform::InverseTransformDerivative(
  const double inPoint[3], double outPoint[3], double derivative[3][3])
{
  double tmp[3];
  svtkRectangularToSpherical(inPoint, outPoint);
  svtkSphericalToRectangular(outPoint, tmp, derivative);
}
