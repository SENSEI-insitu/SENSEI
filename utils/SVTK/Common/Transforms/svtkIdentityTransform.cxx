/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkIdentityTransform.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkIdentityTransform.h"

#include "svtkDataArray.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPoints.h"

svtkStandardNewMacro(svtkIdentityTransform);

//----------------------------------------------------------------------------
svtkIdentityTransform::svtkIdentityTransform() = default;

//----------------------------------------------------------------------------
svtkIdentityTransform::~svtkIdentityTransform() = default;

//----------------------------------------------------------------------------
void svtkIdentityTransform::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------
void svtkIdentityTransform::InternalDeepCopy(svtkAbstractTransform*)
{
  // nothin' to do
}

//----------------------------------------------------------------------------
svtkAbstractTransform* svtkIdentityTransform::MakeTransform()
{
  return svtkIdentityTransform::New();
}

//------------------------------------------------------------------------
template <class T2, class T3>
void svtkIdentityTransformPoint(T2 in[3], T3 out[3])
{
  out[0] = in[0];
  out[1] = in[1];
  out[2] = in[2];
}

//------------------------------------------------------------------------
template <class T2, class T3, class T4>
void svtkIdentityTransformDerivative(T2 in[3], T3 out[3], T4 derivative[3][3])
{
  out[0] = in[0];
  out[1] = in[1];
  out[2] = in[2];

  svtkMath::Identity3x3(derivative);
}

//------------------------------------------------------------------------
void svtkIdentityTransform::InternalTransformPoint(const float in[3], float out[3])
{
  svtkIdentityTransformPoint(in, out);
}

//------------------------------------------------------------------------
void svtkIdentityTransform::InternalTransformPoint(const double in[3], double out[3])
{
  svtkIdentityTransformPoint(in, out);
}

//------------------------------------------------------------------------
void svtkIdentityTransform::InternalTransformNormal(const float in[3], float out[3])
{
  svtkIdentityTransformPoint(in, out);
  svtkMath::Normalize(out);
}

//------------------------------------------------------------------------
void svtkIdentityTransform::InternalTransformNormal(const double in[3], double out[3])
{
  svtkIdentityTransformPoint(in, out);
  svtkMath::Normalize(out);
}

//------------------------------------------------------------------------
void svtkIdentityTransform::InternalTransformVector(const float in[3], float out[3])
{
  svtkIdentityTransformPoint(in, out);
}

//------------------------------------------------------------------------
void svtkIdentityTransform::InternalTransformVector(const double in[3], double out[3])
{
  svtkIdentityTransformPoint(in, out);
}

//----------------------------------------------------------------------------
void svtkIdentityTransform::InternalTransformDerivative(
  const float in[3], float out[3], float derivative[3][3])
{
  svtkIdentityTransformDerivative(in, out, derivative);
}

//----------------------------------------------------------------------------
void svtkIdentityTransform::InternalTransformDerivative(
  const double in[3], double out[3], double derivative[3][3])
{
  svtkIdentityTransformDerivative(in, out, derivative);
}

//----------------------------------------------------------------------------
// Transform the normals and vectors using the derivative of the
// transformation.  Either inNms or inVrs can be set to nullptr.
// Normals are multiplied by the inverse transpose of the transform
// derivative, while vectors are simply multiplied by the derivative.
// Note that the derivative of the inverse transform is simply the
// inverse of the derivative of the forward transform.
void svtkIdentityTransform::TransformPointsNormalsVectors(svtkPoints* inPts, svtkPoints* outPts,
  svtkDataArray* inNms, svtkDataArray* outNms, svtkDataArray* inVrs, svtkDataArray* outVrs,
  int nOptionalVectors, svtkDataArray** inVrsArr, svtkDataArray** outVrsArr)
{
  this->TransformPoints(inPts, outPts);
  if (inNms)
  {
    this->TransformNormals(inNms, outNms);
  }
  if (inVrs)
  {
    this->TransformVectors(inVrs, outVrs);
  }
  if (inVrsArr)
  {
    for (int iArr = 0; iArr < nOptionalVectors; iArr++)
    {
      this->TransformVectors(inVrsArr[iArr], outVrsArr[iArr]);
    }
  }
}

//----------------------------------------------------------------------------
void svtkIdentityTransform::TransformPoints(svtkPoints* inPts, svtkPoints* outPts)
{
  svtkIdType n = inPts->GetNumberOfPoints();
  double point[3];

  for (svtkIdType i = 0; i < n; i++)
  {
    inPts->GetPoint(i, point);
    outPts->InsertNextPoint(point);
  }
}

//----------------------------------------------------------------------------
void svtkIdentityTransform::TransformNormals(svtkDataArray* inNms, svtkDataArray* outNms)
{
  svtkIdType n = inNms->GetNumberOfTuples();
  double normal[3];

  for (svtkIdType i = 0; i < n; i++)
  {
    inNms->GetTuple(i, normal);
    outNms->InsertNextTuple(normal);
  }
}

//----------------------------------------------------------------------------
void svtkIdentityTransform::TransformVectors(svtkDataArray* inNms, svtkDataArray* outNms)
{
  svtkIdType n = inNms->GetNumberOfTuples();
  double vect[3];

  for (svtkIdType i = 0; i < n; i++)
  {
    inNms->GetTuple(i, vect);
    outNms->InsertNextTuple(vect);
  }
}
