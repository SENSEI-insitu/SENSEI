/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLinearTransform.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkLinearTransform.h"

#include "svtkDataArray.h"
#include "svtkMath.h"
#include "svtkMatrix4x4.h"
#include "svtkPoints.h"

//------------------------------------------------------------------------
void svtkLinearTransform::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------
template <class T1, class T2, class T3>
inline void svtkLinearTransformPoint(T1 matrix[4][4], T2 in[3], T3 out[3])
{
  T3 x = static_cast<T3>(
    matrix[0][0] * in[0] + matrix[0][1] * in[1] + matrix[0][2] * in[2] + matrix[0][3]);
  T3 y = static_cast<T3>(
    matrix[1][0] * in[0] + matrix[1][1] * in[1] + matrix[1][2] * in[2] + matrix[1][3]);
  T3 z = static_cast<T3>(
    matrix[2][0] * in[0] + matrix[2][1] * in[1] + matrix[2][2] * in[2] + matrix[2][3]);

  out[0] = x;
  out[1] = y;
  out[2] = z;
}

//------------------------------------------------------------------------
template <class T1, class T2, class T3, class T4>
inline void svtkLinearTransformDerivative(T1 matrix[4][4], T2 in[3], T3 out[3], T4 derivative[3][3])
{
  svtkLinearTransformPoint(matrix, in, out);

  for (int i = 0; i < 3; i++)
  {
    derivative[0][i] = static_cast<T4>(matrix[0][i]);
    derivative[1][i] = static_cast<T4>(matrix[1][i]);
    derivative[2][i] = static_cast<T4>(matrix[2][i]);
  }
}

//------------------------------------------------------------------------
template <class T1, class T2, class T3>
inline void svtkLinearTransformVector(T1 matrix[4][4], T2 in[3], T3 out[3])
{
  T3 x = static_cast<T3>(matrix[0][0] * in[0] + matrix[0][1] * in[1] + matrix[0][2] * in[2]);
  T3 y = static_cast<T3>(matrix[1][0] * in[0] + matrix[1][1] * in[1] + matrix[1][2] * in[2]);
  T3 z = static_cast<T3>(matrix[2][0] * in[0] + matrix[2][1] * in[1] + matrix[2][2] * in[2]);

  out[0] = x;
  out[1] = y;
  out[2] = z;
}

//------------------------------------------------------------------------
template <class T1, class T2, class T3>
inline void svtkLinearTransformNormal(T1 mat[4][4], T2 in[3], T3 out[3])
{
  // to transform the normal, multiply by the transposed inverse matrix
  T1 matrix[4][4];
  memcpy(*matrix, *mat, 16 * sizeof(T1));
  svtkMatrix4x4::Invert(*matrix, *matrix);
  svtkMatrix4x4::Transpose(*matrix, *matrix);

  svtkLinearTransformVector(matrix, in, out);

  svtkMath::Normalize(out);
}

//------------------------------------------------------------------------
template <class T1, class T2, class T3>
inline void svtkLinearTransformPoints(T1 matrix[4][4], T2* in, T3* out, svtkIdType n)
{
  for (svtkIdType i = 0; i < n; i++)
  {
    svtkLinearTransformPoint(matrix, in, out);
    in += 3;
    out += 3;
  }
}

//------------------------------------------------------------------------
template <class T1, class T2, class T3>
inline void svtkLinearTransformVectors(T1 matrix[4][4], T2* in, T3* out, svtkIdType n)
{
  for (svtkIdType i = 0; i < n; i++)
  {
    svtkLinearTransformVector(matrix, in, out);
    in += 3;
    out += 3;
  }
}

//------------------------------------------------------------------------
template <class T1, class T2, class T3>
inline void svtkLinearTransformNormals(T1 matrix[4][4], T2* in, T3* out, svtkIdType n)
{
  for (svtkIdType i = 0; i < n; i++)
  {
    // matrix has been transposed & inverted, so use TransformVector
    svtkLinearTransformVector(matrix, in, out);
    svtkMath::Normalize(out);
    in += 3;
    out += 3;
  }
}

//------------------------------------------------------------------------
void svtkLinearTransform::InternalTransformPoint(const float in[3], float out[3])
{
  svtkLinearTransformPoint(this->Matrix->Element, in, out);
}

//------------------------------------------------------------------------
void svtkLinearTransform::InternalTransformPoint(const double in[3], double out[3])
{
  svtkLinearTransformPoint(this->Matrix->Element, in, out);
}

//------------------------------------------------------------------------
void svtkLinearTransform::InternalTransformNormal(const float in[3], float out[3])
{
  svtkLinearTransformNormal(this->Matrix->Element, in, out);
}

//------------------------------------------------------------------------
void svtkLinearTransform::InternalTransformNormal(const double in[3], double out[3])
{
  svtkLinearTransformNormal(this->Matrix->Element, in, out);
}

//------------------------------------------------------------------------
void svtkLinearTransform::InternalTransformVector(const float in[3], float out[3])
{
  svtkLinearTransformVector(this->Matrix->Element, in, out);
}

//------------------------------------------------------------------------
void svtkLinearTransform::InternalTransformVector(const double in[3], double out[3])
{
  svtkLinearTransformVector(this->Matrix->Element, in, out);
}

//----------------------------------------------------------------------------
void svtkLinearTransform::InternalTransformDerivative(
  const float in[3], float out[3], float derivative[3][3])
{
  svtkLinearTransformDerivative(this->Matrix->Element, in, out, derivative);
}

//----------------------------------------------------------------------------
void svtkLinearTransform::InternalTransformDerivative(
  const double in[3], double out[3], double derivative[3][3])
{
  svtkLinearTransformDerivative(this->Matrix->Element, in, out, derivative);
}

//----------------------------------------------------------------------------
// Transform the normals and vectors using the derivative of the
// transformation.  Either inNms or inVrs can be set to nullptr.
// Normals are multiplied by the inverse transpose of the transform
// derivative, while vectors are simply multiplied by the derivative.
// Note that the derivative of the inverse transform is simply the
// inverse of the derivative of the forward transform.
void svtkLinearTransform::TransformPointsNormalsVectors(svtkPoints* inPts, svtkPoints* outPts,
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
void svtkLinearTransform::TransformPoints(svtkPoints* inPts, svtkPoints* outPts)
{
  svtkIdType n = inPts->GetNumberOfPoints();
  svtkIdType m = outPts->GetNumberOfPoints();
  double(*matrix)[4] = this->Matrix->Element;

  this->Update();

  // operate directly on the memory to avoid GetPoint()/SetPoint() calls.
  svtkDataArray* inArray = inPts->GetData();
  svtkDataArray* outArray = outPts->GetData();
  int inType = inArray->GetDataType();
  int outType = outArray->GetDataType();
  void* inPtr = inArray->GetVoidPointer(0);
  void* outPtr = outArray->WriteVoidPointer(3 * m, 3 * n);

  if (inType == SVTK_FLOAT && outType == SVTK_FLOAT)
  {
    svtkLinearTransformPoints(matrix, static_cast<float*>(inPtr), static_cast<float*>(outPtr), n);
  }
  else if (inType == SVTK_FLOAT && outType == SVTK_DOUBLE)
  {
    svtkLinearTransformPoints(matrix, static_cast<float*>(inPtr), static_cast<double*>(outPtr), n);
  }
  else if (inType == SVTK_DOUBLE && outType == SVTK_FLOAT)
  {
    svtkLinearTransformPoints(matrix, static_cast<double*>(inPtr), static_cast<float*>(outPtr), n);
  }
  else if (inType == SVTK_DOUBLE && outType == SVTK_DOUBLE)
  {
    svtkLinearTransformPoints(matrix, static_cast<double*>(inPtr), static_cast<double*>(outPtr), n);
  }
  else
  {
    double point[3];

    for (svtkIdType i = 0; i < n; i++)
    {
      inPts->GetPoint(i, point);

      svtkLinearTransformPoint(matrix, point, point);

      outPts->SetPoint(m + i, point);
    }
  }
}

//----------------------------------------------------------------------------
void svtkLinearTransform::TransformNormals(svtkDataArray* inNms, svtkDataArray* outNms)
{
  svtkIdType n = inNms->GetNumberOfTuples();
  svtkIdType m = outNms->GetNumberOfTuples();
  double matrix[4][4];

  this->Update();

  // to transform the normal, multiply by the transposed inverse matrix
  svtkMatrix4x4::DeepCopy(*matrix, this->Matrix);
  svtkMatrix4x4::Invert(*matrix, *matrix);
  svtkMatrix4x4::Transpose(*matrix, *matrix);

  // operate directly on the memory to avoid GetTuple()/SetPoint() calls.
  int inType = inNms->GetDataType();
  int outType = outNms->GetDataType();
  void* inPtr = inNms->GetVoidPointer(0);
  void* outPtr = outNms->WriteVoidPointer(3 * m, 3 * n);

  if (inType == SVTK_FLOAT && outType == SVTK_FLOAT)
  {
    svtkLinearTransformNormals(matrix, static_cast<float*>(inPtr), static_cast<float*>(outPtr), n);
  }
  else if (inType == SVTK_FLOAT && outType == SVTK_DOUBLE)
  {
    svtkLinearTransformNormals(matrix, static_cast<float*>(inPtr), static_cast<double*>(outPtr), n);
  }
  else if (inType == SVTK_DOUBLE && outType == SVTK_FLOAT)
  {
    svtkLinearTransformNormals(matrix, static_cast<double*>(inPtr), static_cast<float*>(outPtr), n);
  }
  else if (inType == SVTK_DOUBLE && outType == SVTK_DOUBLE)
  {
    svtkLinearTransformNormals(matrix, static_cast<double*>(inPtr), static_cast<double*>(outPtr), n);
  }
  else
  {
    for (svtkIdType i = 0; i < n; i++)
    {
      double norm[3];

      inNms->GetTuple(i, norm);

      // use TransformVector because matrix is already transposed & inverted
      svtkLinearTransformVector(matrix, norm, norm);
      svtkMath::Normalize(norm);

      outNms->SetTuple(m + i, norm);
    }
  }
}

//----------------------------------------------------------------------------
void svtkLinearTransform::TransformVectors(svtkDataArray* inVrs, svtkDataArray* outVrs)
{
  svtkIdType n = inVrs->GetNumberOfTuples();
  svtkIdType m = outVrs->GetNumberOfTuples();

  double(*matrix)[4] = this->Matrix->Element;

  this->Update();

  // operate directly on the memory to avoid GetTuple()/SetTuple() calls.
  int inType = inVrs->GetDataType();
  int outType = outVrs->GetDataType();
  void* inPtr = inVrs->GetVoidPointer(0);
  void* outPtr = outVrs->WriteVoidPointer(3 * m, 3 * n);

  if (inType == SVTK_FLOAT && outType == SVTK_FLOAT)
  {
    svtkLinearTransformVectors(matrix, static_cast<float*>(inPtr), static_cast<float*>(outPtr), n);
  }
  else if (inType == SVTK_FLOAT && outType == SVTK_DOUBLE)
  {
    svtkLinearTransformVectors(matrix, static_cast<float*>(inPtr), static_cast<double*>(outPtr), n);
  }
  else if (inType == SVTK_DOUBLE && outType == SVTK_FLOAT)
  {
    svtkLinearTransformVectors(matrix, static_cast<double*>(inPtr), static_cast<float*>(outPtr), n);
  }
  else if (inType == SVTK_DOUBLE && outType == SVTK_DOUBLE)
  {
    svtkLinearTransformVectors(matrix, static_cast<double*>(inPtr), static_cast<double*>(outPtr), n);
  }
  else
  {
    for (svtkIdType i = 0; i < n; i++)
    {
      double vec[3];

      inVrs->GetTuple(i, vec);

      svtkLinearTransformVector(matrix, vec, vec);

      outVrs->SetTuple(m + i, vec);
    }
  }
}
