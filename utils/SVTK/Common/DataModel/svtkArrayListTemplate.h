/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayListTemplate.h

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See LICENSE file for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkArrayListTemplate
 * @brief   thread-safe and efficient data attribute manipulation
 *
 *
 * svtkArrayListTemplate supplements the svtkDataSetAttributes class to provide
 * threaded processing of data arrays. It is also more efficient for certain
 * interpolation operations. The expectation is that it will be replaced one
 * day once svtkPointData, svtkCellData, svtkDataSetAttributes, and svtkFieldData
 * properly support multithreading and/or are redesigned. Note that this
 * implementation does not support incremental operations (like InsertNext()).
 *
 * Generally the way this helper class is used is to first invoke
 * svtkDataSetAttributes::CopyInterpolate() or InterpolateAllocate() which
 * performs the initial magic of constructing input and output arrays. Then
 * the input attributes, and output attributes, are passed to initialize the
 * internal structures. Essentially these internal structures are pairs of
 * arrays of the same type, which can be efficiently accessed and
 * assigned. The operations on these array pairs (e.g., interpolation) occur
 * using a typeless, virtual dispatch base class.
 *
 * @sa
 * svtkFieldData svtkDataSetAttributes svtkPointData svtkCellData
 */

#ifndef svtkArrayListTemplate_h
#define svtkArrayListTemplate_h

#include "svtkDataArray.h"
#include "svtkDataSetAttributes.h"
#include "svtkSmartPointer.h"
#include "svtkStdString.h"

#include <algorithm>
#include <vector>

// Create a generic class supporting virtual dispatch to type-specific
// subclasses.
struct BaseArrayPair
{
  svtkIdType Num;
  int NumComp;
  svtkSmartPointer<svtkDataArray> OutputArray;

  BaseArrayPair(svtkIdType num, int numComp, svtkDataArray* outArray)
    : Num(num)
    , NumComp(numComp)
    , OutputArray(outArray)
  {
  }
  virtual ~BaseArrayPair() {}

  virtual void Copy(svtkIdType inId, svtkIdType outId) = 0;
  virtual void Interpolate(
    int numWeights, const svtkIdType* ids, const double* weights, svtkIdType outId) = 0;
  virtual void InterpolateEdge(svtkIdType v0, svtkIdType v1, double t, svtkIdType outId) = 0;
  virtual void AssignNullValue(svtkIdType outId) = 0;
  virtual void Realloc(svtkIdType sze) = 0;
};

// Type specific interpolation on a matched pair of data arrays
template <typename T>
struct ArrayPair : public BaseArrayPair
{
  T* Input;
  T* Output;
  T NullValue;

  ArrayPair(T* in, T* out, svtkIdType num, int numComp, svtkDataArray* outArray, T null)
    : BaseArrayPair(num, numComp, outArray)
    , Input(in)
    , Output(out)
    , NullValue(null)
  {
  }
  ~ArrayPair() override // calm down some finicky compilers
  {
  }

  void Copy(svtkIdType inId, svtkIdType outId) override
  {
    for (int j = 0; j < this->NumComp; ++j)
    {
      this->Output[outId * this->NumComp + j] = this->Input[inId * this->NumComp + j];
    }
  }

  void Interpolate(
    int numWeights, const svtkIdType* ids, const double* weights, svtkIdType outId) override
  {
    for (int j = 0; j < this->NumComp; ++j)
    {
      double v = 0.0;
      for (svtkIdType i = 0; i < numWeights; ++i)
      {
        v += weights[i] * static_cast<double>(this->Input[ids[i] * this->NumComp + j]);
      }
      this->Output[outId * this->NumComp + j] = static_cast<T>(v);
    }
  }

  void InterpolateEdge(svtkIdType v0, svtkIdType v1, double t, svtkIdType outId) override
  {
    double v;
    svtkIdType numComp = this->NumComp;
    for (int j = 0; j < numComp; ++j)
    {
      v = this->Input[v0 * numComp + j] +
        t * (this->Input[v1 * numComp + j] - this->Input[v0 * numComp + j]);
      this->Output[outId * numComp + j] = static_cast<T>(v);
    }
  }

  void AssignNullValue(svtkIdType outId) override
  {
    for (int j = 0; j < this->NumComp; ++j)
    {
      this->Output[outId * this->NumComp + j] = this->NullValue;
    }
  }

  void Realloc(svtkIdType sze) override
  {
    this->OutputArray->WriteVoidPointer(0, sze * this->NumComp);
    this->Output = static_cast<T*>(this->OutputArray->GetVoidPointer(0));
  }
};

// Type specific interpolation on a pair of data arrays with different types, where the
// output type is expected to be a real type (i.e., float or double).
template <typename TInput, typename TOutput>
struct RealArrayPair : public BaseArrayPair
{
  TInput* Input;
  TOutput* Output;
  TOutput NullValue;

  RealArrayPair(
    TInput* in, TOutput* out, svtkIdType num, int numComp, svtkDataArray* outArray, TOutput null)
    : BaseArrayPair(num, numComp, outArray)
    , Input(in)
    , Output(out)
    , NullValue(null)
  {
  }
  ~RealArrayPair() override // calm down some finicky compilers
  {
  }

  void Copy(svtkIdType inId, svtkIdType outId) override
  {
    for (int j = 0; j < this->NumComp; ++j)
    {
      this->Output[outId * this->NumComp + j] =
        static_cast<TOutput>(this->Input[inId * this->NumComp + j]);
    }
  }

  void Interpolate(
    int numWeights, const svtkIdType* ids, const double* weights, svtkIdType outId) override
  {
    for (int j = 0; j < this->NumComp; ++j)
    {
      double v = 0.0;
      for (svtkIdType i = 0; i < numWeights; ++i)
      {
        v += weights[i] * static_cast<double>(this->Input[ids[i] * this->NumComp + j]);
      }
      this->Output[outId * this->NumComp + j] = static_cast<TOutput>(v);
    }
  }

  void InterpolateEdge(svtkIdType v0, svtkIdType v1, double t, svtkIdType outId) override
  {
    double v;
    svtkIdType numComp = this->NumComp;
    for (int j = 0; j < numComp; ++j)
    {
      v = this->Input[v0 * numComp + j] +
        t * (this->Input[v1 * numComp + j] - this->Input[v0 * numComp + j]);
      this->Output[outId * numComp + j] = static_cast<TOutput>(v);
    }
  }

  void AssignNullValue(svtkIdType outId) override
  {
    for (int j = 0; j < this->NumComp; ++j)
    {
      this->Output[outId * this->NumComp + j] = this->NullValue;
    }
  }

  void Realloc(svtkIdType sze) override
  {
    this->OutputArray->WriteVoidPointer(0, sze * this->NumComp);
    this->Output = static_cast<TOutput*>(this->OutputArray->GetVoidPointer(0));
  }
};

// Forward declarations. This makes working with svtkTemplateMacro easier.
struct ArrayList;

template <typename T>
void CreateArrayPair(
  ArrayList* list, T* inData, T* outData, svtkIdType numTuples, int numComp, T nullValue);

// A list of the arrays to interpolate, and a method to invoke interpolation on the list
struct ArrayList
{
  // The list of arrays, and the arrays not to process
  std::vector<BaseArrayPair*> Arrays;
  std::vector<svtkDataArray*> ExcludedArrays;

  // Add the arrays to interpolate here (from attribute data)
  void AddArrays(svtkIdType numOutPts, svtkDataSetAttributes* inPD, svtkDataSetAttributes* outPD,
    double nullValue = 0.0, svtkTypeBool promote = true);

  // Add an array that interpolates from its own attribute values
  void AddSelfInterpolatingArrays(
    svtkIdType numOutPts, svtkDataSetAttributes* attr, double nullValue = 0.0);

  // Add a pair of arrays (manual insertion). Returns the output array created,
  // if any. No array may be created if \c inArray was previously marked as
  // excluded using ExcludeArray().
  svtkDataArray* AddArrayPair(svtkIdType numTuples, svtkDataArray* inArray, svtkStdString& outArrayName,
    double nullValue, svtkTypeBool promote);

  // Any array excluded here is not added by AddArrays() or AddArrayPair, hence not
  // processed. Also check whether an array is excluded.
  void ExcludeArray(svtkDataArray* da);
  svtkTypeBool IsExcluded(svtkDataArray* da);

  // Loop over the array pairs and copy data from one to another
  void Copy(svtkIdType inId, svtkIdType outId)
  {
    for (std::vector<BaseArrayPair*>::iterator it = Arrays.begin(); it != Arrays.end(); ++it)
    {
      (*it)->Copy(inId, outId);
    }
  }

  // Loop over the arrays and have them interpolate themselves
  void Interpolate(int numWeights, const svtkIdType* ids, const double* weights, svtkIdType outId)
  {
    for (std::vector<BaseArrayPair*>::iterator it = Arrays.begin(); it != Arrays.end(); ++it)
    {
      (*it)->Interpolate(numWeights, ids, weights, outId);
    }
  }

  // Loop over the arrays perform edge interpolation
  void InterpolateEdge(svtkIdType v0, svtkIdType v1, double t, svtkIdType outId)
  {
    for (std::vector<BaseArrayPair*>::iterator it = Arrays.begin(); it != Arrays.end(); ++it)
    {
      (*it)->InterpolateEdge(v0, v1, t, outId);
    }
  }

  // Loop over the arrays and assign the null value
  void AssignNullValue(svtkIdType outId)
  {
    for (std::vector<BaseArrayPair*>::iterator it = Arrays.begin(); it != Arrays.end(); ++it)
    {
      (*it)->AssignNullValue(outId);
    }
  }

  // Extend (realloc) the arrays
  void Realloc(svtkIdType sze)
  {
    for (std::vector<BaseArrayPair*>::iterator it = Arrays.begin(); it != Arrays.end(); ++it)
    {
      (*it)->Realloc(sze);
    }
  }

  // Only you can prevent memory leaks!
  ~ArrayList()
  {
    for (std::vector<BaseArrayPair*>::iterator it = Arrays.begin(); it != Arrays.end(); ++it)
    {
      delete (*it);
    }
  }

  // Return the number of arrays
  svtkIdType GetNumberOfArrays() { return static_cast<svtkIdType>(Arrays.size()); }
};

#include "svtkArrayListTemplate.txx"

#endif
// SVTK-HeaderTest-Exclude: svtkArrayListTemplate.h
