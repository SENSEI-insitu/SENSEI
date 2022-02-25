/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSortDataArray.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/*
 * Copyright 2003 Sandia Corporation.
 * Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
 * license for use of this work by or on behalf of the
 * U.S. Government. Redistribution and use in source and binary forms, with
 * or without modification, are permitted provided that this Notice and any
 * statement of authorship are reproduced on all copies.
 */

#include "svtkSortDataArray.h"

#include "svtkAbstractArray.h"
#include "svtkIdList.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkSMPTools.h"
#include "svtkStdString.h"
#include "svtkStringArray.h"
#include "svtkVariant.h"
#include "svtkVariantArray.h"
#include <functional> //std::greater

//-------------------------------------------------------------------------

svtkStandardNewMacro(svtkSortDataArray);

//-------------------------------------------------------------------------
svtkSortDataArray::svtkSortDataArray() = default;

//---------------------------------------------------------------------------
svtkSortDataArray::~svtkSortDataArray() = default;

//---------------------------------------------------------------------------
void svtkSortDataArray::Sort(svtkIdList* keys, int dir)
{
  if (keys == nullptr)
  {
    return;
  }
  svtkIdType* data = keys->GetPointer(0);
  svtkIdType numKeys = keys->GetNumberOfIds();
  if (dir == 0)
  {
    svtkSMPTools::Sort(data, data + numKeys);
  }
  else
  {
    svtkSMPTools::Sort(data, data + numKeys, std::greater<svtkIdType>());
  }
}

//---------------------------------------------------------------------------
void svtkSortDataArray::Sort(svtkAbstractArray* keys, int dir)
{
  if (keys == nullptr)
  {
    return;
  }

  if (keys->GetNumberOfComponents() != 1)
  {
    svtkGenericWarningMacro("Can only sort keys that are 1-tuples.");
    return;
  }

  void* data = keys->GetVoidPointer(0);
  svtkIdType numKeys = keys->GetNumberOfTuples();

  if (dir == 0)
  {
    switch (keys->GetDataType())
    {
      svtkExtendedTemplateMacro(
        svtkSMPTools::Sort(static_cast<SVTK_TT*>(data), static_cast<SVTK_TT*>(data) + numKeys));
    }
  }
  else
  {
    switch (keys->GetDataType())
    {
      svtkExtendedTemplateMacro(svtkSMPTools::Sort(
        static_cast<SVTK_TT*>(data), static_cast<SVTK_TT*>(data) + numKeys, std::greater<SVTK_TT>()));
    }
  }
}

//---------------------------------------------------------------------------
// Hide some stuff; mostly things plugged into templated functions
namespace
{

//---------------------------------------------------------------------------
// We sort the indices based on a key value in another array. Produces sort
// in ascending direction. Note that sort comparison operator is for single
// component arrays.
template <typename T>
struct KeyComp
{
  const T* Array;
  KeyComp(T* array)
    : Array(array)
  {
  }
  bool operator()(svtkIdType idx0, svtkIdType idx1) const { return (Array[idx0] < Array[idx1]); }
};

//-----------------------------------------------------------------------------
// Special comparison functor using tuple component as a key. Note that this
// comparison function is for general arrays of n components.
template <typename T>
struct TupleComp
{
  const T* Array;
  int NumComp;
  int K;
  TupleComp(T* array, int n, int k)
    : Array(array)
    , NumComp(n)
    , K(k)
  {
  }
  bool operator()(svtkIdType idx0, svtkIdType idx1) const
  {
    return Array[idx0 * NumComp + K] < Array[idx1 * NumComp + K];
  }
};

//---------------------------------------------------------------------------
// Given a set of indices (after sorting), copy the data from a pre-sorted
// array to a final, post-sorted array, Implementation note: the direction of
// sort (dir) is treated here rather than in the std::sort() function to
// reduce object file .obj size; e.g., running std::sort with a different
// comporator function causes inline expansion to produce very large object
// files.
template <typename T>
void Shuffle1Tuples(svtkIdType* idx, svtkIdType sze, svtkAbstractArray* arrayIn, T* preSort, int dir)
{
  T* postSort = new T[sze];

  if (dir == 0) // ascending
  {
    for (svtkIdType i = 0; i < sze; ++i)
    {
      postSort[i] = preSort[idx[i]];
    }
  }
  else
  {
    svtkIdType end = sze - 1;
    for (svtkIdType i = 0; i < sze; ++i)
    {
      postSort[i] = preSort[idx[end - i]];
    }
  }

  arrayIn->SetVoidArray(postSort, sze, 0, svtkAbstractArray::SVTK_DATA_ARRAY_DELETE);
}

//---------------------------------------------------------------------------
// Given a set of indices (after sorting), copy the data from a pre-sorted
// data array to a final, post-sorted array. Note that the data array is
// assumed to have arbitrary sized components.
template <typename T>
void ShuffleTuples(
  svtkIdType* idx, svtkIdType sze, int numComp, svtkAbstractArray* arrayIn, T* preSort, int dir)
{
  T* postSort = new T[sze * numComp];

  int k;
  svtkIdType i;
  if (dir == 0) // ascending
  {
    for (i = 0; i < sze; ++i)
    {
      for (k = 0; k < numComp; ++k)
      {
        postSort[i * numComp + k] = preSort[idx[i] * numComp + k];
      }
    }
  }
  else
  {
    svtkIdType end = sze - 1;
    for (i = 0; i < sze; ++i)
    {
      for (k = 0; k < numComp; ++k)
      {
        postSort[i * numComp + k] = preSort[idx[end - i] * numComp + k];
      }
    }
  }

  arrayIn->SetVoidArray(postSort, sze * numComp, 0, svtkAbstractArray::SVTK_DATA_ARRAY_DELETE);
}

} // anonymous namespace

//---------------------------------------------------------------------------
// Allocate and initialize sort indices
svtkIdType* svtkSortDataArray::InitializeSortIndices(svtkIdType num)
{
  svtkIdType* idx = new svtkIdType[num];
  for (svtkIdType i = 0; i < num; ++i)
  {
    idx[i] = i;
  }
  return idx;
}

//---------------------------------------------------------------------------
// Efficient function for generating sort ordering specialized to single
// component arrays.
void svtkSortDataArray::GenerateSort1Indices(
  int dataType, void* dataIn, svtkIdType numKeys, svtkIdType* idx)
{
  if (dataType == SVTK_VARIANT)
  {
    svtkSMPTools::Sort(idx, idx + numKeys, KeyComp<svtkVariant>(static_cast<svtkVariant*>(dataIn)));
  }
  else
  {
    switch (dataType)
    {
      svtkExtendedTemplateMacro(
        svtkSMPTools::Sort(idx, idx + numKeys, KeyComp<SVTK_TT>(static_cast<SVTK_TT*>(dataIn))));
    }
  }
}

//---------------------------------------------------------------------------
// Function for generating sort ordering for general arrays.
void svtkSortDataArray::GenerateSortIndices(
  int dataType, void* dataIn, svtkIdType numKeys, int numComp, int k, svtkIdType* idx)
{
  // Specialized and faster for single component arrays
  if (numComp == 1)
  {
    return svtkSortDataArray::GenerateSort1Indices(dataType, dataIn, numKeys, idx);
  }

  if (dataType == SVTK_VARIANT)
  {
    svtkSMPTools::Sort(
      idx, idx + numKeys, TupleComp<svtkVariant>(static_cast<svtkVariant*>(dataIn), numComp, k));
  }
  else
  {
    switch (dataType)
    {
      svtkExtendedTemplateMacro(svtkSMPTools::Sort(
        idx, idx + numKeys, TupleComp<SVTK_TT>(static_cast<SVTK_TT*>(dataIn), numComp, k)));
    }
  }
}

//-------------------------------------------------------------------------
// Set up the actual templated shuffling operation. This method is for
// SVTK arrays that are precsisely one component.
void svtkSortDataArray::Shuffle1Array(
  svtkIdType* idx, int dataType, svtkIdType numKeys, svtkAbstractArray* arr, void* dataIn, int dir)
{
  if (dataType == SVTK_VARIANT)
  {
    Shuffle1Tuples(idx, numKeys, arr, static_cast<svtkVariant*>(dataIn), dir);
  }
  else
  {
    switch (arr->GetDataType())
    {
      svtkExtendedTemplateMacro(
        Shuffle1Tuples(idx, numKeys, arr, static_cast<SVTK_TT*>(dataIn), dir));
    }
  }
}

//-------------------------------------------------------------------------
// Set up the actual templated shuffling operation
void svtkSortDataArray::ShuffleArray(svtkIdType* idx, int dataType, svtkIdType numKeys, int numComp,
  svtkAbstractArray* arr, void* dataIn, int dir)
{
  // Specialized for single component arrays
  if (numComp == 1)
  {
    return svtkSortDataArray::Shuffle1Array(idx, dataType, numKeys, arr, dataIn, dir);
  }

  if (dataType == SVTK_VARIANT)
  {
    ShuffleTuples(idx, numKeys, numComp, arr, static_cast<svtkVariant*>(dataIn), dir);
  }
  else
  {
    switch (arr->GetDataType())
    {
      svtkExtendedTemplateMacro(
        ShuffleTuples(idx, numKeys, numComp, arr, static_cast<SVTK_TT*>(dataIn), dir));
    }
  }
}

//---------------------------------------------------------------------------
// Given a set of indices (after sorting), copy the ids from a pre-sorted
// id array to a final, post-sorted array.
void svtkSortDataArray::ShuffleIdList(
  svtkIdType* idx, svtkIdType sze, svtkIdList* arrayIn, svtkIdType* preSort, int dir)
{
  svtkIdType* postSort = new svtkIdType[sze];

  if (dir == 0) // ascending
  {
    for (svtkIdType i = 0; i < sze; ++i)
    {
      postSort[i] = preSort[idx[i]];
    }
  }
  else
  {
    svtkIdType end = sze - 1;
    for (svtkIdType i = 0; i < sze; ++i)
    {
      postSort[i] = preSort[idx[end - i]];
    }
  }

  arrayIn->SetArray(postSort, sze);
}

//---------------------------------------------------------------------------
// Sort a position index based on the values in the abstract array. Once
// sorted, then shuffle the keys and values around into new arrays.
void svtkSortDataArray::Sort(svtkAbstractArray* keys, svtkAbstractArray* values, int dir)
{
  // Check input
  if (keys == nullptr || values == nullptr)
  {
    return;
  }
  if (keys->GetNumberOfComponents() != 1)
  {
    svtkGenericWarningMacro("Can only sort keys that are 1-tuples.");
    return;
  }
  svtkIdType numKeys = keys->GetNumberOfTuples();
  svtkIdType numValues = values->GetNumberOfTuples();
  if (numKeys != numValues)
  {
    svtkGenericWarningMacro("Could not sort arrays.  Key and value arrays have different sizes.");
    return;
  }

  // Sort the index array
  svtkIdType* idx = svtkSortDataArray::InitializeSortIndices(numKeys);

  // Generate the sorting index array
  void* dataIn = keys->GetVoidPointer(0);
  int numComp = 1;
  int dataType = keys->GetDataType();
  svtkSortDataArray::GenerateSortIndices(dataType, dataIn, numKeys, numComp, 0, idx);

  // Now shuffle data around based on sorted indices
  svtkSortDataArray::ShuffleArray(idx, dataType, numKeys, numComp, keys, dataIn, dir);

  dataIn = values->GetVoidPointer(0);
  numComp = values->GetNumberOfComponents();
  dataType = values->GetDataType();
  svtkSortDataArray::ShuffleArray(idx, dataType, numKeys, numComp, values, dataIn, dir);

  // Clean up
  delete[] idx;
}

//---------------------------------------------------------------------------
void svtkSortDataArray::Sort(svtkAbstractArray* keys, svtkIdList* values, int dir)
{
  // Check input
  if (keys == nullptr || values == nullptr)
  {
    return;
  }
  if (keys->GetNumberOfComponents() != 1)
  {
    svtkGenericWarningMacro("Can only sort keys that are 1-tuples.");
    return;
  }
  svtkIdType numKeys = keys->GetNumberOfTuples();
  svtkIdType numIds = values->GetNumberOfIds();
  if (numKeys != numIds)
  {
    svtkGenericWarningMacro("Could not sort arrays.  Key and id arrays have different sizes.");
    return;
  }

  // Sort the index array
  svtkIdType* idx = svtkSortDataArray::InitializeSortIndices(numKeys);

  // Generate the sorting index array
  void* dataIn = keys->GetVoidPointer(0);
  int numComp = 1;
  int dataType = keys->GetDataType();
  svtkSortDataArray::GenerateSortIndices(dataType, dataIn, numKeys, numComp, 0, idx);

  // Shuffle the keys
  svtkSortDataArray::ShuffleArray(idx, dataType, numKeys, numComp, keys, dataIn, dir);

  // Now shuffle the ids to match the sort
  svtkIdType* ids = values->GetPointer(0);
  ShuffleIdList(idx, numKeys, values, ids, dir);

  // Clean up
  delete[] idx;
}

//---------------------------------------------------------------------------
void svtkSortDataArray::SortArrayByComponent(svtkAbstractArray* arr, int k, int dir)
{
  // Check input
  if (arr == nullptr)
  {
    return;
  }
  svtkIdType numKeys = arr->GetNumberOfTuples();
  int nc = arr->GetNumberOfComponents();

  if (k < 0 || k >= nc)
  {
    svtkGenericWarningMacro(
      "Cannot sort by column " << k << " since the array only has columns 0 through " << (nc - 1));
    return;
  }

  // Perform the sort
  svtkIdType* idx = svtkSortDataArray::InitializeSortIndices(numKeys);

  void* dataIn = arr->GetVoidPointer(0);
  int dataType = arr->GetDataType();
  svtkSortDataArray::GenerateSortIndices(dataType, dataIn, numKeys, nc, k, idx);

  svtkSortDataArray::ShuffleArray(idx, dataType, numKeys, nc, arr, dataIn, dir);

  // Clean up
  delete[] idx;
}

//-------------------------------------------------------------------------
void svtkSortDataArray::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

// svtkSortDataArray methods -------------------------------------------------------
