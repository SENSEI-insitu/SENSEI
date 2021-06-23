/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayListTemplate.txx

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See LICENSE file for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkArrayListTemplate.h"
#include "svtkFloatArray.h"

#include <cassert>

#ifndef svtkArrayListTemplate_txx
#define svtkArrayListTemplate_txx

//----------------------------------------------------------------------------
// Sort of a little object factory (in conjunction w/ svtkTemplateMacro())
template <typename T>
void CreateArrayPair(ArrayList* list, T* inData, T* outData, svtkIdType numTuples, int numComp,
  svtkDataArray* outArray, T nullValue)
{
  ArrayPair<T>* pair = new ArrayPair<T>(inData, outData, numTuples, numComp, outArray, nullValue);
  list->Arrays.push_back(pair);
}

//----------------------------------------------------------------------------
// Sort of a little object factory (in conjunction w/ svtkTemplateMacro())
template <typename T>
void CreateRealArrayPair(ArrayList* list, T* inData, float* outData, svtkIdType numTuples,
  int numComp, svtkDataArray* outArray, float nullValue)
{
  RealArrayPair<T, float>* pair =
    new RealArrayPair<T, float>(inData, outData, numTuples, numComp, outArray, nullValue);
  list->Arrays.push_back(pair);
}

//----------------------------------------------------------------------------
// Indicate arrays not to process
inline void ArrayList::ExcludeArray(svtkDataArray* da)
{
  ExcludedArrays.push_back(da);
}

//----------------------------------------------------------------------------
// Has the specified array been excluded?
inline svtkTypeBool ArrayList::IsExcluded(svtkDataArray* da)
{
  return (std::find(ExcludedArrays.begin(), ExcludedArrays.end(), da) != ExcludedArrays.end());
}

//----------------------------------------------------------------------------
// Add an array pair (input,output) using the name provided for the output. The
// numTuples is the number of output tuples allocated.
inline svtkDataArray* ArrayList::AddArrayPair(svtkIdType numTuples, svtkDataArray* inArray,
  svtkStdString& outArrayName, double nullValue, svtkTypeBool promote)
{
  if (this->IsExcluded(inArray))
  {
    return nullptr;
  }

  int iType = inArray->GetDataType();
  svtkDataArray* outArray;
  if (promote && iType != SVTK_FLOAT && iType != SVTK_DOUBLE)
  {
    outArray = svtkFloatArray::New();
    outArray->SetNumberOfComponents(inArray->GetNumberOfComponents());
    outArray->SetNumberOfTuples(numTuples);
    outArray->SetName(outArrayName);
    void* iD = inArray->GetVoidPointer(0);
    void* oD = outArray->GetVoidPointer(0);
    switch (iType)
    {
      svtkTemplateMacro(CreateRealArrayPair(this, static_cast<SVTK_TT*>(iD), static_cast<float*>(oD),
        numTuples, inArray->GetNumberOfComponents(), outArray, static_cast<float>(nullValue)));
    } // over all SVTK types
  }
  else
  {
    outArray = inArray->NewInstance();
    outArray->SetNumberOfComponents(inArray->GetNumberOfComponents());
    outArray->SetNumberOfTuples(numTuples);
    outArray->SetName(outArrayName);
    void* iD = inArray->GetVoidPointer(0);
    void* oD = outArray->GetVoidPointer(0);
    switch (iType)
    {
      svtkTemplateMacro(CreateArrayPair(this, static_cast<SVTK_TT*>(iD), static_cast<SVTK_TT*>(oD),
        numTuples, inArray->GetNumberOfComponents(), outArray, static_cast<SVTK_TT>(nullValue)));
    } // over all SVTK types
  }   // promote integral types

  assert(outArray->GetReferenceCount() > 1);
  outArray->FastDelete();
  return outArray;
}

//----------------------------------------------------------------------------
// Add the arrays to interpolate here. This presumes that svtkDataSetAttributes::CopyData() or
// svtkDataSetAttributes::InterpolateData() has been called, and the input and output array
// names match.
inline void ArrayList::AddArrays(svtkIdType numOutPts, svtkDataSetAttributes* inPD,
  svtkDataSetAttributes* outPD, double nullValue, svtkTypeBool promote)
{
  // Build the vector of interpolation pairs. Note that InterpolateAllocate should have
  // been called at this point (output arrays created and allocated).
  char* name;
  svtkDataArray *iArray, *oArray;
  int iType, oType;
  void *iD, *oD;
  int iNumComp, oNumComp;
  int i, numArrays = outPD->GetNumberOfArrays();

  for (i = 0; i < numArrays; ++i)
  {
    oArray = outPD->GetArray(i);
    if (oArray && !this->IsExcluded(oArray))
    {
      name = oArray->GetName();
      iArray = inPD->GetArray(name);
      if (iArray && !this->IsExcluded(iArray))
      {
        iType = iArray->GetDataType();
        oType = oArray->GetDataType();
        iNumComp = iArray->GetNumberOfComponents();
        oNumComp = oArray->GetNumberOfComponents();
        if (promote && oType != SVTK_FLOAT && oType != SVTK_DOUBLE)
        {
          oType = SVTK_FLOAT;
          svtkFloatArray* fArray = svtkFloatArray::New();
          fArray->SetName(oArray->GetName());
          fArray->SetNumberOfComponents(oNumComp);
          outPD->AddArray(fArray); // nasty side effect will replace current array in the same spot
          oArray = fArray;
          fArray->Delete();
        }
        oArray->SetNumberOfTuples(numOutPts);

        assert(iNumComp == oNumComp);
        if (iType == oType)
        {
          iD = iArray->GetVoidPointer(0);
          oD = oArray->GetVoidPointer(0);
          switch (iType)
          {
            svtkTemplateMacro(
              CreateArrayPair(this, static_cast<SVTK_TT*>(iD), static_cast<SVTK_TT*>(oD), numOutPts,
                oNumComp, oArray, static_cast<SVTK_TT>(nullValue)));
          }  // over all SVTK types
        }    // if matching types
        else // promoted type
        {
          iD = iArray->GetVoidPointer(0);
          oD = oArray->GetVoidPointer(0);
          switch (iType)
          {
            svtkTemplateMacro(CreateRealArrayPair(this, static_cast<SVTK_TT*>(iD),
              static_cast<float*>(oD), numOutPts, iNumComp, oArray, static_cast<float>(nullValue)));
          } // over all SVTK types
        }   // if promoted pair
      }     // if matching input array
    }       // if output array
  }         // for each candidate array
}

//----------------------------------------------------------------------------
// Add the arrays to interpolate here. This presumes that svtkDataSetAttributes::CopyData() or
// svtkDataSetAttributes::InterpolateData() has been called. This special version creates an
// array pair that interpolates from itself.
inline void ArrayList::AddSelfInterpolatingArrays(
  svtkIdType numOutPts, svtkDataSetAttributes* attr, double nullValue)
{
  // Build the vector of interpolation pairs. Note that CopyAllocate/InterpolateAllocate should have
  // been called at this point (output arrays created and allocated).
  svtkDataArray* iArray;
  int iType, iNumComp;
  void* iD;
  int i, numArrays = attr->GetNumberOfArrays();

  for (i = 0; i < numArrays; ++i)
  {
    iArray = attr->GetArray(i);
    if (iArray && !this->IsExcluded(iArray))
    {
      iType = iArray->GetDataType();
      iNumComp = iArray->GetNumberOfComponents();
      iArray->WriteVoidPointer(0, numOutPts * iNumComp); // allocates memory, preserves data
      iD = iArray->GetVoidPointer(0);
      switch (iType)
      {
        svtkTemplateMacro(CreateArrayPair(this, static_cast<SVTK_TT*>(iD), static_cast<SVTK_TT*>(iD),
          numOutPts, iNumComp, iArray, static_cast<SVTK_TT>(nullValue)));
      } // over all SVTK types
    }   // if not excluded
  }     // for each candidate array
}

#endif
