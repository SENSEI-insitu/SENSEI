/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSortFieldData.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkSortFieldData.h"

#include "svtkAbstractArray.h"
#include "svtkFieldData.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"

//-------------------------------------------------------------------------
svtkStandardNewMacro(svtkSortFieldData);

//-------------------------------------------------------------------------
svtkSortFieldData::svtkSortFieldData() = default;

//---------------------------------------------------------------------------
svtkSortFieldData::~svtkSortFieldData() = default;

//-------------------------------------------------------------------------
// Using svtkSortDataArray, it's easy to loop over all of the arrays in the
// field data and sort them. Initially we just need to generate the sort
// indices which are then applied to each array in turn.
svtkIdType* svtkSortFieldData::Sort(
  svtkFieldData* fd, const char* arrayName, int k, int retIndices, int dir)
{
  // Verify the input
  if (fd == nullptr || arrayName == nullptr)
  {
    svtkGenericWarningMacro("SortFieldData needs valid input");
    return nullptr;
  }
  int pos;
  svtkAbstractArray* array = fd->GetAbstractArray(arrayName, pos);
  if (pos < 0)
  {
    svtkGenericWarningMacro("Sorting array not found.");
    return nullptr;
  }
  int numComp = array->GetNumberOfComponents();
  if (k < 0 || k >= numComp)
  {
    svtkGenericWarningMacro("Cannot sort by column "
      << k << " since the array only has columns 0 through " << (numComp - 1));
    return nullptr;
  }
  svtkIdType numKeys = array->GetNumberOfTuples();
  if (numKeys <= 0)
  {
    return nullptr;
  }

  // Create and initialize the sorting indices
  svtkIdType* idx = svtkSortDataArray::InitializeSortIndices(numKeys);

  // Sort and generate the sorting indices
  void* dataIn = array->GetVoidPointer(0);
  int dataType = array->GetDataType();
  svtkSortDataArray::GenerateSortIndices(dataType, dataIn, numKeys, numComp, k, idx);

  // Now loop over all arrays in the field data. Those that are the
  // same length as the sorting indices are processed. Otherwise they
  // are skipped and remain unchanged.
  int nc, numArrays = fd->GetNumberOfArrays();
  for (int arrayNum = 0; arrayNum < numArrays; ++arrayNum)
  {
    array = fd->GetAbstractArray(arrayNum);
    if (array != nullptr && array->GetNumberOfTuples() == numKeys)
    { // process the array
      dataIn = array->GetVoidPointer(0);
      dataType = array->GetDataType();
      nc = array->GetNumberOfComponents();
      svtkSortDataArray::ShuffleArray(idx, dataType, numKeys, nc, array, dataIn, dir);
    }
  }

  // Clean up
  if (retIndices)
  {
    return idx;
  }
  else
  {
    delete[] idx;
    return nullptr;
  }
}

//-------------------------------------------------------------------------
void svtkSortFieldData::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

// svtkSortFieldData methods -------------------------------------------------------
