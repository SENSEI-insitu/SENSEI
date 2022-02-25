/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestDataArrayIterators.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkFloatArray.h"
#include "svtkNew.h"
#include "svtkTimerLog.h"
#include "svtkTypedDataArray.h"
#include "svtkTypedDataArrayIterator.h"

#include <cassert>
#include <iostream>

// undefine this to print benchmark results:
#define SILENT

// Create a subclass of svtkTypedDataArray:
namespace
{
class MyArray : public svtkTypedDataArray<float>
{
  svtkFloatArray* Data;

public:
  static MyArray* New() { SVTK_STANDARD_NEW_BODY(MyArray); }
  void Init(svtkFloatArray* array)
  {
    this->Data = array;
    this->NumberOfComponents = array->GetNumberOfComponents();
    this->MaxId = array->GetMaxId();
  }
  ValueType& GetValueReference(svtkIdType idx) override { return *this->Data->GetPointer(idx); }

  // These pure virtuals are no-op -- all we care about is GetValueReference
  // to test the iterator.
  void SetTypedTuple(svtkIdType, const ValueType*) override {}
  void InsertTypedTuple(svtkIdType, const ValueType*) override {}
  svtkIdType InsertNextTypedTuple(const ValueType*) override { return 0; }
  svtkIdType LookupTypedValue(ValueType) override { return 0; }
  void LookupTypedValue(ValueType, svtkIdList*) override {}
  ValueType GetValue(svtkIdType) const override { return 0; }
  void SetValue(svtkIdType, ValueType) override {}
  void GetTypedTuple(svtkIdType, ValueType*) const override {}
  svtkIdType InsertNextValue(ValueType) override { return 0; }
  void InsertValue(svtkIdType, ValueType) override {}
  svtkTypeBool Allocate(svtkIdType, svtkIdType) override { return 0; }
  svtkTypeBool Resize(svtkIdType) override { return 0; }
};
}

int TestDataArrayIterators(int, char*[])
{
  svtkIdType numComps = 4;
  svtkIdType numValues = 100000000; // 10 million
  assert(numValues % numComps == 0);
  svtkIdType numTuples = numValues / numComps;

  svtkNew<svtkFloatArray> arrayContainer;
  svtkFloatArray* array = arrayContainer;
  array->SetNumberOfComponents(numComps);
  array->SetNumberOfTuples(numTuples);
  for (svtkIdType i = 0; i < numValues; ++i)
  {
    // Just fill with consistent data
    array->SetValue(i, i % 97);
  }

  // Create the svtkTypedDataArray testing implementation:
  svtkNew<MyArray> tdaContainer;
  MyArray* tda = tdaContainer;
  tda->Init(array);

  // should be svtkAOSDataArrayTemplate<float>::Iterator (float*):
  svtkFloatArray::Iterator datBegin = array->Begin();
  svtkFloatArray::Iterator datIter = array->Begin();
  if (typeid(datBegin) != typeid(float*))
  {
    std::cerr << "Error: svtkFloatArray::Iterator is not a float*.";
    return EXIT_FAILURE;
  }

  // should be svtkTypedDataArrayIterator<float>:
  svtkTypedDataArray<float>::Iterator tdaBegin =
    svtkTypedDataArray<float>::FastDownCast(tda)->Begin();
  svtkTypedDataArray<float>::Iterator tdaIter = svtkTypedDataArray<float>::FastDownCast(tda)->Begin();
  if (typeid(tdaBegin) != typeid(svtkTypedDataArrayIterator<float>))
  {
    std::cerr << "Error: svtkTypedDataArray<float>::Iterator is not a "
                 "svtkTypedDataArrayIterator<float>.";
    return EXIT_FAILURE;
  }

  // Validate that the iterators return the same values from operator[] and
  // operator* as GetValue;
  for (svtkIdType i = 0; i < numValues; ++i)
  {
    float lookup = array->GetValue(i);
    if (lookup != datBegin[i] || lookup != tdaBegin[i] || lookup != *datIter || lookup != *tdaIter)
    {
      std::cerr << "Mismatch at " << i << ":"
                << " GetValue(i)=" << lookup << " datBegin[i]=" << datBegin[i]
                << " tdaBegin[i]=" << tdaBegin[i] << " *datIter=" << *datIter
                << " *tdaIter=" << *tdaIter << std::endl;
      return EXIT_FAILURE;
    }
    ++datIter;
    ++tdaIter;
  }

#ifndef SILENT
  // Iterator timings.
  svtkNew<svtkTimerLog> timer;

  // Lookup:
  float lookupSum = 0.f;
  timer->StartTimer();
  for (svtkIdType i = 0; i < numValues; ++i)
  {
    lookupSum += *array->GetPointer(i);
  }
  timer->StopTimer();
  double lookupTime = timer->GetElapsedTime();

  // Scalar iterator:
  float datSum = 0.f;
  timer->StartTimer();
  svtkFloatArray::Iterator datEnd = array->End();
  while (datBegin != datEnd)
  {
    datSum += *datBegin++;
  }
  timer->StopTimer();
  double datTime = timer->GetElapsedTime();

  // svtkTypedDataArrayIterator:
  svtkTypedDataArray<float>::Iterator tdaEnd = tda->End();
  float tdaSum = 0.f;
  timer->StartTimer();
  while (tdaBegin != tdaEnd)
  {
    tdaSum += *tdaBegin++;
  }
  timer->StopTimer();
  double tdaTime = timer->GetElapsedTime();

  std::cout << "GetValue time, sum: " << lookupTime << ", " << lookupSum << std::endl;
  std::cout << "dat time, sum:      " << datTime << ", " << datSum << std::endl;
  std::cout << "tda time, sum:      " << tdaTime << ", " << tdaSum << std::endl;
#endif

  return EXIT_SUCCESS;
}
