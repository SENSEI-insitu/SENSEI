/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestArrayLookup.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkBitArray.h"
#include "svtkFloatArray.h"
#include "svtkIdList.h"
#include "svtkIdTypeArray.h"
#include "svtkIntArray.h"
#include "svtkNew.h"
#include "svtkSortDataArray.h"
#include "svtkStringArray.h"
#include "svtkTimerLog.h"
#include "svtkVariantArray.h"

#include "svtkSmartPointer.h"
#define SVTK_CREATE(type, name) svtkSmartPointer<type> name = svtkSmartPointer<type>::New()

#include <algorithm>
#include <limits>
#include <map>
#include <utility>
#include <vector>

struct NodeCompare
{
  bool operator()(const std::pair<int, svtkIdType>& a, const std::pair<int, svtkIdType>& b) const
  {
    return a.first < b.first;
  }
};

svtkIdType LookupValue(std::multimap<int, svtkIdType>& lookup, int value)
{
  std::pair<const int, svtkIdType> found = *lookup.lower_bound(value);
  if (found.first == value)
  {
    return found.second;
  }
  return -1;
}

svtkIdType LookupValue(std::vector<std::pair<int, svtkIdType> >& lookup, int value)
{
  NodeCompare comp;
  std::pair<int, svtkIdType> val(value, 0);
  std::pair<int, svtkIdType> found = *std::lower_bound(lookup.begin(), lookup.end(), val, comp);
  if (found.first == value)
  {
    return found.second;
  }
  return -1;
}

svtkIdType LookupValue(svtkIntArray* lookup, svtkIdTypeArray* index, int value)
{
  int* ptr = lookup->GetPointer(0);
  svtkIdType place =
    static_cast<svtkIdType>(std::lower_bound(ptr, ptr + lookup->GetNumberOfTuples(), value) - ptr);
  if (place < lookup->GetNumberOfTuples() && ptr[place] == value)
  {
    return index->GetValue(place);
  }
  return -1;
}

int TestArrayLookupBit(svtkIdType numVal)
{
  int errors = 0;

  // Create the array
  svtkIdType arrSize = (numVal - 1) * numVal / 2;
  SVTK_CREATE(svtkBitArray, arr);
  for (svtkIdType i = 0; i < arrSize; i++)
  {
    arr->InsertNextValue(i < arrSize / 2);
  }

  //
  // Test lookup implemented inside data array
  //

  // Time the lookup creation
  SVTK_CREATE(svtkTimerLog, timer);
  timer->StartTimer();
  arr->LookupValue(0);
  timer->StopTimer();
  cerr << "," << timer->GetElapsedTime();

  // Time simple lookup
  timer->StartTimer();
  for (svtkIdType i = 0; i < numVal; i++)
  {
    arr->LookupValue(i % 2);
  }
  timer->StopTimer();
  cerr << "," << (timer->GetElapsedTime() / static_cast<double>(numVal));

  // Time list lookup
  SVTK_CREATE(svtkIdList, list);
  timer->StartTimer();
  for (svtkIdType i = 0; i < numVal; i++)
  {
    arr->LookupValue(i % 2, list);
  }
  timer->StopTimer();
  cerr << "," << (timer->GetElapsedTime() / static_cast<double>(numVal));

  // Test for correctness (-1)
  svtkIdType index = -1;
  index = arr->LookupValue(-1);
  if (index != -1)
  {
    cerr << "ERROR: lookup found value at " << index << " but is not there (should return -1)"
         << endl;
    errors++;
  }
  arr->LookupValue(-1, list);
  if (list->GetNumberOfIds() != 0)
  {
    cerr << "ERROR: lookup found " << list->GetNumberOfIds() << " matches but there should be " << 0
         << endl;
    errors++;
  }

  // Test for correctness (0)
  index = arr->LookupValue(0);
  if (index < arrSize / 2 || index > arrSize - 1)
  {
    cerr << "ERROR: vector lookup found value at " << index << " but is in range [" << arrSize / 2
         << "," << arrSize - 1 << "]" << endl;
    errors++;
  }
  arr->LookupValue(0, list);
  if (list->GetNumberOfIds() != arrSize - arrSize / 2)
  {
    cerr << "ERROR: lookup found " << list->GetNumberOfIds() << " matches but there should be "
         << arrSize - arrSize / 2 << endl;
    errors++;
  }
  else
  {
    for (svtkIdType j = 0; j < list->GetNumberOfIds(); j++)
    {
      if (arr->GetValue(list->GetId(j)) != 0)
      {
        cerr << "ERROR: could not find " << j << " in found list" << endl;
        errors++;
      }
    }
  }

  // Test for correctness (1)
  index = arr->LookupValue(1);
  if (index < 0 || index > arrSize / 2 - 1)
  {
    cerr << "ERROR: vector lookup found value at " << index << " but is in range [" << 0 << ","
         << arrSize / 2 - 1 << "]" << endl;
    errors++;
  }
  arr->LookupValue(1, list);
  if (list->GetNumberOfIds() != arrSize / 2)
  {
    cerr << "ERROR: lookup found " << list->GetNumberOfIds() << " matches but there should be "
         << arrSize / 2 << endl;
    errors++;
  }
  else
  {
    for (svtkIdType j = 0; j < list->GetNumberOfIds(); j++)
    {
      if (arr->GetValue(list->GetId(j)) != 1)
      {
        cerr << "ERROR: could not find " << j << " in found list" << endl;
        errors++;
      }
    }
  }

  return errors;
}

int TestArrayLookupVariant(svtkIdType numVal)
{
  int errors = 0;

  // Create the array
  svtkIdType arrSize = (numVal - 1) * numVal / 2;
  SVTK_CREATE(svtkVariantArray, arr);
  for (svtkIdType i = 0; i < numVal; i++)
  {
    for (svtkIdType j = 0; j < numVal - 1 - i; j++)
    {
      arr->InsertNextValue(numVal - 1 - i);
    }
  }

  //
  // Test lookup implemented inside data array
  //

  // Time the lookup creation
  SVTK_CREATE(svtkTimerLog, timer);
  timer->StartTimer();
  arr->LookupValue(0);
  timer->StopTimer();
  cerr << "," << timer->GetElapsedTime();

  // Time simple lookup
  timer->StartTimer();
  for (svtkIdType i = 0; i < numVal; i++)
  {
    arr->LookupValue(i);
  }
  timer->StopTimer();
  cerr << "," << (timer->GetElapsedTime() / static_cast<double>(numVal));

  // Time list lookup
  SVTK_CREATE(svtkIdList, list);
  timer->StartTimer();
  for (svtkIdType i = 0; i < numVal; i++)
  {
    arr->LookupValue(i, list);
  }
  timer->StopTimer();
  cerr << "," << (timer->GetElapsedTime() / static_cast<double>(numVal));

  // Test for correctness
  svtkIdType correctIndex = arrSize;
  for (svtkIdType i = 0; i < numVal; i++)
  {
    correctIndex -= i;
    svtkIdType index = arr->LookupValue(i);
    if (i == 0 && index != -1)
    {
      cerr << "ERROR: lookup found value at " << index << " but is at -1" << endl;
      errors++;
    }
    if (i != 0 && (index < correctIndex || index > correctIndex + i - 1))
    {
      cerr << "ERROR: vector lookup found value at " << index << " but is in range ["
           << correctIndex << "," << correctIndex + i - 1 << "]" << endl;
      errors++;
    }
    arr->LookupValue(i, list);
    if (list->GetNumberOfIds() != i)
    {
      cerr << "ERROR: lookup found " << list->GetNumberOfIds() << " matches but there should be "
           << i << endl;
      errors++;
    }
    else
    {
      for (svtkIdType j = correctIndex; j < correctIndex + i; j++)
      {
        bool inList = false;
        for (svtkIdType k = 0; k < i; ++k)
        {
          if (list->GetId(k) == j)
          {
            inList = true;
            break;
          }
        }
        if (!inList)
        {
          cerr << "ERROR: could not find " << j << " in found list" << endl;
          errors++;
        }
      }
    }
  }
  return errors;
}

int TestArrayLookupFloat(svtkIdType numVal)
{
  int errors = 0;

  // Create the array
  svtkIdType arrSize = (numVal - 1) * numVal / 2;
  SVTK_CREATE(svtkFloatArray, arr);
  for (svtkIdType i = 0; i < numVal; i++)
  {
    for (svtkIdType j = 0; j < numVal - 1 - i; j++)
    {
      arr->InsertNextValue(numVal - 1 - i);
    }
  }
  arr->InsertNextValue(std::numeric_limits<float>::quiet_NaN());

  //
  // Test lookup implemented inside data array
  //

  // Time the lookup creation
  SVTK_CREATE(svtkTimerLog, timer);
  timer->StartTimer();
  arr->LookupValue(0);
  timer->StopTimer();
  cerr << "," << timer->GetElapsedTime();

  // Time simple lookup
  timer->StartTimer();
  for (svtkIdType i = 0; i < numVal; i++)
  {
    arr->LookupValue(i);
  }
  timer->StopTimer();
  cerr << "," << (timer->GetElapsedTime() / static_cast<double>(numVal));

  // Time list lookup
  SVTK_CREATE(svtkIdList, list);
  timer->StartTimer();
  for (svtkIdType i = 0; i < numVal; i++)
  {
    arr->LookupValue(i, list);
  }
  timer->StopTimer();
  cerr << "," << (timer->GetElapsedTime() / static_cast<double>(numVal));

  // Test for NaN
  {
    svtkIdType index = arr->LookupValue(std::numeric_limits<float>::quiet_NaN());
    if (index != arrSize)
    {
      cerr << "ERROR: lookup found NaN at " << index << " instead of " << arrSize << endl;
      errors++;
    }
  }
  {
    svtkNew<svtkIdList> NaNlist;
    arr->LookupValue(std::numeric_limits<float>::quiet_NaN(), NaNlist);
    if (NaNlist->GetNumberOfIds() != 1)
    {
      cerr << "ERROR: lookup found " << list->GetNumberOfIds() << " values of NaN instead of " << 1
           << endl;
      errors++;
    }
    if (NaNlist->GetId(0) != arrSize)
    {
      cerr << "ERROR: lookup found NaN at " << list->GetId(0) << " instead of " << arrSize << endl;
      errors++;
    }
  }

  // Test for correctness
  svtkIdType correctIndex = arrSize;
  for (svtkIdType i = 0; i < numVal; i++)
  {
    correctIndex -= i;
    svtkIdType index = arr->LookupValue(i);
    if (i == 0 && index != -1)
    {
      cerr << "ERROR: lookup found value at " << index << " but is at -1" << endl;
      errors++;
    }
    if (i != 0 && (index < correctIndex || index > correctIndex + i - 1))
    {
      cerr << "ERROR: vector lookup found value at " << index << " but is in range ["
           << correctIndex << "," << correctIndex + i - 1 << "]" << endl;
      errors++;
    }
    arr->LookupValue(i, list);
    if (list->GetNumberOfIds() != i)
    {
      cerr << "ERROR: lookup found " << list->GetNumberOfIds() << " matches but there should be "
           << i << endl;
      errors++;
    }
    else
    {
      for (svtkIdType j = correctIndex; j < correctIndex + i; j++)
      {
        bool inList = false;
        for (svtkIdType k = 0; k < i; ++k)
        {
          if (list->GetId(k) == j)
          {
            inList = true;
            break;
          }
        }
        if (!inList)
        {
          cerr << "ERROR: could not find " << j << " in found list" << endl;
          errors++;
        }
      }
    }
  }
  return errors;
}

int TestArrayLookupString(svtkIdType numVal)
{
  int errors = 0;

  // Create the array
  svtkIdType arrSize = (numVal - 1) * numVal / 2;
  SVTK_CREATE(svtkStringArray, arr);
  for (svtkIdType i = 0; i < numVal; i++)
  {
    for (svtkIdType j = 0; j < numVal - 1 - i; j++)
    {
      arr->InsertNextValue(svtkVariant(numVal - 1 - i).ToString());
    }
  }

  //
  // Test lookup implemented inside data array
  //

  // Time the lookup creation
  SVTK_CREATE(svtkTimerLog, timer);
  timer->StartTimer();
  arr->LookupValue("0");
  timer->StopTimer();
  cerr << "," << timer->GetElapsedTime();

  // Time simple lookup
  timer->StartTimer();
  for (svtkIdType i = 0; i < numVal; i++)
  {
    arr->LookupValue(svtkVariant(i).ToString());
  }
  timer->StopTimer();
  cerr << "," << (timer->GetElapsedTime() / static_cast<double>(numVal));

  // Time list lookup
  SVTK_CREATE(svtkIdList, list);
  timer->StartTimer();
  for (svtkIdType i = 0; i < numVal; i++)
  {
    arr->LookupValue(svtkVariant(i).ToString(), list);
  }
  timer->StopTimer();
  cerr << "," << (timer->GetElapsedTime() / static_cast<double>(numVal));

  // Test for correctness
  svtkIdType correctIndex = arrSize;
  for (svtkIdType i = 0; i < numVal; i++)
  {
    correctIndex -= i;
    svtkIdType index = arr->LookupValue(svtkVariant(i).ToString());
    if (i == 0 && index != -1)
    {
      cerr << "ERROR: lookup found value at " << index << " but is at -1" << endl;
      errors++;
    }
    if (i != 0 && (index < correctIndex || index > correctIndex + i - 1))
    {
      cerr << "ERROR: vector lookup found value at " << index << " but is in range ["
           << correctIndex << "," << correctIndex + i - 1 << "]" << endl;
      errors++;
    }
    arr->LookupValue(svtkVariant(i).ToString(), list);
    if (list->GetNumberOfIds() != i)
    {
      cerr << "ERROR: lookup found " << list->GetNumberOfIds() << " matches but there should be "
           << i << endl;
      errors++;
    }
    else
    {
      for (svtkIdType j = correctIndex; j < correctIndex + i; j++)
      {
        bool inList = false;
        for (svtkIdType k = 0; k < i; ++k)
        {
          if (list->GetId(k) == j)
          {
            inList = true;
            break;
          }
        }
        if (!inList)
        {
          cerr << "ERROR: could not find " << j << " in found list" << endl;
          errors++;
        }
      }
    }
  }
  return errors;
}

int TestArrayLookupInt(svtkIdType numVal, bool runComparison)
{
  int errors = 0;

  // Create the array
  svtkIdType arrSize = (numVal - 1) * numVal / 2;
  SVTK_CREATE(svtkIntArray, arr);
  for (svtkIdType i = 0; i < numVal; i++)
  {
    for (svtkIdType j = 0; j < numVal - 1 - i; j++)
    {
      arr->InsertNextValue(numVal - 1 - i);
    }
  }

  //
  // Test lookup implemented inside data array
  //

  // Time the lookup creation
  SVTK_CREATE(svtkTimerLog, timer);
  timer->StartTimer();
  arr->LookupValue(0);
  timer->StopTimer();
  cerr << "," << timer->GetElapsedTime();

  // Time simple lookup
  timer->StartTimer();
  for (svtkIdType i = 0; i < numVal; i++)
  {
    arr->LookupValue(i);
  }
  timer->StopTimer();
  cerr << "," << (timer->GetElapsedTime() / static_cast<double>(numVal));

  // Time list lookup
  SVTK_CREATE(svtkIdList, list);
  timer->StartTimer();
  for (svtkIdType i = 0; i < numVal; i++)
  {
    arr->LookupValue(i, list);
  }
  timer->StopTimer();
  cerr << "," << (timer->GetElapsedTime() / static_cast<double>(numVal));

  // Test for correctness
  svtkIdType correctIndex = arrSize;
  for (svtkIdType i = 0; i < numVal; i++)
  {
    correctIndex -= i;
    svtkIdType index = arr->LookupValue(i);
    if (i == 0 && index != -1)
    {
      cerr << "ERROR: lookup found value at " << index << " but is at -1" << endl;
      errors++;
    }
    if (i != 0 && (index < correctIndex || index > correctIndex + i - 1))
    {
      cerr << "ERROR: vector lookup found value at " << index << " but is in range ["
           << correctIndex << "," << correctIndex + i - 1 << "]" << endl;
      errors++;
    }
    arr->LookupValue(i, list);
    if (list->GetNumberOfIds() != i)
    {
      cerr << "ERROR: lookup found " << list->GetNumberOfIds() << " matches but there should be "
           << i << endl;
      errors++;
    }
    else
    {
      for (svtkIdType j = correctIndex; j < correctIndex + i; j++)
      {
        bool inList = false;
        for (svtkIdType k = 0; k < i; ++k)
        {
          if (list->GetId(k) == j)
          {
            inList = true;
            break;
          }
        }
        if (!inList)
        {
          cerr << "ERROR: could not find " << j << " in found list" << endl;
          errors++;
        }
      }
    }
  }

  if (runComparison)
  {
    //
    // Test STL map lookup
    //

    // Time the lookup creation
    timer->StartTimer();
    int* ptr = arr->GetPointer(0);
    std::multimap<int, svtkIdType> map;
    for (svtkIdType i = 0; i < arrSize; ++i, ++ptr)
    {
      map.insert(std::pair<const int, svtkIdType>(*ptr, i));
    }
    timer->StopTimer();
    cerr << "," << timer->GetElapsedTime();

    // Time simple lookup
    timer->StartTimer();
    for (svtkIdType i = 0; i < numVal; i++)
    {
      LookupValue(map, i);
    }
    timer->StopTimer();
    cerr << "," << (timer->GetElapsedTime() / static_cast<double>(numVal));

    // Test for correctness
    correctIndex = arrSize;
    for (svtkIdType i = 0; i < numVal; i++)
    {
      correctIndex -= i;
      svtkIdType index = LookupValue(map, i);
      if (i == 0 && index != -1)
      {
        cerr << "ERROR: lookup found value at " << index << " but is at -1" << endl;
        errors++;
      }
      if (i != 0 && index != correctIndex)
      {
        cerr << "ERROR: lookup found value at " << index << " but is at " << correctIndex << endl;
        errors++;
      }
    }

    //
    // Test STL vector lookup
    //

    // Time lookup creation
    timer->StartTimer();
    ptr = arr->GetPointer(0);
    std::vector<std::pair<int, svtkIdType> > vec(arrSize);
    for (svtkIdType i = 0; i < arrSize; ++i, ++ptr)
    {
      vec[i] = std::pair<int, svtkIdType>(*ptr, i);
    }
    NodeCompare comp;
    std::sort(vec.begin(), vec.end(), comp);
    timer->StopTimer();
    cerr << "," << timer->GetElapsedTime();

    // Time simple lookup
    timer->StartTimer();
    for (svtkIdType i = 0; i < numVal; i++)
    {
      LookupValue(vec, i);
    }
    timer->StopTimer();
    cerr << "," << (timer->GetElapsedTime() / static_cast<double>(numVal));

    // Test for correctness
    correctIndex = arrSize;
    for (svtkIdType i = 0; i < numVal; i++)
    {
      correctIndex -= i;
      svtkIdType index = LookupValue(vec, i);
      if (i == 0 && index != -1)
      {
        cerr << "ERROR: vector lookup found value at " << index << " but is at -1" << endl;
        errors++;
      }
      if (i != 0 && (index < correctIndex || index > correctIndex + i - 1))
      {
        cerr << "ERROR: vector lookup found value at " << index << " but is in range ["
             << correctIndex << "," << correctIndex + i - 1 << "]" << endl;
        errors++;
      }
    }

    //
    // Test sorted data array lookup
    //

    // Time lookup creation
    timer->StartTimer();
    SVTK_CREATE(svtkIdTypeArray, indices);
    indices->SetNumberOfTuples(arrSize);
    svtkIdType* keyptr = indices->GetPointer(0);
    for (svtkIdType i = 0; i < arrSize; ++i, ++keyptr)
    {
      *keyptr = i;
    }
    SVTK_CREATE(svtkIntArray, sorted);
    sorted->DeepCopy(arr);
    svtkSortDataArray::Sort(sorted, indices);
    timer->StopTimer();
    cerr << "," << timer->GetElapsedTime();

    // Time simple lookup
    timer->StartTimer();
    for (svtkIdType i = 0; i < numVal; i++)
    {
      LookupValue(sorted, indices, i);
    }
    timer->StopTimer();
    cerr << "," << (timer->GetElapsedTime() / static_cast<double>(numVal));

    // Test for correctness
    correctIndex = arrSize;
    for (svtkIdType i = 0; i < numVal; i++)
    {
      correctIndex -= i;
      svtkIdType index = LookupValue(sorted, indices, i);
      if (i == 0 && index != -1)
      {
        cerr << "ERROR: arr lookup found value at " << index << " but is at -1" << endl;
        errors++;
      }
      if (i != 0 && (index < correctIndex || index > correctIndex + i - 1))
      {
        cerr << "ERROR: arr lookup found value at " << index << " but is in range [" << correctIndex
             << "," << correctIndex + i - 1 << "]" << endl;
        errors++;
      }
    }
  }

  return errors;
}

int TestMultiComponent()
{
  int errors = 0;
  auto array = svtkSmartPointer<svtkFloatArray>::New();
  array->SetNumberOfComponents(3);
  static const float data[3][3] = { { 1., 2., 3. }, { 2., 3., 4. }, { 3., 4., 5. } };
  for (auto tuple : data)
  {
    array->InsertNextTypedTuple(tuple);
  }

  // a list of values and the index expected to be returned
  static const int expected[][2] = { { 2, 1 }, { 3, 2 }, { 4, 5 }, { 5, 8 }, { 6, -1 } };
  for (auto e : expected)
  {
    svtkIdType index = array->LookupTypedValue(e[0]);
    if (index != e[1])
    {
      cerr << "TestMultiComponent: "
           << "index of " << e[0] << " expected " << e[1] << " actual " << index;
      ++errors;
    }
  }

  // overwrite 3.0 (3rd component of 1st tuple) with NaN.
  array->SetTypedComponent(0, 2, std::numeric_limits<float>::quiet_NaN());

  // We need to trigger rebuilding the auxiliary data structures explicitly
  array->ClearLookup();
  svtkIdType index = array->LookupValue(std::numeric_limits<float>::quiet_NaN());
  if (2 != index)
  {
    cerr << "TestMultiComponent: lookup of NaN: "
         << "expected 0 actual " << index;
    ++errors;
  }
  index = array->LookupValue(3.);
  if (4 != index)
  {
    cerr << "TestMultiComponent: lookup of value 3.: "
         << "expected 1 actual " << index;
    ++errors;
  }
  return errors;
}

int TestArrayLookup(int argc, char* argv[])
{
  svtkIdType min = 100;
  svtkIdType max = 200;
  int steps = 2;
  bool runComparison = false;
  for (int i = 0; i < argc; ++i)
  {
    if (!strcmp(argv[i], "-C"))
    {
      runComparison = true;
    }
    if (!strcmp(argv[i], "-m") && i + 1 < argc)
    {
      ++i;
      int size = atoi(argv[i]);
      min = static_cast<int>((-1.0 + sqrt(1 + 8.0 * size)) / 2.0);
    }
    if (!strcmp(argv[i], "-M") && i + 1 < argc)
    {
      ++i;
      int size = atoi(argv[i]);
      max = static_cast<int>((-1.0 + sqrt(1 + 8.0 * size)) / 2.0);
    }
    if (!strcmp(argv[i], "-S") && i + 1 < argc)
    {
      ++i;
      steps = atoi(argv[i]);
    }
  }

  svtkIdType stepSize = (max - min) / (steps - 1);
  if (stepSize <= 0)
  {
    stepSize = 1;
  }

  int errors = 0;
  cerr << "distinct values";
  cerr << ",size";
  cerr << ",create lookup";
  cerr << ",index lookup";
  cerr << ",list lookup";
  if (runComparison)
  {
    cerr << ",create map lookup";
    cerr << ",index map lookup";
    cerr << ",create vector lookup";
    cerr << ",index vector lookup";
    cerr << ",create array lookup";
    cerr << ",index array lookup";
  }
  cerr << ",string create lookup";
  cerr << ",string index lookup";
  cerr << ",string list lookup";
  cerr << ",variant create lookup";
  cerr << ",variant index lookup";
  cerr << ",variant list lookup";
  cerr << ",bit create lookup";
  cerr << ",bit index lookup";
  cerr << ",bit list lookup";
  cerr << endl;
  for (svtkIdType numVal = min; numVal <= max; numVal += stepSize)
  {
    svtkIdType total = numVal * (numVal + 1) / 2;
    cerr << numVal << "," << total;
    errors += TestArrayLookupInt(numVal, runComparison);
    errors += TestArrayLookupFloat(numVal);
    errors += TestArrayLookupString(numVal);
    errors += TestArrayLookupVariant(numVal);
    errors += TestArrayLookupBit(numVal);
    cerr << endl;
  }
  errors += TestMultiComponent();
  return errors;
}
