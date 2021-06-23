/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestSortDataArray.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/*
 * Copyright 2004 Sandia Corporation.
 * Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
 * license for use of this work by or on behalf of the
 * U.S. Government. Redistribution and use in source and binary forms, with
 * or without modification, are permitted provided that this Notice and any
 * statement of authorship are reproduced on all copies.
 */
// -*- c++ -*- *******************************************************

#include "svtkFloatArray.h"
#include "svtkIdList.h"
#include "svtkIntArray.h"
#include "svtkMath.h"
#include "svtkSortDataArray.h"
#include "svtkStringArray.h"
#include "svtkTimerLog.h"

#include <locale> // C++ locale
#include <sstream>

//#define ARRAY_SIZE (2*1024*1024)
#define ARRAY_SIZE 2048

int TestSortDataArray(int, char*[])
{
  svtkIdType i;
  svtkTimerLog* timer = svtkTimerLog::New();
  int retVal = 0;

  //---------------------------------------------------------------------------
  // Sort data array
  cout << "Building array----------" << endl;
  svtkIntArray* keys = svtkIntArray::New();
  keys->SetNumberOfComponents(1);
  keys->SetNumberOfTuples(ARRAY_SIZE);
  for (i = 0; i < ARRAY_SIZE; i++)
  {
    keys->SetComponent(i, 0, static_cast<int>(svtkMath::Random(0, ARRAY_SIZE * 4)));
  }

  cout << "Sorting array" << endl;
  timer->StartTimer();
  svtkSortDataArray::Sort(keys);
  timer->StopTimer();

  cout << "Time to sort array: " << timer->GetElapsedTime() << " sec" << endl;

  for (i = 0; i < ARRAY_SIZE - 1; i++)
  {
    if (keys->GetComponent(i, 0) > keys->GetComponent(i + 1, 0))
    {
      cout << "Array not properly sorted!" << endl;
      retVal = 1;
      break;
    }
  }
  cout << "Array consistency check finished\n" << endl;

  cout << "Sorting sorted array" << endl;
  timer->StartTimer();
  svtkSortDataArray::Sort(keys);
  timer->StopTimer();

  cout << "Time to sort array: " << timer->GetElapsedTime() << " sec" << endl;

  for (i = 0; i < ARRAY_SIZE - 1; i++)
  {
    if (keys->GetComponent(i, 0) > keys->GetComponent(i + 1, 0))
    {
      cout << "Array not properly sorted!" << endl;
      retVal = 1;
      break;
    }
  }
  cout << "Array consistency check finished\n" << endl;

  //---------------------------------------------------------------------------
  // Sort id list (ascending)
  cout << "Building id list (ascending order)----------" << endl;
  svtkIdList* ids = svtkIdList::New();
  ids->SetNumberOfIds(ARRAY_SIZE);
  for (i = 0; i < ARRAY_SIZE; i++)
  {
    ids->SetId(i, static_cast<svtkIdType>(svtkMath::Random(0, ARRAY_SIZE * 4)));
  }

  cout << "Sorting ids" << endl;
  timer->StartTimer();
  svtkSortDataArray::Sort(ids);
  timer->StopTimer();

  cout << "Time to sort ids: " << timer->GetElapsedTime() << " sec" << endl;

  for (i = 0; i < ARRAY_SIZE - 1; i++)
  {
    if (ids->GetId(i) > ids->GetId(i + 1))
    {
      cout << "Id list not properly sorted!" << endl;
      retVal = 1;
      break;
    }
  }
  cout << "Id list consistency check finished\n" << endl;

  //---------------------------------------------------------------------------
  // Sort id list (descending)
  cout << "Building id list (descending order)----------" << endl;
  ids->SetNumberOfIds(ARRAY_SIZE);
  for (i = 0; i < ARRAY_SIZE; i++)
  {
    ids->SetId(i, static_cast<svtkIdType>(svtkMath::Random(0, ARRAY_SIZE * 4)));
  }

  cout << "Sorting ids" << endl;
  timer->StartTimer();
  svtkSortDataArray::Sort(ids, 1);
  timer->StopTimer();

  cout << "Time to sort ids: " << timer->GetElapsedTime() << " sec" << endl;

  for (i = 0; i < ARRAY_SIZE - 1; i++)
  {
    if (ids->GetId(i) < ids->GetId(i + 1))
    {
      cout << "Id list not properly sorted!" << endl;
      retVal = 1;
      break;
    }
  }
  cout << "Id list consistency check finished\n" << endl;

  //---------------------------------------------------------------------------
  // Sort key/value pairs
  cout << "Building key/value arrays----------\n" << endl;
  svtkIntArray* values = svtkIntArray::New();
  values->SetNumberOfComponents(2);
  values->SetNumberOfTuples(ARRAY_SIZE);
  for (i = 0; i < ARRAY_SIZE; i++)
  {
    keys->SetComponent(i, 0, static_cast<int>(svtkMath::Random(0, ARRAY_SIZE * 4)));
    values->SetComponent(i, 0, i);
    values->SetComponent(i, 1, static_cast<int>(svtkMath::Random(0, ARRAY_SIZE * 4)));
  }
  svtkIntArray* saveKeys = svtkIntArray::New();
  saveKeys->DeepCopy(keys);
  svtkIntArray* saveValues = svtkIntArray::New();
  saveValues->DeepCopy(values);

  cout << "Sorting arrays" << endl;
  timer->StartTimer();
  svtkSortDataArray::Sort(keys, values);
  timer->StopTimer();

  cout << "Time to sort array: " << timer->GetElapsedTime() << " sec" << endl;

  for (i = 0; i < ARRAY_SIZE - 1; i++)
  {
    int lookup = static_cast<int>(values->GetComponent(i, 0));
    if (keys->GetComponent(i, 0) > keys->GetComponent(i + 1, 0))
    {
      cout << "Array not properly sorted!" << endl;
      retVal = 1;
      break;
    }
    if (keys->GetComponent(i, 0) != saveKeys->GetComponent(lookup, 0))
    {
      cout << "Values array not consistent with keys array!" << endl;
      retVal = 1;
      break;
    }
    if (values->GetComponent(i, 1) != saveValues->GetComponent(lookup, 1))
    {
      cout << "Values array not consistent with keys array!" << endl;
      retVal = 1;
      break;
    }
  }
  cout << "Array consistency check finished\n" << endl;

  cout << "Sorting sorted arrays" << endl;
  timer->StartTimer();
  svtkSortDataArray::Sort(keys, values);
  timer->StopTimer();

  cout << "Time to sort array: " << timer->GetElapsedTime() << " sec" << endl;

  for (i = 0; i < ARRAY_SIZE - 1; i++)
  {
    int lookup = static_cast<int>(values->GetComponent(i, 0));
    if (keys->GetComponent(i, 0) > keys->GetComponent(i + 1, 0))
    {
      cout << "Array not properly sorted!" << endl;
      retVal = 1;
      break;
    }
    if (keys->GetComponent(i, 0) != saveKeys->GetComponent(lookup, 0))
    {
      cout << "Values array not consistent with keys array!" << endl;
      retVal = 1;
      break;
    }
    if (values->GetComponent(i, 1) != saveValues->GetComponent(lookup, 1))
    {
      cout << "Values array not consistent with keys array!" << endl;
      retVal = 1;
      break;
    }
  }
  cout << "Array consistency check finished\n" << endl;

  //---------------------------------------------------------------------------
  // Sort data array on component value pairs
  cout << "Building data array----------\n" << endl;
  svtkFloatArray* fvalues = svtkFloatArray::New();
  fvalues->SetNumberOfComponents(3);
  fvalues->SetNumberOfTuples(ARRAY_SIZE);
  for (i = 0; i < ARRAY_SIZE; i++)
  {
    fvalues->SetComponent(i, 0, i);
    fvalues->SetComponent(i, 1, static_cast<float>(svtkMath::Random(0, ARRAY_SIZE * 4)));
    fvalues->SetComponent(i, 2, i);
  }
  svtkFloatArray* saveFValues = svtkFloatArray::New();
  saveFValues->DeepCopy(fvalues);

  cout << "Sorting data array with component #1" << endl;
  timer->StartTimer();
  svtkSortDataArray::SortArrayByComponent(fvalues, 1);
  timer->StopTimer();

  cout << "Time to sort data array: " << timer->GetElapsedTime() << " sec" << endl;

  for (i = 0; i < ARRAY_SIZE - 1; i++)
  {
    if (fvalues->GetComponent(i, 1) > fvalues->GetComponent(i + 1, 1))
    {
      cout << "Data array sorted incorrectly!" << endl;
      retVal = 1;
      break;
    }
    if (fvalues->GetComponent(i, 0) != fvalues->GetComponent(i, 2))
    {
      cout << "Data array tuples inconsistent!" << endl;
      retVal = 1;
      break;
    }
  }
  cout << "Data array consistency check finished\n" << endl;

  //---------------------------------------------------------------------------
  // Sort string array
  std::ostringstream ostr;
  ostr.imbue(std::locale::classic());
  cout << "Building string array----------\n" << endl;
  svtkStringArray* sarray = svtkStringArray::New();
  sarray->SetNumberOfTuples(ARRAY_SIZE);
  for (i = 0; i < ARRAY_SIZE; ++i)
  {
    ostr.str(""); // clear it out
    ostr << static_cast<int>(svtkMath::Random(0, ARRAY_SIZE * 4));
    sarray->SetValue(i, ostr.str());
  }

  cout << "Sorting string array" << endl;
  timer->StartTimer();
  svtkSortDataArray::Sort(sarray, 1);
  timer->StopTimer();
  cout << "Time to sort strings: " << timer->GetElapsedTime() << " sec" << endl;

  svtkStdString s1, s2;
  for (i = 0; i < ARRAY_SIZE - 1; ++i)
  {
    // s1 = std::stoi(sarray->GetValue(i));
    // s2 = std::stoi(sarray->GetValue(i+1));
    s1 = sarray->GetValue(i);
    s2 = sarray->GetValue(i + 1);
    if (s1 < s2)
    {
      cout << "String array sorted incorrectly!" << endl;
      retVal = 1;
      break;
    }
  }
  cout << "String array consistency check finished\n" << endl;

  timer->Delete();
  keys->Delete();
  ids->Delete();
  values->Delete();
  fvalues->Delete();
  saveKeys->Delete();
  saveValues->Delete();
  saveFValues->Delete();
  sarray->Delete();

  return retVal;
}
