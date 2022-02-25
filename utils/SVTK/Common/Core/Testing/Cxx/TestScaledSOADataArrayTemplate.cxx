/*==============================================================================

  Program:   Visualization Toolkit
  Module:    TestScaledSOADataArrayTemplate.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

==============================================================================*/
#include "svtkMathUtilities.h"
#include "svtkScaledSOADataArrayTemplate.h"

// Needed for portable setenv on MSVC...
#include "svtksys/SystemTools.hxx"

int TestScaledSOADataArrayTemplate(int, char*[])
{
  const int numValues = 5;
  const double trueFirstData[numValues] = { 0, 1, 2, 3, 4 };
  const double trueSecondData[numValues] = { 10, 11, 12, 13, 14 };
  double firstData[numValues];
  double secondData[numValues];
  for (int i = 0; i < 5; i++)
  {
    firstData[i] = trueFirstData[i];
    secondData[i] = trueSecondData[i];
  }

  svtkScaledSOADataArrayTemplate<double>* array = svtkScaledSOADataArrayTemplate<double>::New();
  array->SetNumberOfComponents(2);
  array->SetNumberOfTuples(numValues);
  array->SetArray(0, firstData, numValues, false, true);
  array->SetArray(1, secondData, numValues, false, true);
  array->SetScale(2.);

  // first check that we get twice the values that are stored in firstData and secondData
  // returned by GetTypedTuple()
  double vals[2];
  for (svtkIdType i = 0; i < array->GetNumberOfTuples(); i++)
  {
    array->GetTypedTuple(i, vals);
    if (!svtkMathUtilities::NearlyEqual(vals[0], trueFirstData[i] * array->GetScale()) ||
      !svtkMathUtilities::NearlyEqual(vals[1], trueSecondData[i] * array->GetScale()))
    {
      svtkGenericWarningMacro("Incorrect values returned from scaled array");
      return 1;
    }
  }

  // second check that if we set information based on firstData and secondData
  // that we get that back
  for (svtkIdType i = 0; i < array->GetNumberOfTuples(); i++)
  {
    vals[0] = trueFirstData[i];
    vals[1] = trueSecondData[i];
    array->SetTypedTuple(i, vals);
    array->GetTypedTuple(i, vals);
    if (!svtkMathUtilities::NearlyEqual(vals[0], trueFirstData[i]) ||
      !svtkMathUtilities::NearlyEqual(vals[1], trueSecondData[i]))
    {
      svtkGenericWarningMacro(
        "Incorrect values returned from scaled array after setting values in the array");
      return 1;
    }
  }

  // third check is for FillValue()
  array->FillValue(2.);
  for (svtkIdType i = 0; i < array->GetNumberOfTuples(); i++)
  {
    array->GetTypedTuple(i, vals);
    if (!svtkMathUtilities::NearlyEqual(vals[0], 2.) || !svtkMathUtilities::NearlyEqual(vals[1], 2.))
    {
      svtkGenericWarningMacro(
        "Incorrect values returned from scaled array after setting with FillValue(2.)");
      return 1;
    }
  }

  // fourth check is for getting raw pointer
  // Silence the void pointer warnings for these calls
  svtksys::SystemTools::PutEnv("SVTK_SILENCE_GET_VOID_POINTER_WARNINGS=1");
  double* rawPointer = array->GetPointer(0);
  if (!svtkMathUtilities::NearlyEqual(rawPointer[0], 2.))
  {
    svtkGenericWarningMacro("Incorrect values returned from scaled array after GetPointer()");
    return 1;
  }

  array->Delete();

  return 0; // success
}
