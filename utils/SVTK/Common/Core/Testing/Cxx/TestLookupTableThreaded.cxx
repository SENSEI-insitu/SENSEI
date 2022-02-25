/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestLookupTableThreaded.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkLookupTable.h"
#include "svtkMultiThreader.h"
#include "svtkNew.h"

namespace
{

svtkLookupTable* lut;

SVTK_THREAD_RETURN_TYPE ThreadedMethod(void*)
{
  int numberOfValues = 25;
  double* input = new double[numberOfValues];
  for (int i = 0; i < numberOfValues; ++i)
  {
    input[i] = static_cast<double>(i);
  }
  unsigned char* output = new unsigned char[4 * numberOfValues];
  int inputType = SVTK_DOUBLE;
  int inputIncrement = 1;
  int outputFormat = SVTK_RGBA;

  lut->MapScalarsThroughTable2(
    input, output, inputType, numberOfValues, inputIncrement, outputFormat);

  delete[] input;
  delete[] output;

  return SVTK_THREAD_RETURN_VALUE;
}

} // end anonymous namespace

int TestLookupTableThreaded(int, char*[])
{
  lut = svtkLookupTable::New();
  lut->SetNumberOfTableValues(1024);

  svtkNew<svtkMultiThreader> threader;
  threader->SetSingleMethod(ThreadedMethod, nullptr);
  threader->SetNumberOfThreads(4);
  threader->SingleMethodExecute();

  lut->Delete();

  return EXIT_SUCCESS;
}
