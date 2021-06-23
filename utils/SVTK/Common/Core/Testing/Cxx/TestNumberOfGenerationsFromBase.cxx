/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestVariant.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include <svtkFloatArray.h>
#include <svtkIntArray.h>
#include <svtkLogger.h>
#include <svtkNew.h>
#include <svtkObject.h>

int TestNumberOfGenerationsFromBase(int, char*[])
{
  svtkNew<svtkFloatArray> floatArray;
  if (floatArray->GetNumberOfGenerationsFromBase(svtkNew<svtkObject>()->GetClassName()) != 5)
  {
    svtkLog(ERROR,
      "Incorrect number of generations between "
        << floatArray->GetClassName() << " and " << svtkNew<svtkObject>()->GetClassName()
        << " (received "
        << floatArray->GetNumberOfGenerationsFromBase(svtkNew<svtkObject>()->GetClassName())
        << ", should be " << 5);
    return EXIT_FAILURE;
  }

  if (floatArray->GetNumberOfGenerationsFromBase("svtkAbstractArray") != 4)
  {
    svtkLog(ERROR,
      "Incorrect number of generations between "
        << floatArray->GetClassName() << " and "
        << "svtkAbstractArray"
        << " (received " << floatArray->GetNumberOfGenerationsFromBase("svtkAbstractArray")
        << ", should be " << 4);
    return EXIT_FAILURE;
  }

  if (floatArray->GetNumberOfGenerationsFromBase(floatArray->GetClassName()) != 0)
  {
    svtkLog(ERROR,
      "Incorrect number of generations between "
        << floatArray->GetClassName() << " and " << floatArray->GetClassName() << " (received "
        << floatArray->GetNumberOfGenerationsFromBase(svtkNew<svtkObject>()->GetClassName())
        << ", should be " << 0);
    return EXIT_FAILURE;
  }

  svtkNew<svtkIntArray> intArray;
  if (floatArray->GetNumberOfGenerationsFromBase(intArray->GetClassName()) >= 0)
  {
    svtkLog(ERROR,
      "Incorrect number of generations between "
        << floatArray->GetClassName() << " and " << intArray->GetClassName() << " (received "
        << floatArray->GetNumberOfGenerationsFromBase(intArray->GetClassName())
        << ", should be < 0");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
