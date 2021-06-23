/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestInformationKeyLookup.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkInformationKeyLookup.h"

#include "svtkAbstractArray.h"
#include "svtkDataArray.h"
#include "svtkInformationDoubleVectorKey.h"
#include "svtkInformationInformationVectorKey.h"
#include "svtkInformationIntegerKey.h"
#include "svtkInformationVariantVectorKey.h"

// Lookup a key and test that it matches what it should.
namespace
{
bool VerifyKey(const std::string& name, const std::string& location, svtkInformationKey* key)
{
  return svtkInformationKeyLookup::Find(name, location) == key;
}
} // end anon namespace

#define VERIFY_KEY(name, location)                                                                 \
  if (!VerifyKey(#name, #location, location::name()))                                              \
  {                                                                                                \
    std::cerr << "Error finding key: " << #location << "::" << #name << "\n";                      \
    return EXIT_FAILURE;                                                                           \
  }

int TestInformationKeyLookup(int, char*[])
{
  // Test some keys in svtkAbstractArray and svtkDataArray (those are currently
  // the only ones available to this test, since the lookup only knows about
  // keys in currently linked modules).
  VERIFY_KEY(DISCRETE_VALUES, svtkAbstractArray)
  VERIFY_KEY(DISCRETE_VALUE_SAMPLE_PARAMETERS, svtkAbstractArray)
  VERIFY_KEY(GUI_HIDE, svtkAbstractArray)
  VERIFY_KEY(PER_COMPONENT, svtkAbstractArray)
  VERIFY_KEY(COMPONENT_RANGE, svtkDataArray)
  VERIFY_KEY(L2_NORM_RANGE, svtkDataArray)

  return EXIT_SUCCESS;
}
