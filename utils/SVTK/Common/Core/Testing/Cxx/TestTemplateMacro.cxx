/*=========================================================================

  Program:   Visualization Toolkit
  Module:    ObjectFactory.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkSetGet.h"

template <typename T1, typename T2>
static void myFunc1(T1* p1, T2* p2)
{
  *p2 = T2(*p1);
}

static bool RunTemplate2Macro1(int tIn, void* pIn, int tOut, void* pOut)
{
  switch (svtkTemplate2PackMacro(tIn, tOut))
  {
    // Test implicit deduction of multiple template arguments.
    svtkTemplate2Macro(myFunc1(static_cast<SVTK_T1*>(pIn), static_cast<SVTK_T2*>(pOut)));
    default:
      // Unknown input or output SVTK type id.
      return false;
  }
  return true;
}

template <typename T1, typename T2>
static void myFunc2(void* p1, void* p2)
{
  *static_cast<T2*>(p2) = T2(*static_cast<T1*>(p1));
}

static bool RunTemplate2Macro2(int tIn, void* pIn, int tOut, void* pOut)
{
  switch (svtkTemplate2PackMacro(tIn, tOut))
  {
    // Test explicit specification of multiple template arguments.
    svtkTemplate2Macro((myFunc2<SVTK_T1, SVTK_T2>(pIn, pOut)));
    default:
      // Unknown input or output SVTK type id.
      return false;
  }
  return true;
}

template <int NIn, typename TIn, int NOut, typename TOut>
static bool TestTemplate2Macro()
{
  TIn in = 1;
  TOut out = 0;
  if (!RunTemplate2Macro1(NIn, &in, NOut, &out) || out != 1)
  {
    return false;
  }
  in = 2;
  if (!RunTemplate2Macro2(NIn, &in, NOut, &out) || out != 2)
  {
    return false;
  }
  return true;
}

int TestTemplateMacro(int, char*[])
{
  bool res = true;

  // Verify that a few combinations are dispatched.
  res = TestTemplate2Macro<SVTK_FLOAT, float, SVTK_INT, int>() && res;
  res = TestTemplate2Macro<SVTK_DOUBLE, double, SVTK_ID_TYPE, svtkIdType>() && res;
  res = TestTemplate2Macro<SVTK_INT, int, SVTK_LONG, long>() && res;
  res = TestTemplate2Macro<SVTK_CHAR, char, SVTK_LONG, long>() && res;

  // Verify that bad SVTK type ids are rejected.
  res = !TestTemplate2Macro<127, char, SVTK_LONG, long>() && res;
  res = !TestTemplate2Macro<SVTK_CHAR, char, 127, long>() && res;

  return res ? EXIT_SUCCESS : EXIT_FAILURE;
}
