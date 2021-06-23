/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestWeakPointer.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME Test of svtkWeakPointer.
// .SECTION Description
// Tests instantiations of the svtkWeakPointer class template.

#include "svtkDebugLeaks.h"
#include "svtkIntArray.h"
#include "svtkWeakPointer.h"

int TestWeakPointer(int, char*[])
{
  int rval = 0;
  svtkIntArray* ia = svtkIntArray::New();

  // Coverage:
  unsigned int testbits = 0;
  unsigned int correctbits = 0x00000953;
  const char* tests[] = { "da2 == ia", "da2 != ia", "da2 < ia", "da2 <= ia", "da2 > ia",
    "da2 <= ia", "da2 > ia", "da2 >= ia", "da1 == 0", "da1 != 0", "da1 < 0", "da1 <= 0", "da1 > 0",
    "da1 >= 0", nullptr };

  auto da2 = svtk::TakeWeakPointer(ia); // da2 is svtkWeakPointer<svtkIntArray>
  svtkWeakPointer<svtkDataArray> da1(da2);
  da1 = ia;
  da1 = da2;
  testbits = (testbits << 1) | ((da2 == ia) ? 1 : 0);
  testbits = (testbits << 1) | ((da2 != ia) ? 1 : 0);
  testbits = (testbits << 1) | ((da2 < ia) ? 1 : 0);
  testbits = (testbits << 1) | ((da2 <= ia) ? 1 : 0);
  testbits = (testbits << 1) | ((da2 > ia) ? 1 : 0);
  testbits = (testbits << 1) | ((da2 >= ia) ? 1 : 0);
  testbits = (testbits << 1) | ((da1 == nullptr) ? 1 : 0);
  testbits = (testbits << 1) | ((da1 != nullptr) ? 1 : 0);
  testbits = (testbits << 1) | ((da1 < nullptr) ? 1 : 0);
  testbits = (testbits << 1) | ((da1 <= nullptr) ? 1 : 0);
  testbits = (testbits << 1) | ((da1 > nullptr) ? 1 : 0);
  testbits = (testbits << 1) | ((da1 >= nullptr) ? 1 : 0);
  if (testbits != correctbits)
  {
    unsigned int diffbits = (testbits ^ correctbits);
    int bitcount = 0;
    while (tests[bitcount] != nullptr)
    {
      bitcount++;
    }
    for (int ib = 0; ib < bitcount; ++ib)
    {
      if (((diffbits >> (bitcount - ib - 1)) & 1) != 0)
      {
        cerr << "comparison (" << tests[ib] << ") failed!\n";
      }
    }
    rval = 1;
  }

  (*da1).SetNumberOfComponents(1);
  if (da2)
  {
    da2->SetNumberOfComponents(1);
  }
  if (!da2)
  {
    cerr << "da2 is nullptr!"
         << "\n";
    rval = 1;
  }
  cout << "IntArray: " << da2 << "\n";

  if (da1 == nullptr)
  {
    cerr << "da1 is nullptr\n";
    rval = 1;
  }
  if (da2 == nullptr)
  {
    cerr << "da2 is nullptr\n";
    rval = 1;
  }

  da2 = nullptr;
  ia->Delete();

  if (da1 != nullptr)
  {
    cerr << "da1 is not nullptr\n";
    rval = 1;
  }

  {
    svtkNew<svtkIntArray> array;
    svtkWeakPointer<svtkIntArray> intArray(array);
    if (array != intArray || array->GetReferenceCount() != 1)
    {
      std::cerr << "Constructing svtkWeakPointer from svtkNew failed.\n";
      rval = 1;
    }
    array.Reset();
    if (intArray)
    {
      std::cerr << "Weak pointer not nullptr\n";
      rval = 1;
    }
  }

  {
    svtkNew<svtkIntArray> array;
    svtkWeakPointer<svtkDataArray> dataArray(array);
    if (array != dataArray || array->GetReferenceCount() != 1)
    {
      std::cerr << "Constructing svtkWeakPointer from svtkNew failed.\n";
      rval = 1;
    }
    array.Reset();
    if (dataArray)
    {
      std::cerr << "Weak pointer not nullptr\n";
      rval = 1;
    }
  }

  {
    svtkNew<svtkIntArray> array;
    svtkWeakPointer<svtkIntArray> intArray(array);
    svtkWeakPointer<svtkIntArray> intArray2(intArray);
    if (array != intArray || array != intArray2 || array->GetReferenceCount() != 1)
    {
      std::cerr << "Copy failed.\n";
      rval = 1;
    }
    array.Reset();
    if (intArray || intArray2)
    {
      std::cerr << "Weak pointer not nullptr\n";
      rval = 1;
    }
  }

  {
    svtkNew<svtkIntArray> array;
    svtkWeakPointer<svtkIntArray> intArray(array);
    svtkWeakPointer<svtkIntArray> intArray2(std::move(intArray));
    if (intArray || array != intArray2 || array->GetReferenceCount() != 1)
    {
      std::cerr << "Move failed.\n";
      rval = 1;
    }
    array.Reset();
    if (intArray || intArray2)
    {
      std::cerr << "Weak pointer not nullptr\n";
      rval = 1;
    }
  }

  {
    svtkNew<svtkIntArray> array;
    svtkWeakPointer<svtkIntArray> intArray(array);
    svtkWeakPointer<svtkDataArray> dataArray(intArray);
    if (array != intArray || array != dataArray || array->GetReferenceCount() != 1)
    {
      std::cerr << "Copy failed.\n";
      rval = 1;
    }
    array.Reset();
    if (dataArray || intArray)
    {
      std::cerr << "Weak pointer not nullptr\n";
      rval = 1;
    }
  }

  {
    svtkNew<svtkIntArray> array;
    svtkWeakPointer<svtkIntArray> intArray(array);
    svtkWeakPointer<svtkDataArray> dataArray(std::move(intArray));
    if (intArray || array != dataArray || array->GetReferenceCount() != 1)
    {
      std::cerr << "Move failed.\n";
      rval = 1;
    }
    array.Reset();
    if (dataArray)
    {
      std::cerr << "Weak pointer not nullptr\n";
      rval = 1;
    }
  }

  return rval;
}
