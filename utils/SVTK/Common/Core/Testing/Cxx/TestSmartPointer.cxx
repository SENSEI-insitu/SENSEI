/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestSmartPointer.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME Test of svtkSmartPointer.
// .SECTION Description
// Tests instantiations of the svtkSmartPointer class template.

#include "svtkDebugLeaks.h"
#include "svtkIntArray.h"
#include "svtkNew.h"
#include "svtkSmartPointer.h"

#include <vector>

int TestSmartPointer(int, char*[])
{
  int rval = 0;
  svtkIntArray* ia = svtkIntArray::New();

  // Coverage:
  unsigned int testbits = 0;
  unsigned int correctbits = 0x00000953;
  const char* tests[] = { "da2 == ia", "da2 != ia", "da2 < ia", "da2 <= ia", "da2 > ia",
    "da2 <= ia", "da2 > ia", "da2 >= ia", "da1 == 0", "da1 != 0", "da1 < 0", "da1 <= 0", "da1 > 0",
    "da1 >= 0", nullptr };

  auto da2 = svtk::MakeSmartPointer(ia); // da2 is a svtkSmartPointer<svtkIntArray>
  svtkSmartPointer<svtkDataArray> da1(da2);
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
  da1 = svtkSmartPointer<svtkDataArray>::NewInstance(ia);
  da1.TakeReference(svtkIntArray::New());
  auto da4 = svtk::TakeSmartPointer(svtkIntArray::New());
  (void)da4;
  ia->Delete();

  std::vector<svtkSmartPointer<svtkIntArray> > intarrays;
  { // local scope for svtkNew object
    svtkNew<svtkIntArray> svtknew;
    svtkSmartPointer<svtkIntArray> aa(svtknew);
    intarrays.push_back(svtknew);
  }
  if (intarrays[0]->GetReferenceCount() != 1)
  {
    cerr << "Didn't properly add svtkNew object to stl vector of smart pointers\n";
    rval = 1;
  }

  // Test move constructors
  {
    svtkSmartPointer<svtkIntArray> intArray{ svtkNew<svtkIntArray>{} };
    if (intArray == nullptr || intArray->GetReferenceCount() != 1)
    {
      std::cerr << "Move constructing a svtkSmartPointer from a svtkNew "
                   "failed.\n";
      rval = 1;
    }

    svtkSmartPointer<svtkIntArray> intArrayCopy(intArray);
    if (intArrayCopy != intArray || intArray->GetReferenceCount() != 2 ||
      intArrayCopy->GetReferenceCount() != 2)
    {
      std::cerr << "Copy constructing svtkSmartPointer yielded unexpected "
                   "result.\n";
      rval = 1;
    }

    svtkSmartPointer<svtkIntArray> intArrayMoved(std::move(intArrayCopy));
    if (intArrayCopy || !intArrayMoved || intArrayMoved->GetReferenceCount() != 2)
    {
      std::cerr << "Move constructing svtkSmartPointer yielded unexpected "
                   "result.\n";
      rval = 1;
    }

    svtkSmartPointer<svtkDataArray> dataArrayCopy(intArray);
    if (dataArrayCopy != intArray || intArray->GetReferenceCount() != 3 ||
      dataArrayCopy->GetReferenceCount() != 3)
    {
      std::cerr << "Cast constructing svtkSmartPointer failed.\n";
      rval = 1;
    }

    svtkSmartPointer<svtkDataArray> dataArrayMoved(std::move(intArrayMoved));
    if (!dataArrayMoved || intArrayMoved || dataArrayMoved->GetReferenceCount() != 3)
    {
      std::cerr << "Cast move-constructing svtkSmartPointer failed.\n";
      rval = 1;
    }
  }

  return rval;
}
