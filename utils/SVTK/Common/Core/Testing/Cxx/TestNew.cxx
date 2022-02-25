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
// .NAME Test of svtkNew.
// .SECTION Description
// Tests instantiations of the svtkNew class template.

#include "svtkDebugLeaks.h"
#include "svtkFloatArray.h"
#include "svtkIntArray.h"
#include "svtkNew.h"
#include "svtkSmartPointer.h"
#include "svtkWeakPointer.h"

#include "svtkTestNewVar.h"

int TestNew(int, char*[])
{
  bool error = false;
  // This one should be cleaned up when the main function ends.
  svtkNew<svtkIntArray> a;
  if (a->GetReferenceCount() != 1)
  {
    error = true;
    cerr << "Error, reference count should be 1, was " << a->GetReferenceCount() << endl;
  }
  cout << "svtkNew streaming " << a << endl;

  svtkWeakPointer<svtkFloatArray> wf;
  // Test scoping, and deletion.
  if (wf == nullptr)
  {
    svtkNew<svtkFloatArray> f;
    wf = f;
  }
  if (wf != nullptr)
  {
    error = true;
    cerr << "Error, svtkNew failed to delete the object it contained." << endl;
  }
  // Test implicit conversion svtkNew::operator T* () const
  if (wf == nullptr)
  {
    svtkNew<svtkFloatArray> f;
    wf = f;
  }
  if (wf != nullptr)
  {
    error = true;
    cerr << "Error, svtkNew failed to delete the object it contained (implicit cast to raw pointer)."
         << endl;
  }

  // Now test interaction with the smart pointer.
  svtkSmartPointer<svtkIntArray> si;
  if (si == nullptr)
  {
    svtkNew<svtkIntArray> i;
    si = i;
  }
  if (si->GetReferenceCount() != 1)
  {
    error = true;
    cerr << "Error, svtkNew failed to delete the object it contained, "
         << "or the smart pointer failed to increment it. Reference count: "
         << si->GetReferenceCount() << endl;
  }

  // Test raw object reference
  svtkObject& p = *si;
  if (p.GetReferenceCount() != 1)
  {
    error = true;
    cerr << "Error, svtkNew failed to keep the object it contained, "
         << "or setting a raw reference incremented it. Reference count: " << p.GetReferenceCount()
         << endl;
  }

  svtkNew<svtkTestNewVar> newVarObj;
  if (newVarObj->GetPointsRefCount() != 1)
  {
    error = true;
    cerr << "The member pointer failed to set the correct reference count: "
         << newVarObj->GetPointsRefCount() << endl;
  }

  svtkSmartPointer<svtkObject> points = newVarObj->GetPoints();
  if (points->GetReferenceCount() != 2)
  {
    error = true;
    cerr << "Error, svtkNew failed to keep the object it contained, "
         << "or the smart pointer failed to increment it. Reference count: "
         << points->GetReferenceCount() << endl;
  }
  svtkSmartPointer<svtkObject> points2 = newVarObj->GetPoints2();
  if (points2->GetReferenceCount() != 3)
  {
    error = true;
    cerr << "Error, svtkNew failed to keep the object it contained, "
         << "or the smart pointer failed to increment it. Reference count: "
         << points->GetReferenceCount() << endl;
  }

  svtkNew<svtkIntArray> intarray;
  svtkIntArray* intarrayp = intarray.GetPointer();
  if (intarrayp != intarray || intarray != intarrayp)
  {
    error = true;
    cerr << "Error, comparison of svtkNew object to it's raw pointer fails\n";
  }

  {
    svtkNew<svtkIntArray> testArray1;
    svtkNew<svtkIntArray> testArray2(std::move(testArray1));
    if (testArray1 || !testArray2)
    {
      std::cerr << "Error, move construction of svtkNew failed.\n";
      error = true;
    }
    svtkNew<svtkDataArray> testArray3(std::move(testArray2));
    if (testArray2 || !testArray3)
    {
      std::cerr << "Error, move construction of svtkNew failed.\n";
      error = true;
    }
  }

  return error ? 1 : 0;
}
