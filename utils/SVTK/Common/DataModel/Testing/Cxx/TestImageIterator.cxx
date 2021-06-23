/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestImageIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// .NAME Test of image iterators
// .SECTION Description
// this program tests the image iterators
// At this point it only creates an object of every supported type.

#include "svtkDebugLeaks.h"
#include "svtkImageData.h"
#include "svtkImageIterator.h"
#include "svtkImageProgressIterator.h"

template <class T>
inline int DoTest(T*)
{
  int ext[6] = { 0, 0, 0, 0, 0, 0 };
  svtkImageData* id = svtkImageData::New();
  id->SetExtent(ext);
  id->AllocateScalars(SVTK_DOUBLE, 1);
  svtkImageIterator<T>* it = new svtkImageIterator<T>(id, ext);
  svtkImageProgressIterator<T>* ipt = new svtkImageProgressIterator<T>(id, ext, nullptr, 0);
  delete it;
  delete ipt;
  id->Delete();
  return 0;
}

int TestImageIterator(int, char*[])
{
  DoTest(static_cast<char*>(nullptr));
  DoTest(static_cast<int*>(nullptr));
  DoTest(static_cast<long*>(nullptr));
  DoTest(static_cast<short*>(nullptr));
  DoTest(static_cast<float*>(nullptr));
  DoTest(static_cast<double*>(nullptr));
  DoTest(static_cast<unsigned long*>(nullptr));
  DoTest(static_cast<unsigned short*>(nullptr));
  DoTest(static_cast<unsigned char*>(nullptr));
  DoTest(static_cast<unsigned int*>(nullptr));

  return 0;
}
