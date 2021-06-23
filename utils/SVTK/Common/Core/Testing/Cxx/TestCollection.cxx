/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestCollection.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkCollection.h"
#include "svtkCollectionRange.h"
#include "svtkIntArray.h"
#include "svtkNew.h"
#include "svtkSmartPointer.h"

#include <algorithm>

bool TestRegister();
bool TestRemoveItem(int index, bool removeIndex);

int TestCollection(int, char*[])
{
  bool res = true;
  res = TestRegister() && res;
  res = TestRemoveItem(0, false) && res;
  res = TestRemoveItem(1, false) && res;
  res = TestRemoveItem(5, false) && res;
  res = TestRemoveItem(8, false) && res;
  res = TestRemoveItem(9, false) && res;
  res = TestRemoveItem(0, true) && res;
  res = TestRemoveItem(1, true) && res;
  res = TestRemoveItem(5, true) && res;
  res = TestRemoveItem(8, true) && res;
  res = TestRemoveItem(9, true) && res;
  return res ? EXIT_SUCCESS : EXIT_FAILURE;
}

static bool IsEqualRange(
  svtkCollection* collection, const std::vector<svtkSmartPointer<svtkIntArray> >& v)
{
  const auto range = svtk::Range(collection);
  if (range.size() != static_cast<int>(v.size()))
  {
    std::cerr << "Range size invalid.\n";
    return false;
  }

  // Test C++11 for-range interop
  auto vecIter = v.begin();
  for (auto item : range)
  {
    if (item != vecIter->GetPointer())
    {
      std::cerr << "Range iterator returned unexpected value.\n";
      return false;
    }
    ++vecIter;
  }

  return true;
}

static bool IsEqual(svtkCollection* collection, const std::vector<svtkSmartPointer<svtkIntArray> >& v)
{
  if (collection->GetNumberOfItems() != static_cast<int>(v.size()))
  {
    return false;
  }
  svtkIntArray* dataArray = nullptr;
  svtkCollectionSimpleIterator it;
  int i = 0;
  for (collection->InitTraversal(it);
       (dataArray = svtkIntArray::SafeDownCast(collection->GetNextItemAsObject(it))); ++i)
  {
    if (v[i] != dataArray)
    {
      return false;
    }
  }
  return IsEqualRange(collection, v); // test range iterators, too.
}

bool TestRegister()
{
  svtkNew<svtkCollection> collection;
  svtkIntArray* object = svtkIntArray::New();
  collection->AddItem(object);
  object->Delete();
  if (object->GetReferenceCount() != 1)
  {
    std::cout << object->GetReferenceCount() << std::endl;
    return false;
  }
  object->Register(nullptr);
  collection->RemoveItem(object);
  if (object->GetReferenceCount() != 1)
  {
    std::cout << object->GetReferenceCount() << std::endl;
    return false;
  }
  object->UnRegister(nullptr);
  return true;
}

bool TestRemoveItem(int index, bool removeIndex)
{
  svtkNew<svtkCollection> collection;
  std::vector<svtkSmartPointer<svtkIntArray> > objects;
  for (int i = 0; i < 10; ++i)
  {
    svtkNew<svtkIntArray> object;
    collection->AddItem(object);
    objects.push_back(object.GetPointer());
  }
  if (removeIndex)
  {
    collection->RemoveItem(index);
  }
  else
  {
    svtkObject* objectToRemove = objects[index];
    collection->RemoveItem(objectToRemove);
  }
  objects.erase(objects.begin() + index);
  if (!IsEqual(collection, objects))
  {
    std::cout << "TestRemoveItem failed:" << std::endl;
    collection->Print(std::cout);
    return false;
  }
  return true;
}
