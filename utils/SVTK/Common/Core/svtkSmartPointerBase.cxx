/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSmartPointerBase.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkSmartPointerBase.h"

#include "svtkGarbageCollector.h"

//----------------------------------------------------------------------------
svtkSmartPointerBase::svtkSmartPointerBase() noexcept : Object(nullptr) {}

//----------------------------------------------------------------------------
svtkSmartPointerBase::svtkSmartPointerBase(svtkObjectBase* r)
  : Object(r)
{
  // Add a reference to the object.
  this->Register();
}

//----------------------------------------------------------------------------
svtkSmartPointerBase::svtkSmartPointerBase(svtkObjectBase* r, const NoReference&)
  : Object(r)
{
  // Do not add a reference to the object because we received the
  // NoReference argument.
}

//----------------------------------------------------------------------------
svtkSmartPointerBase::svtkSmartPointerBase(const svtkSmartPointerBase& r)
  : Object(r.Object)
{
  // Add a reference to the object.
  this->Register();
}

//----------------------------------------------------------------------------
svtkSmartPointerBase::~svtkSmartPointerBase()
{
  // The main pointer must be set to nullptr before calling UnRegister,
  // so use a local variable to save the pointer.  This is because the
  // garbage collection reference graph traversal may make it back to
  // this smart pointer, and we do not want to include this reference.
  svtkObjectBase* object = this->Object;
  if (object)
  {
    this->Object = nullptr;
    object->UnRegister(nullptr);
  }
}

//----------------------------------------------------------------------------
svtkSmartPointerBase& svtkSmartPointerBase::operator=(svtkObjectBase* r)
{
  if (r != this->Object)
  {
    // This is an exception-safe assignment idiom that also gives the
    // correct order of register/unregister calls to all objects
    // involved.  A temporary is constructed that references the new
    // object.  Then the main pointer and temporary are swapped and the
    // temporary's destructor unreferences the old object.
    svtkSmartPointerBase(r).Swap(*this);
  }
  return *this;
}

//----------------------------------------------------------------------------
svtkSmartPointerBase& svtkSmartPointerBase::operator=(const svtkSmartPointerBase& r)
{
  if (&r != this && r.Object != this->Object)
  {
    // This is an exception-safe assignment idiom that also gives the
    // correct order of register/unregister calls to all objects
    // involved.  A temporary is constructed that references the new
    // object.  Then the main pointer and temporary are swapped and the
    // temporary's destructor unreferences the old object.
    svtkSmartPointerBase(r).Swap(*this);
  }
  return *this;
}

//----------------------------------------------------------------------------
void svtkSmartPointerBase::Report(svtkGarbageCollector* collector, const char* desc)
{
  svtkGarbageCollectorReport(collector, this->Object, desc);
}

//----------------------------------------------------------------------------
void svtkSmartPointerBase::Swap(svtkSmartPointerBase& r) noexcept
{
  // Just swap the pointers.  This is used internally by the
  // assignment operator.
  svtkObjectBase* temp = r.Object;
  r.Object = this->Object;
  this->Object = temp;
}

//----------------------------------------------------------------------------
void svtkSmartPointerBase::Register()
{
  // Add a reference only if the object is not nullptr.
  if (this->Object)
  {
    this->Object->Register(nullptr);
  }
}

//----------------------------------------------------------------------------
ostream& operator<<(ostream& os, const svtkSmartPointerBase& p)
{
  // Just print the pointer value into the stream.
  return os << static_cast<void*>(p.GetPointer());
}
