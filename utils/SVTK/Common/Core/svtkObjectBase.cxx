/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkObjectBase.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkObjectBase.h"
#include "svtkDebugLeaks.h"
#include "svtkGarbageCollector.h"
#include "svtkWeakPointerBase.h"

#include <sstream>

#define svtkBaseDebugMacro(x)

class svtkObjectBaseToGarbageCollectorFriendship
{
public:
  static int GiveReference(svtkObjectBase* obj) { return svtkGarbageCollector::GiveReference(obj); }
  static int TakeReference(svtkObjectBase* obj) { return svtkGarbageCollector::TakeReference(obj); }
};

class svtkObjectBaseToWeakPointerBaseFriendship
{
public:
  static void ClearPointer(svtkWeakPointerBase* p) { p->Object = nullptr; }
};

// avoid dll boundary problems
#ifdef _WIN32
void* svtkObjectBase::operator new(size_t nSize)
{
  void* p = malloc(nSize);
  return p;
}

void svtkObjectBase::operator delete(void* p)
{
  free(p);
}
#endif

// ------------------------------------svtkObjectBase----------------------
// This operator allows all subclasses of svtkObjectBase to be printed via <<.
// It in turn invokes the Print method, which in turn will invoke the
// PrintSelf method that all objects should define, if they have anything
// interesting to print out.
ostream& operator<<(ostream& os, svtkObjectBase& o)
{
  o.Print(os);
  return os;
}

// Create an object with Debug turned off and modified time initialized
// to zero.
svtkObjectBase::svtkObjectBase()
{
  this->ReferenceCount = 1;
  this->WeakPointers = nullptr;
#ifdef SVTK_DEBUG_LEAKS
  svtkDebugLeaks::ConstructingObject(this);
#endif
}

svtkObjectBase::~svtkObjectBase()
{
#ifdef SVTK_DEBUG_LEAKS
  svtkDebugLeaks::DestructingObject(this);
#endif

  // warn user if reference counting is on and the object is being referenced
  // by another object
  if (this->ReferenceCount > 0)
  {
    svtkGenericWarningMacro(<< "Trying to delete object with non-zero reference count.");
  }
}

//----------------------------------------------------------------------------
void svtkObjectBase::InitializeObjectBase()
{
#ifdef SVTK_DEBUG_LEAKS
  svtkDebugLeaks::ConstructClass(this);
#endif // SVTK_DEBUG_LEAKS
}

//----------------------------------------------------------------------------
#ifdef SVTK_WORKAROUND_WINDOWS_MANGLE
#undef GetClassName
// Define possible mangled names.
const char* svtkObjectBase::GetClassNameA() const
{
  return this->GetClassNameInternal();
}
const char* svtkObjectBase::GetClassNameW() const
{
  return this->GetClassNameInternal();
}
#endif
const char* svtkObjectBase::GetClassName() const
{
  return this->GetClassNameInternal();
}

svtkTypeBool svtkObjectBase::IsTypeOf(const char* name)
{
  if (!strcmp("svtkObjectBase", name))
  {
    return 1;
  }
  return 0;
}

svtkTypeBool svtkObjectBase::IsA(const char* type)
{
  return this->svtkObjectBase::IsTypeOf(type);
}

svtkIdType svtkObjectBase::GetNumberOfGenerationsFromBaseType(const char* name)
{
  if (!strcmp("svtkObjectBase", name))
  {
    return 0;
  }
  // Return the lowest value for svtkIdType. Because of recursion, the returned
  // value for derived classes will be this value added to the type distance to
  // svtkObjectBase. This sum will still be a negative (and, therefore, invalid)
  // value.
  return SVTK_ID_MIN;
}

svtkIdType svtkObjectBase::GetNumberOfGenerationsFromBase(const char* type)
{
  return this->svtkObjectBase::GetNumberOfGenerationsFromBaseType(type);
}

// Delete a svtk object. This method should always be used to delete an object
// when the new operator was used to create it. Using the C++ delete method
// will not work with reference counting.
void svtkObjectBase::Delete()
{
  this->UnRegister(static_cast<svtkObjectBase*>(nullptr));
}

void svtkObjectBase::FastDelete()
{
  // Remove the reference without doing a collection check even if
  // this object normally participates in garbage collection.
  this->UnRegisterInternal(nullptr, 0);
}

void svtkObjectBase::Print(ostream& os)
{
  svtkIndent indent;

  this->PrintHeader(os, svtkIndent(0));
  this->PrintSelf(os, indent.GetNextIndent());
  this->PrintTrailer(os, svtkIndent(0));
}

void svtkObjectBase::PrintHeader(ostream& os, svtkIndent indent)
{
  os << indent << this->GetClassName() << " (" << this << ")\n";
}

// Chaining method to print an object's instance variables, as well as
// its superclasses.
void svtkObjectBase::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "Reference Count: " << this->ReferenceCount << "\n";
}

void svtkObjectBase::PrintTrailer(ostream& os, svtkIndent indent)
{
  os << indent << "\n";
}

// Description:
// Sets the reference count (use with care)
void svtkObjectBase::SetReferenceCount(int ref)
{
  this->ReferenceCount = ref;
  svtkBaseDebugMacro(<< "Reference Count set to " << this->ReferenceCount);
}

//----------------------------------------------------------------------------
void svtkObjectBase::Register(svtkObjectBase* o)
{
  // Do not participate in garbage collection by default.
  this->RegisterInternal(o, 0);
}

//----------------------------------------------------------------------------
void svtkObjectBase::UnRegister(svtkObjectBase* o)
{
  // Do not participate in garbage collection by default.
  this->UnRegisterInternal(o, 0);
}

//----------------------------------------------------------------------------
void svtkObjectBase::RegisterInternal(svtkObjectBase*, svtkTypeBool check)
{
  // If a reference is available from the garbage collector, use it.
  // Otherwise create a new reference by incrementing the reference
  // count.
  if (!(check && svtkObjectBaseToGarbageCollectorFriendship::TakeReference(this)))
  {
    this->ReferenceCount++;
  }
}

//----------------------------------------------------------------------------
void svtkObjectBase::UnRegisterInternal(svtkObjectBase*, svtkTypeBool check)
{
  // If the garbage collector accepts a reference, do not decrement
  // the count.
  if (check && this->ReferenceCount > 1 &&
    svtkObjectBaseToGarbageCollectorFriendship::GiveReference(this))
  {
    return;
  }

  // Decrement the reference count, delete object if count goes to zero.
  if (--this->ReferenceCount <= 0)
  {
    // Clear all weak pointers to the object before deleting it.
    if (this->WeakPointers)
    {
      svtkWeakPointerBase** p = this->WeakPointers;
      while (*p)
      {
        svtkObjectBaseToWeakPointerBaseFriendship::ClearPointer(*p++);
      }
      delete[] this->WeakPointers;
    }
#ifdef SVTK_DEBUG_LEAKS
    svtkDebugLeaks::DestructClass(this);
#endif
    delete this;
  }
  else if (check)
  {
    // The garbage collector did not accept the reference, but the
    // object still exists and is participating in garbage collection.
    // This means either that delayed garbage collection is disabled
    // or the collector has decided it is time to do a check.
    svtkGarbageCollector::Collect(this);
  }
}

//----------------------------------------------------------------------------
void svtkObjectBase::ReportReferences(svtkGarbageCollector*)
{
  // svtkObjectBase has no references to report.
}
