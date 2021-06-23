/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSmartPointerBase.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkSmartPointerBase
 * @brief   Non-templated superclass for svtkSmartPointer.
 *
 * svtkSmartPointerBase holds a pointer to a svtkObjectBase or subclass
 * instance and performs one Register/UnRegister pair.  This is useful
 * for storing SVTK objects in STL containers.  This class is not
 * intended to be used directly.  Instead, use the svtkSmartPointer
 * class template to automatically perform proper cast operations.
 */

#ifndef svtkSmartPointerBase_h
#define svtkSmartPointerBase_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObjectBase.h"

class SVTKCOMMONCORE_EXPORT svtkSmartPointerBase
{
public:
  /**
   * Initialize smart pointer to nullptr.
   */
  svtkSmartPointerBase() noexcept;

  /**
   * Initialize smart pointer to given object.
   */
  svtkSmartPointerBase(svtkObjectBase* r);

  /**
   * Initialize smart pointer with a new reference to the same object
   * referenced by given smart pointer.
   */
  svtkSmartPointerBase(const svtkSmartPointerBase& r);

  /**
   * Move the pointee from @a r into @a this and reset @ r.
   */
  svtkSmartPointerBase(svtkSmartPointerBase&& r) noexcept : Object(r.Object) { r.Object = nullptr; }

  /**
   * Destroy smart pointer and remove the reference to its object.
   */
  ~svtkSmartPointerBase();

  //@{
  /**
   * Assign object to reference.  This removes any reference to an old
   * object.
   */
  svtkSmartPointerBase& operator=(svtkObjectBase* r);
  svtkSmartPointerBase& operator=(const svtkSmartPointerBase& r);
  //@}

  /**
   * Get the contained pointer.
   */
  svtkObjectBase* GetPointer() const noexcept
  {
    // Inline implementation so smart pointer comparisons can be fully
    // inlined.
    return this->Object;
  }

  /**
   * Report the reference held by the smart pointer to a collector.
   */
  void Report(svtkGarbageCollector* collector, const char* desc);

protected:
  // Initialize smart pointer to given object, but do not increment
  // reference count.  The destructor will still decrement the count.
  // This effectively makes it an auto-ptr.
  class NoReference
  {
  };
  svtkSmartPointerBase(svtkObjectBase* r, const NoReference&);

  // Pointer to the actual object.
  svtkObjectBase* Object;

private:
  // Internal utility methods.
  void Swap(svtkSmartPointerBase& r) noexcept;
  void Register();
};

//----------------------------------------------------------------------------
#define SVTK_SMART_POINTER_BASE_DEFINE_OPERATOR(op)                                                 \
  inline bool operator op(const svtkSmartPointerBase& l, const svtkSmartPointerBase& r)              \
  {                                                                                                \
    return (static_cast<void*>(l.GetPointer()) op static_cast<void*>(r.GetPointer()));             \
  }                                                                                                \
  inline bool operator op(svtkObjectBase* l, const svtkSmartPointerBase& r)                          \
  {                                                                                                \
    return (static_cast<void*>(l) op static_cast<void*>(r.GetPointer()));                          \
  }                                                                                                \
  inline bool operator op(const svtkSmartPointerBase& l, svtkObjectBase* r)                          \
  {                                                                                                \
    return (static_cast<void*>(l.GetPointer()) op static_cast<void*>(r));                          \
  }
/**
 * Compare smart pointer values.
 */
SVTK_SMART_POINTER_BASE_DEFINE_OPERATOR(==)
SVTK_SMART_POINTER_BASE_DEFINE_OPERATOR(!=)
SVTK_SMART_POINTER_BASE_DEFINE_OPERATOR(<)
SVTK_SMART_POINTER_BASE_DEFINE_OPERATOR(<=)
SVTK_SMART_POINTER_BASE_DEFINE_OPERATOR(>)
SVTK_SMART_POINTER_BASE_DEFINE_OPERATOR(>=)

#undef SVTK_SMART_POINTER_BASE_DEFINE_OPERATOR

/**
 * Streaming operator to print smart pointer like regular pointers.
 */
SVTKCOMMONCORE_EXPORT ostream& operator<<(ostream& os, const svtkSmartPointerBase& p);

#endif
// SVTK-HeaderTest-Exclude: svtkSmartPointerBase.h
