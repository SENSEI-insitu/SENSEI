/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkWeakPointerBase.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkWeakPointerBase
 * @brief   Non-templated superclass for svtkWeakPointer.
 *
 * svtkWeakPointerBase holds a pointer to a svtkObjectBase or subclass
 * instance, but it never affects the reference count of the svtkObjectBase. However,
 * when the svtkObjectBase referred to is destroyed, the pointer gets initialized to
 * nullptr, thus avoid dangling references.
 */

#ifndef svtkWeakPointerBase_h
#define svtkWeakPointerBase_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObjectBase.h"

class svtkObjectBaseToWeakPointerBaseFriendship;

class SVTKCOMMONCORE_EXPORT svtkWeakPointerBase
{
public:
  /**
   * Initialize smart pointer to nullptr.
   */
  svtkWeakPointerBase() noexcept : Object(nullptr) {}

  /**
   * Initialize smart pointer to given object.
   */
  svtkWeakPointerBase(svtkObjectBase* r);

  /**
   * Copy r's data object into the new weak pointer.
   */
  svtkWeakPointerBase(const svtkWeakPointerBase& r);

  /**
   * Move r's object into the new weak pointer, setting r to nullptr.
   */
  svtkWeakPointerBase(svtkWeakPointerBase&& r) noexcept;

  /**
   * Destroy smart pointer.
   */
  ~svtkWeakPointerBase();

  //@{
  /**
   * Assign object to reference.  This removes any reference to an old
   * object.
   */
  svtkWeakPointerBase& operator=(svtkObjectBase* r);
  svtkWeakPointerBase& operator=(const svtkWeakPointerBase& r);
  svtkWeakPointerBase& operator=(svtkWeakPointerBase&& r) noexcept;
  //@}

  /**
   * Get the contained pointer.
   */
  svtkObjectBase* GetPointer() const
  {
    // Inline implementation so smart pointer comparisons can be fully
    // inlined.
    return this->Object;
  }

private:
  friend class svtkObjectBaseToWeakPointerBaseFriendship;

protected:
  // Initialize weak pointer to given object.
  class NoReference
  {
  };
  svtkWeakPointerBase(svtkObjectBase* r, const NoReference&);

  // Pointer to the actual object.
  svtkObjectBase* Object;
};

//----------------------------------------------------------------------------
#define SVTK_WEAK_POINTER_BASE_DEFINE_OPERATOR(op)                                                  \
  inline bool operator op(const svtkWeakPointerBase& l, const svtkWeakPointerBase& r)                \
  {                                                                                                \
    return (static_cast<void*>(l.GetPointer()) op static_cast<void*>(r.GetPointer()));             \
  }                                                                                                \
  inline bool operator op(svtkObjectBase* l, const svtkWeakPointerBase& r)                           \
  {                                                                                                \
    return (static_cast<void*>(l) op static_cast<void*>(r.GetPointer()));                          \
  }                                                                                                \
  inline bool operator op(const svtkWeakPointerBase& l, svtkObjectBase* r)                           \
  {                                                                                                \
    return (static_cast<void*>(l.GetPointer()) op static_cast<void*>(r));                          \
  }
/**
 * Compare smart pointer values.
 */
SVTK_WEAK_POINTER_BASE_DEFINE_OPERATOR(==)
SVTK_WEAK_POINTER_BASE_DEFINE_OPERATOR(!=)
SVTK_WEAK_POINTER_BASE_DEFINE_OPERATOR(<)
SVTK_WEAK_POINTER_BASE_DEFINE_OPERATOR(<=)
SVTK_WEAK_POINTER_BASE_DEFINE_OPERATOR(>)
SVTK_WEAK_POINTER_BASE_DEFINE_OPERATOR(>=)

#undef SVTK_WEAK_POINTER_BASE_DEFINE_OPERATOR

/**
 * Streaming operator to print smart pointer like regular pointers.
 */
SVTKCOMMONCORE_EXPORT ostream& operator<<(ostream& os, const svtkWeakPointerBase& p);

#endif
// SVTK-HeaderTest-Exclude: svtkWeakPointerBase.h
