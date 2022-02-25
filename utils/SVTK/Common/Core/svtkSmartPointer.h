/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSmartPointer.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkSmartPointer
 * @brief   Hold a reference to a svtkObjectBase instance.
 *
 * svtkSmartPointer is a class template that provides automatic casting
 * for objects held by the svtkSmartPointerBase superclass.
 */

#ifndef svtkSmartPointer_h
#define svtkSmartPointer_h

#include "svtkSmartPointerBase.h"

#include "svtkMeta.h" // for IsComplete
#include "svtkNew.h"  // for svtkNew.h

#include <type_traits> // for is_base_of
#include <utility>     // for std::move

template <class T>
class svtkSmartPointer : public svtkSmartPointerBase
{
  // These static asserts only fire when the function calling CheckTypes is
  // used. Thus, this smart pointer class may still be used as a member variable
  // with a forward declared T, so long as T is defined by the time the calling
  // function is used.
  template <typename U = T>
  static void CheckTypes() noexcept
  {
    static_assert(svtk::detail::IsComplete<T>::value,
      "svtkSmartPointer<T>'s T type has not been defined. Missing "
      "include?");
    static_assert(svtk::detail::IsComplete<U>::value,
      "Cannot store an object with undefined type in "
      "svtkSmartPointer. Missing include?");
    static_assert(std::is_base_of<T, U>::value,
      "Argument type is not compatible with svtkSmartPointer<T>'s "
      "T type.");
    static_assert(std::is_base_of<svtkObjectBase, T>::value,
      "svtkSmartPointer can only be used with subclasses of "
      "svtkObjectBase.");
  }

public:
  /**
   * Initialize smart pointer to nullptr.
   */
  svtkSmartPointer() noexcept : svtkSmartPointerBase() {}

  /**
   * Initialize smart pointer with a new reference to the same object
   * referenced by given smart pointer.
   * @{
   */
  // Need both overloads because the copy-constructor must be non-templated:
  svtkSmartPointer(const svtkSmartPointer& r)
    : svtkSmartPointerBase(r)
  {
  }

  template <class U>
  svtkSmartPointer(const svtkSmartPointer<U>& r)
    : svtkSmartPointerBase(r)
  {
    svtkSmartPointer::CheckTypes<U>();
  }
  /* @} **/

  /**
   * Move the contents of @a r into @a this.
   * @{
   */
  // Need both overloads because the move-constructor must be non-templated:
  svtkSmartPointer(svtkSmartPointer&& r) noexcept : svtkSmartPointerBase(std::move(r)) {}

  template <class U>
  svtkSmartPointer(svtkSmartPointer<U>&& r) noexcept : svtkSmartPointerBase(std::move(r))
  {
    svtkSmartPointer::CheckTypes<U>();
  }
  /**@}*/

  /**
   * Initialize smart pointer to given object.
   * @{
   */
  svtkSmartPointer(T* r)
    : svtkSmartPointerBase(r)
  {
    svtkSmartPointer::CheckTypes();
  }

  template <typename U>
  svtkSmartPointer(const svtkNew<U>& r)
    : svtkSmartPointerBase(r.Object)
  { // Create a new reference on copy
    svtkSmartPointer::CheckTypes<U>();
  }
  //@}

  /**
   * Move the pointer from the svtkNew smart pointer to the new svtkSmartPointer,
   * stealing its reference and resetting the svtkNew object to nullptr.
   */
  template <typename U>
  svtkSmartPointer(svtkNew<U>&& r) noexcept
    : svtkSmartPointerBase(r.Object, svtkSmartPointerBase::NoReference{})
  { // Steal the reference on move
    svtkSmartPointer::CheckTypes<U>();

    r.Object = nullptr;
  }

  //@{
  /**
   * Assign object to reference.  This removes any reference to an old
   * object.
   */
  // Need this since the compiler won't recognize template functions as
  // assignment operators.
  svtkSmartPointer& operator=(const svtkSmartPointer& r)
  {
    this->svtkSmartPointerBase::operator=(r.GetPointer());
    return *this;
  }

  template <class U>
  svtkSmartPointer& operator=(const svtkSmartPointer<U>& r)
  {
    svtkSmartPointer::CheckTypes<U>();

    this->svtkSmartPointerBase::operator=(r.GetPointer());
    return *this;
  }
  //@}

  /**
   * Assign object to reference.  This removes any reference to an old
   * object.
   */
  template <typename U>
  svtkSmartPointer& operator=(const svtkNew<U>& r)
  {
    svtkSmartPointer::CheckTypes<U>();

    this->svtkSmartPointerBase::operator=(r.Object);
    return *this;
  }

  /**
   * Assign object to reference.  This adds a new reference to an old
   * object.
   */
  template <typename U>
  svtkSmartPointer& operator=(U* r)
  {
    svtkSmartPointer::CheckTypes<U>();

    this->svtkSmartPointerBase::operator=(r);
    return *this;
  }

  //@{
  /**
   * Get the contained pointer.
   */
  T* GetPointer() const noexcept { return static_cast<T*>(this->Object); }
  T* Get() const noexcept { return static_cast<T*>(this->Object); }
  //@}

  /**
   * Get the contained pointer.
   */
  operator T*() const noexcept { return static_cast<T*>(this->Object); }

  /**
   * Dereference the pointer and return a reference to the contained
   * object.
   */
  T& operator*() const noexcept { return *static_cast<T*>(this->Object); }

  /**
   * Provides normal pointer target member access using operator ->.
   */
  T* operator->() const noexcept { return static_cast<T*>(this->Object); }

  /**
   * Transfer ownership of one reference to the given SVTK object to
   * this smart pointer.  This does not increment the reference count
   * of the object, but will decrement it later.  The caller is
   * effectively passing ownership of one reference to the smart
   * pointer.  This is useful for code like:

   * svtkSmartPointer<svtkFoo> foo;
   * foo.TakeReference(bar->NewFoo());

   * The input argument may not be another smart pointer.
   */
  void TakeReference(T* t) { *this = svtkSmartPointer<T>(t, NoReference()); }

  /**
   * Create an instance of a SVTK object.
   */
  static svtkSmartPointer<T> New() { return svtkSmartPointer<T>(T::New(), NoReference()); }

  /**
   * Create a new instance of the given SVTK object.
   */
  static svtkSmartPointer<T> NewInstance(T* t)
  {
    return svtkSmartPointer<T>(t->NewInstance(), NoReference());
  }

  /**
   * Transfer ownership of one reference to the given SVTK object to a
   * new smart pointer.  The returned smart pointer does not increment
   * the reference count of the object on construction but will
   * decrement it on destruction.  The caller is effectively passing
   * ownership of one reference to the smart pointer.  This is useful
   * for code like:

   * svtkSmartPointer<svtkFoo> foo =
   * svtkSmartPointer<svtkFoo>::Take(bar->NewFoo());

   * The input argument may not be another smart pointer.
   */
  static svtkSmartPointer<T> Take(T* t) { return svtkSmartPointer<T>(t, NoReference()); }

  // Work-around for HP and IBM overload resolution bug.  Since
  // NullPointerOnly is a private type the only pointer value that can
  // be passed by user code is a null pointer.  This operator will be
  // chosen by the compiler when comparing against null explicitly and
  // avoid the bogus ambiguous overload error.
#if defined(__HP_aCC) || defined(__IBMCPP__)
#define SVTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND(op)                                           \
  bool operator op(NullPointerOnly*) const { return ::operator op(*this, 0); }

private:
  class NullPointerOnly
  {
  };

public:
  SVTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND(==)
  SVTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND(!=)
  SVTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND(<)
  SVTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND(<=)
  SVTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND(>)
  SVTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND(>=)
#undef SVTK_SMART_POINTER_DEFINE_OPERATOR_WORKAROUND
#endif
protected:
  svtkSmartPointer(T* r, const NoReference& n)
    : svtkSmartPointerBase(r, n)
  {
  }

private:
  // These are purposely not implemented to prevent callers from
  // trying to take references from other smart pointers.
  void TakeReference(const svtkSmartPointerBase&) = delete;
  static void Take(const svtkSmartPointerBase&) = delete;
};

#define SVTK_SMART_POINTER_DEFINE_OPERATOR(op)                                                      \
  template <class T, class U>                                                                      \
  inline bool operator op(const svtkSmartPointer<T>& l, const svtkSmartPointer<U>& r)                \
  {                                                                                                \
    return (l.GetPointer() op r.GetPointer());                                                     \
  }                                                                                                \
  template <class T, class U>                                                                      \
  inline bool operator op(T* l, const svtkSmartPointer<U>& r)                                       \
  {                                                                                                \
    return (l op r.GetPointer());                                                                  \
  }                                                                                                \
  template <class T, class U>                                                                      \
  inline bool operator op(const svtkSmartPointer<T>& l, U* r)                                       \
  {                                                                                                \
    return (l.GetPointer() op r);                                                                  \
  }                                                                                                \
  template <class T, class U>                                                                      \
  inline bool operator op(const svtkNew<T>& l, const svtkSmartPointer<U>& r)                         \
  {                                                                                                \
    return (l.GetPointer() op r.GetPointer());                                                     \
  }                                                                                                \
  template <class T, class U>                                                                      \
  inline bool operator op(const svtkSmartPointer<T>& l, const svtkNew<U>& r)                         \
  {                                                                                                \
    return (l.GetPointer() op r.GetPointer);                                                       \
  }

/**
 * Compare smart pointer values.
 */
SVTK_SMART_POINTER_DEFINE_OPERATOR(==)
SVTK_SMART_POINTER_DEFINE_OPERATOR(!=)
SVTK_SMART_POINTER_DEFINE_OPERATOR(<)
SVTK_SMART_POINTER_DEFINE_OPERATOR(<=)
SVTK_SMART_POINTER_DEFINE_OPERATOR(>)
SVTK_SMART_POINTER_DEFINE_OPERATOR(>=)

#undef SVTK_SMART_POINTER_DEFINE_OPERATOR

namespace svtk
{

/// Construct a svtkSmartPointer<T> containing @a obj. A new reference is added
/// to @a obj.
template <typename T>
svtkSmartPointer<T> MakeSmartPointer(T* obj)
{
  return svtkSmartPointer<T>{ obj };
}

/// Construct a svtkSmartPointer<T> containing @a obj. @a obj's reference count
/// is not changed.
template <typename T>
svtkSmartPointer<T> TakeSmartPointer(T* obj)
{
  return svtkSmartPointer<T>::Take(obj);
}

} // end namespace svtk

/**
 * Streaming operator to print smart pointer like regular pointers.
 */
template <class T>
inline ostream& operator<<(ostream& os, const svtkSmartPointer<T>& p)
{
  return os << static_cast<const svtkSmartPointerBase&>(p);
}

#endif
// SVTK-HeaderTest-Exclude: svtkSmartPointer.h
