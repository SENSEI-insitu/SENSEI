/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkWeakPointer.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkWeakPointer
 * @brief   a weak reference to a svtkObject.
 *
 * A weak reference to a svtkObject, which means that assigning
 * a svtkObject to the svtkWeakPointer does not affect the reference count of the
 * svtkObject. However, when the svtkObject is destroyed, the svtkWeakPointer gets
 * initialized to nullptr, thus avoiding any dangling references.
 *
 * \code
 * svtkTable *table = svtkTable::New();
 * svtkWeakPointer<svtkTable> weakTable = table;
 * \endcode
 *
 * Some time later the table may be deleted, but if it is tested for null then
 * the weak pointer will not leave a dangling pointer.
 *
 * \code
 * table->Delete();
 * if (weakTable)
 *   {
 *   // Never executed as the weak table pointer will be null here
 *   cout << "Number of columns in table: " << weakTable->GetNumberOfColumns()
 *        << endl;
 *   }
 * \endcode
 */

#ifndef svtkWeakPointer_h
#define svtkWeakPointer_h

#include "svtkWeakPointerBase.h"

#include "svtkMeta.h" // for IsComplete
#include "svtkNew.h"  // for svtkNew

#include <type_traits> // for is_base_of
#include <utility>     // for std::move

template <class T>
class svtkWeakPointer : public svtkWeakPointerBase
{
  // These static asserts only fire when the function calling CheckTypes is
  // used. Thus, this smart pointer class may still be used as a member variable
  // with a forward declared T, so long as T is defined by the time the calling
  // function is used.
  template <typename U = T>
  static void CheckTypes() noexcept
  {
    static_assert(svtk::detail::IsComplete<T>::value,
      "svtkWeakPointer<T>'s T type has not been defined. Missing "
      "include?");
    static_assert(svtk::detail::IsComplete<U>::value,
      "Cannot store an object with undefined type in "
      "svtkWeakPointer. Missing include?");
    static_assert(std::is_base_of<T, U>::value,
      "Argument type is not compatible with svtkWeakPointer<T>'s "
      "T type.");
    static_assert(std::is_base_of<svtkObjectBase, T>::value,
      "svtkWeakPointer can only be used with subclasses of "
      "svtkObjectBase.");
  }

public:
  /**
   * Initialize smart pointer to nullptr.
   */
  svtkWeakPointer() noexcept : svtkWeakPointerBase() {}

  /**
   * Initialize smart pointer with the given smart pointer.
   * @{
   */
  svtkWeakPointer(const svtkWeakPointer& r)
    : svtkWeakPointerBase(r)
  {
  }

  template <class U>
  svtkWeakPointer(const svtkWeakPointer<U>& r)
    : svtkWeakPointerBase(r)
  {
    svtkWeakPointer::CheckTypes<U>();
  }
  /* @} **/

  /**
   * Move r's object into the new weak pointer, setting r to nullptr.
   * @{
   */
  svtkWeakPointer(svtkWeakPointer&& r) noexcept : svtkWeakPointerBase(std::move(r)) {}

  template <class U>
  svtkWeakPointer(svtkWeakPointer<U>&& r) noexcept : svtkWeakPointerBase(std::move(r))
  {
    svtkWeakPointer::CheckTypes<U>();
  }
  /* @} **/

  /**
   * Initialize smart pointer to given object.
   * @{
   */
  svtkWeakPointer(T* r)
    : svtkWeakPointerBase(r)
  {
    svtkWeakPointer::CheckTypes();
  }

  template <typename U>
  svtkWeakPointer(const svtkNew<U>& r)
    : svtkWeakPointerBase(r.Object)
  { // Create a new reference on copy
    svtkWeakPointer::CheckTypes<U>();
  }
  //@}

  //@{
  /**
   * Assign object to reference.
   */
  svtkWeakPointer& operator=(const svtkWeakPointer& r)
  {
    this->svtkWeakPointerBase::operator=(r);
    return *this;
  }

  template <class U>
  svtkWeakPointer& operator=(const svtkWeakPointer<U>& r)
  {
    svtkWeakPointer::CheckTypes<U>();

    this->svtkWeakPointerBase::operator=(r);
    return *this;
  }
  //@}

  //@{
  /**
   * Move r's object into this weak pointer, setting r to nullptr.
   */
  svtkWeakPointer& operator=(svtkWeakPointer&& r) noexcept
  {
    this->svtkWeakPointerBase::operator=(std::move(r));
    return *this;
  }

  template <class U>
  svtkWeakPointer& operator=(svtkWeakPointer<U>&& r) noexcept
  {
    svtkWeakPointer::CheckTypes<U>();

    this->svtkWeakPointerBase::operator=(std::move(r));
    return *this;
  }
  //@}

  //@{
  /**
   * Assign object to reference.
   */
  svtkWeakPointer& operator=(T* r)
  {
    svtkWeakPointer::CheckTypes();
    this->svtkWeakPointerBase::operator=(r);
    return *this;
  }

  template <typename U>
  svtkWeakPointer& operator=(const svtkNew<U>& r)
  {
    svtkWeakPointer::CheckTypes<U>();

    this->svtkWeakPointerBase::operator=(r.Object);
    return *this;
  }
  //@}

  //@{
  /**
   * Get the contained pointer.
   */
  T* GetPointer() const noexcept { return static_cast<T*>(this->Object); }
  T* Get() const noexcept { return static_cast<T*>(this->Object); }
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

  // Work-around for HP and IBM overload resolution bug.  Since
  // NullPointerOnly is a private type the only pointer value that can
  // be passed by user code is a null pointer.  This operator will be
  // chosen by the compiler when comparing against null explicitly and
  // avoid the bogus ambiguous overload error.
#if defined(__HP_aCC) || defined(__IBMCPP__)
#define SVTK_WEAK_POINTER_DEFINE_OPERATOR_WORKAROUND(op)                                            \
  bool operator op(NullPointerOnly*) const { return ::operator op(*this, 0); }

private:
  class NullPointerOnly
  {
  };

public:
  SVTK_WEAK_POINTER_DEFINE_OPERATOR_WORKAROUND(==)
  SVTK_WEAK_POINTER_DEFINE_OPERATOR_WORKAROUND(!=)
  SVTK_WEAK_POINTER_DEFINE_OPERATOR_WORKAROUND(<)
  SVTK_WEAK_POINTER_DEFINE_OPERATOR_WORKAROUND(<=)
  SVTK_WEAK_POINTER_DEFINE_OPERATOR_WORKAROUND(>)
  SVTK_WEAK_POINTER_DEFINE_OPERATOR_WORKAROUND(>=)
#undef SVTK_WEAK_POINTER_DEFINE_OPERATOR_WORKAROUND
#endif
protected:
  svtkWeakPointer(T* r, const NoReference& n)
    : svtkWeakPointerBase(r, n)
  {
  }

private:
  // These are purposely not implemented to prevent callers from
  // trying to take references from other smart pointers.
  void TakeReference(const svtkWeakPointerBase&) = delete;
  static void Take(const svtkWeakPointerBase&) = delete;
};

#define SVTK_WEAK_POINTER_DEFINE_OPERATOR(op)                                                       \
  template <class T, class U>                                                                      \
  inline bool operator op(const svtkWeakPointer<T>& l, const svtkWeakPointer<U>& r)                  \
  {                                                                                                \
    return (l.GetPointer() op r.GetPointer());                                                     \
  }                                                                                                \
  template <class T, class U>                                                                      \
  inline bool operator op(T* l, const svtkWeakPointer<U>& r)                                        \
  {                                                                                                \
    return (l op r.GetPointer());                                                                  \
  }                                                                                                \
  template <class T, class U>                                                                      \
  inline bool operator op(const svtkWeakPointer<T>& l, U* r)                                        \
  {                                                                                                \
    return (l.GetPointer() op r);                                                                  \
  }                                                                                                \
  template <class T, class U>                                                                      \
  inline bool operator op(const svtkNew<T>& l, const svtkWeakPointer<U>& r)                          \
  {                                                                                                \
    return (l.GetPointer() op r.GetPointer());                                                     \
  }                                                                                                \
  template <class T, class U>                                                                      \
  inline bool operator op(const svtkWeakPointer<T>& l, const svtkNew<U>& r)                          \
  {                                                                                                \
    return (l.GetPointer() op r.GetPointer);                                                       \
  }

/**
 * Compare smart pointer values.
 */
SVTK_WEAK_POINTER_DEFINE_OPERATOR(==)
SVTK_WEAK_POINTER_DEFINE_OPERATOR(!=)
SVTK_WEAK_POINTER_DEFINE_OPERATOR(<)
SVTK_WEAK_POINTER_DEFINE_OPERATOR(<=)
SVTK_WEAK_POINTER_DEFINE_OPERATOR(>)
SVTK_WEAK_POINTER_DEFINE_OPERATOR(>=)

#undef SVTK_WEAK_POINTER_DEFINE_OPERATOR

namespace svtk
{

/// Construct a svtkWeakPointer<T> containing @a obj. @a obj's reference count
/// is not changed.
template <typename T>
svtkWeakPointer<T> TakeWeakPointer(T* obj)
{
  return svtkWeakPointer<T>(obj);
}

} // end namespace svtk

/**
 * Streaming operator to print smart pointer like regular pointers.
 */
template <class T>
inline ostream& operator<<(ostream& os, const svtkWeakPointer<T>& p)
{
  return os << static_cast<const svtkWeakPointerBase&>(p);
}

#endif

// SVTK-HeaderTest-Exclude: svtkWeakPointer.h
