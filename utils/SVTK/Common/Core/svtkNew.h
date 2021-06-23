/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkNew.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkNew
 * @brief   Allocate and hold a SVTK object.
 *
 * svtkNew is a class template that on construction allocates and
 * initializes an instance of its template argument using T::New().
 * It assumes ownership of one reference during its lifetime, and calls
 * T->Delete() on destruction.
 *
 * Automatic casting to raw pointer is available for convenience, but
 * users of this method should ensure that they do not
 * return this pointer if the svtkNew will go out of scope without
 * incrementing its reference count.
 *
 * svtkNew is a drop in replacement for svtkSmartPointer, for example,
 *
 * \code
 * svtkNew<svtkRenderer> ren;
 * svtkNew<svtkRenderWindow> renWin;
 * renWin->AddRenderer(ren);
 * svtkNew<svtkRenderWindowInteractor> iren;
 * iren->SetRenderWindow(renWin);
 * \endcode
 *
 *
 * @sa
 * svtkSmartPointer svtkWeakPointer
 */

#ifndef svtkNew_h
#define svtkNew_h

#include "svtkIOStream.h"
#include "svtkMeta.h" // for IsComplete

#include <type_traits> // for is_base_of

class svtkObjectBase;

template <class T>
class svtkNew
{
  // Allow other smart pointers friend access:
  template <typename U>
  friend class svtkNew;
  template <typename U>
  friend class svtkSmartPointer;
  template <typename U>
  friend class svtkWeakPointer;

  // These static asserts only fire when the function calling CheckTypes is
  // used. Thus, this smart pointer class may still be used as a member variable
  // with a forward declared T, so long as T is defined by the time the calling
  // function is used.
  template <typename U = T>
  static void CheckTypes() noexcept
  {
    static_assert(svtk::detail::IsComplete<T>::value,
      "svtkNew<T>'s T type has not been defined. Missing include?");
    static_assert(svtk::detail::IsComplete<U>::value,
      "Cannot store an object with undefined type in "
      "svtkNew. Missing include?");
    static_assert(std::is_base_of<T, U>::value,
      "Argument type is not compatible with svtkNew<T>'s "
      "T type.");
    static_assert(std::is_base_of<svtkObjectBase, T>::value,
      "svtkNew can only be used with subclasses of svtkObjectBase.");
  }

public:
  /**
   * Create a new T on construction.
   */
  svtkNew()
    : Object(T::New())
  {
    svtkNew::CheckTypes();
  }

  /**
   * Move the object into the constructed svtkNew wrapper, stealing its
   * reference. The argument is reset to nullptr.
   * @{
   */
  svtkNew(svtkNew&& o) noexcept : Object(o.Object) { o.Object = nullptr; }

  template <typename U>
  svtkNew(svtkNew<U>&& o) noexcept : Object(o.Object)
  {
    svtkNew::CheckTypes<U>();

    o.Object = nullptr;
  }
  //@}

  //@{
  /**
   * Deletes reference to instance of T.
   */
  ~svtkNew() { this->Reset(); }

  void Reset()
  {
    T* obj = this->Object;
    if (obj)
    {
      this->Object = nullptr;
      obj->Delete();
    }
  }
  //@}

  /**
   * Enable pointer-like dereference syntax. Returns a pointer to the contained
   * object.
   */
  T* operator->() const noexcept { return this->Object; }

  //@{
  /**
   * Get a raw pointer to the contained object. When using this function be
   * careful that the reference count does not drop to 0 when using the pointer
   * returned. This will happen when the svtkNew object goes out of
   * scope for example.
   */
  T* GetPointer() const noexcept { return this->Object; }
  T* Get() const noexcept { return this->Object; }
  operator T*() const noexcept { return static_cast<T*>(this->Object); }
  //@}
  /**
   * Dereference the pointer and return a reference to the contained object.
   * When using this function be careful that the reference count does not
   * drop to 0 when using the pointer returned.
   * This will happen when the svtkNew object goes out of scope for example.
   */
  T& operator*() const noexcept { return *static_cast<T*>(this->Object); }

private:
  svtkNew(svtkNew<T> const&) = delete;
  void operator=(svtkNew<T> const&) = delete;
  T* Object;
};

#endif
// SVTK-HeaderTest-Exclude: svtkNew.h
