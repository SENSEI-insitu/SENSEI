/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCompositeDataSetNodeReference.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef svtkCompositeDataSetNodeReference_h
#define svtkCompositeDataSetNodeReference_h

#include "svtkCompositeDataIterator.h"
#include "svtkCompositeDataSet.h"
#include "svtkWeakPointer.h"

#include <cassert>
#include <type_traits>

#ifndef __SVTK_WRAP__

namespace svtk
{

namespace detail
{

//------------------------------------------------------------------------------
// MTimeWatcher:
// operator() return true if the MTime of its argument is less than or equal
// to the MTime of the object used to construct it.
//
// Create/reset using `mtime_watcher = MTimeWatcher{obj};`
//
// Test using `bool cacheIsValid = mtime_watcher(obj);`
//
// There are two variants of this:
// - MTimeWatcher can be used to ALWAYS check for valid mtimes.
// - DebugMTimeWatcher can be used to check mtimes ONLY in debugging builds,
//   and is defined as an empty, transparent no-op object in optimized builds.
//   The optimized version will always return true from operator().
struct MTimeWatcher
{
  svtkMTimeType MTime{ 0 };

  MTimeWatcher() {}
  explicit MTimeWatcher(svtkObject* o)
    : MTime{ o->GetMTime() }
  {
  }
  bool operator()(svtkObject* o) const { return o->GetMTime() <= this->MTime; }
  void Reset(svtkObject* o) { this->MTime = o->GetMTime(); }
  bool MTimeIsValid(svtkObject* o) const { return o->GetMTime() <= this->MTime; }
};

// empty, transparent, does nothing. operator() always returns true.
struct NoOpMTimeWatcher
{
  NoOpMTimeWatcher() {}
  explicit NoOpMTimeWatcher(svtkObject*) {}
  bool operator()(svtkObject*) const { return true; }
  void Reset(svtkObject*) {}
  bool MTimeIsValid(svtkObject*) const { return true; }
};

// Debug-dependent version:
#ifndef _NDEBUG
using DebugMTimeWatcher = MTimeWatcher;
#else
using DebugMTimeWatcher = NoOpMTimeWatcher;
#endif

//------------------------------------------------------------------------------
// DebugWeakPointer: Defined to svtkWeakPointer on debugging builds, T* on
// non-debugging builds.
#ifndef _NDEBUG
template <class ObjectType>
using DebugWeakPointer = svtkWeakPointer<ObjectType>;
#else
template <class ObjectType>
using DebugWeakPointer = ObjectType*;
#endif

} // end namespace detail

/**
 * A reference proxy into a svtkCompositeDataSet, obtained by dereferencing an
 * iterator from the svtk::Range(svtkCompositeDataSet*) overloads.
 *
 * This proxy may be used as a pointer, in which case it will forward the
 * currently pointed-to svtkDataObject*. This means that the following code is
 * legal:
 *
 * ```cpp
 * for (auto node : svtk::Range(cds))
 * { // decltype(node) == CompositeDataSetNodeReference
 *   if (node)                  // same as: if (node.GetDataObject() != nullptr)
 *   {
 *     assert(node->IsA("svtkDataObject"));     // node.GetDataObject()->IsA(...)
 *     node = nullptr;                         // node.SetDataObject(nullptr)
 *   }
 * }
 *
 * for (svtkDataObject *dObj : svtk::Range(cds))
 * {
 *   // Work with dObj
 * }
 * ```
 *
 * This allows for simple access to the objects in the composite dataset. If
 * more advanced operations are required, the CompositeDataSetNodeReference can:
 *
 * - Access the current svtkDataObject*:
 *   - `svtkDataObject* NodeReference::GetDataObject() const`
 *   - `NodeReference::operator svtkDataObject* () const` (implicit conversion)
 *   - `svtkDataObject* NodeReference::operator->() const` (arrow operator)
 * - Replace the current svtkDataObject* in the composite dataset:
 *   - `void NodeReference::SetDataObject(svtkDataObject*)`
 *   - `NodeReference& NodeReference::operator=(svtkDataObject*)` (assignment)
 * - SetGet the svtkDataObject at the same position in another composite dataset
 *   - `void NodeReference::SetDataObject(svtkCompositeDataSet*, svtkDataObject*)`
 *   - `svtkDataObject* NodeReference::GetDataObject(svtkCompositeDataSet*) const`
 * - Check and access node metadata (if any):
 *   - `bool NodeReference::HasMetaData() const`
 *   - `svtkInformation* NodeReference::GetMetaData() const`
 * - Get the current flat index within the parent range:
 *   - `unsigned int NodeReference::GetFlatIndex() const`
 *
 * Assigning one reference to another assigns the svtkDataObject* pointer to the
 * target reference. Assigning to non-leaf nodes invalidates all iterators /
 * references.
 *
 * Equality testing compares each reference's DataObject and FlatIndex.
 *
 * @warning The NodeReference shares state with the OwnerType iterator that
 * generates it. Incrementing or destroying the parent iterator will invalidate
 * the reference. In debugging builds, these misuses will be caught via runtime
 * assertions.
 */
template <typename IteratorType,
  typename OwnerType>
class CompositeDataSetNodeReference
  : private detail::DebugMTimeWatcher // empty-base optimization when NDEBUG
{
  static_assert(std::is_base_of<svtkCompositeDataIterator, IteratorType>::value,
    "CompositeDataSetNodeReference's IteratorType must be a "
    "subclass of svtkCompositeDataIterator.");

  // Either a svtkWeakPointer (debug builds) or raw pointer (non-debug builds)
  mutable detail::DebugWeakPointer<IteratorType> Iterator{ nullptr };

  // Check that the reference has not been invalidated by having the
  // borrowed internal iterator modified.
  void AssertValid() const
  {

    // Test that the weak pointer hasn't been cleared
    assert(
      "Invalid CompositeDataNodeReference accessed (iterator freed)." && this->Iterator != nullptr);
    // Check MTime:
    assert("Invalid CompositeDataNodeReference accessed (iterator modified)." &&
      this->MTimeIsValid(this->Iterator));
  }

protected:
  explicit CompositeDataSetNodeReference(IteratorType* iterator)
    : detail::DebugMTimeWatcher(iterator)
    , Iterator(iterator)
  {
  }

public:
  friend OwnerType; // To allow access to protected methods/base class

  CompositeDataSetNodeReference() = delete;
  CompositeDataSetNodeReference(const CompositeDataSetNodeReference& src) = default;
  CompositeDataSetNodeReference(CompositeDataSetNodeReference&&) noexcept = default;
  ~CompositeDataSetNodeReference() = default;

  // Assigns the DataObject from src to this:
  CompositeDataSetNodeReference& operator=(const CompositeDataSetNodeReference& src)
  {
    this->SetDataObject(src.GetDataObject());
    return *this;
  }

  // Compares data object and flat index:
  friend bool operator==(
    const CompositeDataSetNodeReference& lhs, const CompositeDataSetNodeReference& rhs)
  {
    return lhs.GetDataObject() == rhs.GetDataObject() && lhs.GetFlatIndex() == rhs.GetFlatIndex();
  }

  // Compares data object and flat index:
  friend bool operator!=(
    const CompositeDataSetNodeReference& lhs, const CompositeDataSetNodeReference& rhs)
  {
    return lhs != rhs;
  }

  svtkDataObject* GetDataObject() const
  {
    this->AssertValid();
    // GetCurrentDataObject is buggy -- the iterator caches the current dataset
    // internally, so if the object has changed since the iterator was
    // incremented, the changes will not be visible through the iterator's
    // API. See SVTK issue #17529.
    // Instead, look it up in the dataset. It's a bit slower, but will always be
    // correct.
    //    return this->Iterator->GetCurrentDataObject();
    return this->Iterator->GetDataSet()->GetDataSet(this->Iterator);
  }

  svtkDataObject* GetDataObject(svtkCompositeDataSet* other)
  {
    this->AssertValid();
    return other->GetDataSet(this->Iterator);
  }

  operator bool() const { return this->GetDataObject() != nullptr; }

  operator svtkDataObject*() const { return this->GetDataObject(); }

  svtkDataObject* operator->() const { return this->GetDataObject(); }

  void SetDataObject(svtkDataObject* obj)
  {
    this->AssertValid();
    svtkCompositeDataSet* cds = this->Iterator->GetDataSet();
    cds->SetDataSet(this->Iterator, obj);
  }

  void SetDataObject(svtkCompositeDataSet* other, svtkDataObject* dObj)
  {
    this->AssertValid();
    other->SetDataSet(this->Iterator, dObj);
  }

  CompositeDataSetNodeReference& operator=(svtkDataObject* obj)
  {
    this->SetDataObject(obj);
    return *this;
  }

  unsigned int GetFlatIndex() const
  {
    this->AssertValid();
    return this->Iterator->GetCurrentFlatIndex();
  }

  bool HasMetaData() const
  {
    this->AssertValid();
    return this->Iterator->HasCurrentMetaData() != 0;
  }

  svtkInformation* GetMetaData() const
  {
    this->AssertValid();
    return this->Iterator->GetCurrentMetaData();
  }
};

} // end namespace svtk

#endif // __SVTK_WRAP__

#endif // svtkCompositeDataSetNodeReference_h

// SVTK-HeaderTest-Exclude: svtkCompositeDataSetNodeReference.h
