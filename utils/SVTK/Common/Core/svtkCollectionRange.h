/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCollectionRange.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef svtkCollectionRange_h
#define svtkCollectionRange_h

#ifndef __SVTK_WRAP__

#include "svtkCollection.h"
#include "svtkMeta.h"
#include "svtkRange.h"
#include "svtkSmartPointer.h"
#include "svtkIterator.h"

#include <cassert>

namespace svtk
{
namespace detail
{

template <typename CollectionType>
struct CollectionRange;
template <typename CollectionType>
struct CollectionIterator;

//------------------------------------------------------------------------------
// Detect svtkCollection types
template <typename T>
struct IsCollection : std::is_base_of<svtkCollection, T>
{
};

template <typename CollectionType, typename T = CollectionType>
using EnableIfIsCollection = typename std::enable_if<IsCollection<CollectionType>::value, T>::type;

//------------------------------------------------------------------------------
// Detect the type of items held by the collection by checking the return type
// of GetNextItem(), or GetNextItemAsObject() as a fallback.
template <typename CollectionType>
struct GetCollectionItemType
{
  static_assert(IsCollection<CollectionType>::value, "Invalid svtkCollection subclass.");

private:
  // The GetType methods are only used in a decltype context and are left
  // unimplemented as we only care about their signatures. They are used to
  // determine the type of object held by the collection.
  //
  // By passing literal 0 as the argument, the overload taking `int` is
  // preferred and returns the same type as CollectionType::GetNextItem, which
  // is usually the exact type held by the collection (e.g.
  // svtkRendererCollection::GetNextItem returns svtkRenderer*).
  //
  // If the collection class does not define GetNextItem, SFINAE removes the
  // preferred `int` overload, and the `...` overload is used instead. This
  // method returns the same type as svtkCollection::GetNextItemAsObject, which
  // is svtkObject*. This lets us define a more derived collection item type
  // when possible, while falling back to the general svtkObject if a more
  // refined type is not known.

  // not implemented
  template <typename T>
  static auto GetType(...) -> decltype(std::declval<T>().GetNextItemAsObject());

  // not implemented
  template <typename T>
  static auto GetType(int) -> decltype(std::declval<T>().GetNextItem());

  using PointerType = decltype(GetType<CollectionType>(0));

public:
  // Just use std::remove pointer, svtk::detail::StripPointer is overkill.
  using Type = typename std::remove_pointer<PointerType>::type;
};

//------------------------------------------------------------------------------
// Collection iterator. Reference, value, and pointer types are all ItemType
// pointers, since:
// a) values: ItemType* instead of ItemType because svtkObjects can't be
//    copied/assigned.
// b) references: No good usecase to change the pointers held by the collection
//    by returning ItemType*&, nor would returning ItemType& be useful, since
//    it'd have to be dereferenced anyway to pass it anywhere, and svtkObjects
//    are conventionally held by address.
// c) pointers: Returning ItemType** from operator-> would be useless.
//
// There are no const_reference, etc, since SVTK is not const correct and marking
// svtkObjects consts makes them unusable.
template <typename CollectionType>
struct CollectionIterator
  : public svtkIterator<std::forward_iterator_tag,
      typename GetCollectionItemType<CollectionType>::Type*, int,
      typename GetCollectionItemType<CollectionType>::Type*,
      typename GetCollectionItemType<CollectionType>::Type*>
{
  static_assert(IsCollection<CollectionType>::value, "Invalid svtkCollection subclass.");

private:
  using ItemType = typename GetCollectionItemType<CollectionType>::Type;
  using Superclass = svtkIterator<std::forward_iterator_tag, ItemType*, int, ItemType*, ItemType*>;

public:
  using iterator_category = typename Superclass::iterator_category;
  using value_type = typename Superclass::value_type;
  using difference_type = typename Superclass::difference_type;
  using pointer = typename Superclass::pointer;
  using reference = typename Superclass::reference;

  CollectionIterator() noexcept : Element(nullptr) {}

  CollectionIterator(const CollectionIterator& o) noexcept = default;
  CollectionIterator& operator=(const CollectionIterator& o) noexcept = default;

  CollectionIterator& operator++() noexcept // prefix
  {
    this->Increment();
    return *this;
  }

  CollectionIterator operator++(int) noexcept // postfix
  {
    auto elem = this->Element;
    this->Increment();
    return CollectionIterator{ elem };
  }

  reference operator*() const noexcept { return this->GetItem(); }

  pointer operator->() const noexcept { return this->GetItem(); }

  friend bool operator==(const CollectionIterator& lhs, const CollectionIterator& rhs) noexcept
  {
    return lhs.Element == rhs.Element;
  }

  friend bool operator!=(const CollectionIterator& lhs, const CollectionIterator& rhs) noexcept
  {
    return lhs.Element != rhs.Element;
  }

  friend void swap(CollectionIterator& lhs, CollectionIterator& rhs) noexcept
  {
    using std::swap;
    swap(lhs.Element, rhs.Element);
  }

  friend struct CollectionRange<CollectionType>;

protected:
  CollectionIterator(svtkCollectionElement* element) noexcept : Element(element) {}

private:
  void Increment() noexcept
  { // incrementing an invalid iterator is UB, no need to check for non-null.
    this->Element = this->Element->Next;
  }

  ItemType* GetItem() const noexcept { return static_cast<ItemType*>(this->Element->Item); }

  svtkCollectionElement* Element;
};

//------------------------------------------------------------------------------
// Collection range proxy.
// The const_iterators/references are the same as the non-const versions, since
// svtkObjects marked const are unusable.
template <typename CollectionType>
struct CollectionRange
{
  static_assert(IsCollection<CollectionType>::value, "Invalid svtkCollection subclass.");

  using ItemType = typename GetCollectionItemType<CollectionType>::Type;

  // NOTE: The const items are the same as the mutable ones, since const
  // svtkObjects are generally unusable.
  using size_type = int; // int is used by the svtkCollection API.
  using iterator = CollectionIterator<CollectionType>;
  using const_iterator = CollectionIterator<CollectionType>;
  using reference = ItemType*;
  using const_reference = ItemType*;
  using value_type = ItemType*;

  CollectionRange(CollectionType* coll) noexcept : Collection(coll) { assert(this->Collection); }

  CollectionType* GetCollection() const noexcept { return this->Collection; }

  size_type size() const noexcept { return this->Collection->GetNumberOfItems(); }

  iterator begin() const
  {
    svtkCollectionSimpleIterator cookie;
    this->Collection->InitTraversal(cookie);
    // The cookie is a linked list node pointer, svtkCollectionElement:
    return iterator{ static_cast<svtkCollectionElement*>(cookie) };
  }

  iterator end() const { return iterator{ nullptr }; }

  // Note: These return mutable objects because const svtkObject are unusable.
  const_iterator cbegin() const
  {
    svtkCollectionSimpleIterator cookie;
    this->Collection->InitTraversal(cookie);
    // The cookie is a linked list node pointer, svtkCollectionElement:
    return const_iterator{ static_cast<svtkCollectionElement*>(cookie) };
  }

  // Note: These return mutable objects because const svtkObjects are unusable.
  const_iterator cend() const { return const_iterator{ nullptr }; }

private:
  svtkSmartPointer<CollectionType> Collection;
};

}
} // end namespace svtk::detail

#endif // __SVTK_WRAP__

#endif // svtkCollectionRange_h

// SVTK-HeaderTest-Exclude: svtkCollectionRange.h
