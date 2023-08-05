/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCompositeDataSetRange.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef svtkCompositeDataSetRange_h
#define svtkCompositeDataSetRange_h

#include "svtkCompositeDataIterator.h"
#include "svtkCompositeDataSet.h"
#include "svtkCompositeDataSetNodeReference.h"
#include "svtkMeta.h"
#include "svtkRange.h"
#include "svtkSmartPointer.h"
#include "svtkIterator.h"

#include <cassert>

#ifndef __SVTK_WRAP__

namespace svtk
{

// Pass these to svtk::Range(cds, options):
enum class CompositeDataSetOptions : unsigned int
{
  None = 0,
  SkipEmptyNodes = 1 << 1 // Skip null datasets.
};

} // end namespace svtk (for bitflag op definition)

SVTK_GENERATE_BITFLAG_OPS(svtk::CompositeDataSetOptions)

namespace svtk
{

namespace detail
{

struct CompositeDataSetRange;
struct CompositeDataSetIterator;

using CompositeDataSetIteratorReference =
  svtk::CompositeDataSetNodeReference<svtkCompositeDataIterator, CompositeDataSetIterator>;

//------------------------------------------------------------------------------
// svtkCompositeDataSet iterator. Returns svtk::CompositeDataSetNodeReference.
struct CompositeDataSetIterator
  : public svtkIterator<std::forward_iterator_tag, svtkDataObject*, int,
      CompositeDataSetIteratorReference, CompositeDataSetIteratorReference>
{
private:
  using Superclass = svtkIterator<std::forward_iterator_tag, svtkDataObject*, int,
    CompositeDataSetIteratorReference, CompositeDataSetIteratorReference>;
  using InternalIterator = svtkCompositeDataIterator;
  using SmartIterator = svtkSmartPointer<InternalIterator>;

public:
  using iterator_category = typename Superclass::iterator_category;
  using value_type = typename Superclass::value_type;
  using difference_type = typename Superclass::difference_type;
  using pointer = typename Superclass::pointer;
  using reference = typename Superclass::reference;

  CompositeDataSetIterator(const CompositeDataSetIterator& o)
    : Iterator(o.Iterator ? SmartIterator::Take(o.Iterator->NewInstance()) : nullptr)
  {
    this->CopyState(o.Iterator);
  }

  CompositeDataSetIterator(CompositeDataSetIterator&&) noexcept = default;

  CompositeDataSetIterator& operator=(const CompositeDataSetIterator& o)
  {
    this->Iterator = o.Iterator ? SmartIterator::Take(o.Iterator->NewInstance()) : nullptr;
    this->CopyState(o.Iterator);
    return *this;
  }

  CompositeDataSetIterator& operator++() // prefix
  {
    this->Increment();
    return *this;
  }

  CompositeDataSetIterator operator++(int) // postfix
  {
    CompositeDataSetIterator other(*this);
    this->Increment();
    return other;
  }

  reference operator*() const { return this->GetData(); }

  pointer operator->() const { return this->GetData(); }

  friend bool operator==(const CompositeDataSetIterator& lhs, const CompositeDataSetIterator& rhs)
  {
    // A null internal iterator means it is an 'end' sentinal.
    InternalIterator* l = lhs.Iterator;
    InternalIterator* r = rhs.Iterator;

    if (!r && !l)
    { // end == end
      return true;
    }
    else if (!r)
    { // right is end
      return l->IsDoneWithTraversal() != 0;
    }
    else if (!l)
    { // left is end
      return r->IsDoneWithTraversal() != 0;
    }
    else
    { // Both iterators are valid, check unique idx:
      return r->GetCurrentFlatIndex() == l->GetCurrentFlatIndex();
    }
  }

  friend bool operator!=(const CompositeDataSetIterator& lhs, const CompositeDataSetIterator& rhs)
  {
    return !(lhs == rhs); // let the compiler handle this one =)
  }

  friend void swap(CompositeDataSetIterator& lhs, CompositeDataSetIterator& rhs) noexcept
  {
    using std::swap;
    swap(lhs.Iterator, rhs.Iterator);
  }

  friend struct CompositeDataSetRange;

protected:
  // Note: This takes ownership of iter and manages its lifetime.
  // Iter should not be used past this point by the caller.
  CompositeDataSetIterator(SmartIterator&& iter) noexcept : Iterator(std::move(iter)) {}

  // Note: Iterators constructed using this ctor will be considered
  // 'end' iterators via a sentinal pattern.
  CompositeDataSetIterator() noexcept : Iterator(nullptr) {}

private:
  void CopyState(InternalIterator* source)
  {
    if (source)
    {
      assert(this->Iterator != nullptr);
      this->Iterator->SetDataSet(source->GetDataSet());
      this->Iterator->SetSkipEmptyNodes(source->GetSkipEmptyNodes());
      this->Iterator->InitTraversal();
      this->AdvanceTo(source->GetCurrentFlatIndex());
    }
  }

  void AdvanceTo(const unsigned int flatIdx)
  {
    assert(this->Iterator != nullptr);
    assert(this->Iterator->GetCurrentFlatIndex() <= flatIdx);
    while (this->Iterator->GetCurrentFlatIndex() < flatIdx)
    {
      this->Increment();
    }
  }

  void Increment()
  {
    assert(this->Iterator != nullptr);
    assert(!this->Iterator->IsDoneWithTraversal());
    this->Iterator->GoToNextItem();
  }

  CompositeDataSetIteratorReference GetData() const
  {
    assert(this->Iterator != nullptr);
    assert(!this->Iterator->IsDoneWithTraversal());
    return CompositeDataSetIteratorReference{ this->Iterator };
  }

  mutable SmartIterator Iterator;
};

//------------------------------------------------------------------------------
// CompositeDataSet range proxy.
// The const_iterators/references are the same as the non-const versions, since
// svtkObjects marked const are unusable.
struct CompositeDataSetRange
{
private:
  using InternalIterator = svtkCompositeDataIterator;
  using SmartIterator = svtkSmartPointer<InternalIterator>;

public:
  using size_type = int;
  using iterator = CompositeDataSetIterator;
  using const_iterator = CompositeDataSetIterator;
  using reference = CompositeDataSetIteratorReference;
  using const_reference = const CompositeDataSetIteratorReference;
  using value_type = svtkDataObject*;

  CompositeDataSetRange(
    svtkCompositeDataSet* cds, CompositeDataSetOptions opts = CompositeDataSetOptions::None)
    : CompositeDataSet(cds)
    , Options(opts)
  {
    assert(this->CompositeDataSet);
  }

  svtkCompositeDataSet* GetCompositeDataSet() const noexcept { return this->CompositeDataSet; }

  CompositeDataSetOptions GetOptions() const noexcept { return this->Options; }

  // This is O(N), since the size requires traversal due to various options.
  size_type size() const
  {
    size_type result = 0;
    auto iter = this->NewIterator();
    iter->InitTraversal();
    while (!iter->IsDoneWithTraversal())
    {
      ++result;
      iter->GoToNextItem();
    }
    return result;
  }

  iterator begin() const { return CompositeDataSetIterator{ this->NewIterator() }; }

  iterator end() const { return CompositeDataSetIterator{}; }

  // Note: These return mutable objects because const svtkObject are unusable.
  const_iterator cbegin() const { return CompositeDataSetIterator{ this->NewIterator() }; }

  // Note: These return mutable objects because const svtkObjects are unusable.
  const_iterator cend() const { return CompositeDataSetIterator{}; }

private:
  SmartIterator NewIterator() const
  {
    using Opts = svtk::CompositeDataSetOptions;

    auto result = SmartIterator::Take(this->CompositeDataSet->NewIterator());
    result->SetSkipEmptyNodes((this->Options & Opts::SkipEmptyNodes) != Opts::None);
    result->InitTraversal();
    return result;
  }

  mutable svtkSmartPointer<svtkCompositeDataSet> CompositeDataSet;
  CompositeDataSetOptions Options;
};

}
} // end namespace svtk::detail

#endif // __SVTK_WRAP__

#endif // svtkCompositeDataSetRange_h

// SVTK-HeaderTest-Exclude: svtkCompositeDataSetRange.h
