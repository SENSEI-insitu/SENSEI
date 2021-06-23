/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkRange.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef svtkRange_h
#define svtkRange_h

#include "svtkMeta.h"
#include "svtkRangeIterableTraits.h"

#include <iterator>
#include <type_traits>
#include <utility>

namespace svtk
{

/**
 * Generate an iterable STL proxy object for a SVTK container.
 *
 * Currently supports:
 *
 * - svtkCollection and subclasses (`#include <svtkCollectionRange.h>`):
 *   - ItemType is the (non-pointer) result type of GetNextItem() if this method
 *     exists on the collection type, otherwise svtkObject is used.
 *   - Iterators fulfill the STL InputIterator concept with some exceptions:
 *     - Const iterators/references are mutable, since svtkObjects are generally
 *       unusable when const.
 *     - Value/pointer/reference types are just ItemType*, since:
 *       - Plain ItemType wouldn't be usable (svtkObjects cannot be
 *         copied/assigned)
 *       - ItemType*& references aren't generally desired.
 *       - ItemType& references are unconventional for svtkObjects.
 *       - ItemType** pointers are unruly.
 *
 * - svtkCompositeDataSet (`#include <svtkCompositeDataSetRange.h>`)
 *   - svtk::CompositeDataSetOptions: None, SkipEmptyNodes.
 *     - Ex. svtk::Range(compDS, svtk::CompositeDataSetOptions::SkipEmptyNodes);
 *   - Reverse iteration is not supported. Use svtkCompositeDataIterator directly
 *     instead for this.
 *   - Dereferencing the iterator yields a svtk::CompositeDataSetNodeReference
 *     that provides additional API to get the node's flat index, data object,
 *     and metadata. See that class's documentation for more information.
 *
 * - svtkDataObjectTree (`#include <svtkDataObjectTreeRange.h>`)
 *   - svtk::DataObjectTreeOptions:
 *     None, SkipEmptyNodes, VisitOnlyLeaves, TraverseSubTree.
 *     - Ex. svtk::Range(dObjTree, svtk::DataObjectTreeOptions::TraverseSubTree |
 *                                svtk::DataObjectTreeOptions::SkipEmptyNodes);
 *   - Reverse iteration is not supported. Use svtkDataObjectTreeIterator
 *     directly instead for this.
 *   - Dereferencing the iterator yields a svtk::CompositeDataSetNodeReference
 *     that provides additional API to get the node's flat index, data object,
 *     and metadata. See that class's documentation for more information.
 *
 * Usage:
 *
 * ```
 * for (auto item : svtk::Range(myCollection))
 * {
 *   // Use item.
 * }
 *
 * // or:
 *
 * using Opts = svtk::svtkDataObjectTreeOptions;
 * auto range = svtk::Range(dataObjTree,
 *                         Opts::TraverseSubTree | Opts::VisitOnlyLeaves);
 * some_algo(range.begin(), range.end());
 *
 * ```
 */
template <typename IterablePtr, typename... Options>
auto Range(IterablePtr iterable, Options&&... opts) ->
  typename detail::IterableTraits<typename detail::StripPointers<IterablePtr>::type>::RangeType
{
  using Iterable = typename detail::StripPointers<IterablePtr>::type;
  using RangeType = typename detail::IterableTraits<Iterable>::RangeType;
  return RangeType{ iterable, std::forward<Options>(opts)... };
}

} // end namespace svtk

#endif // svtkRange_h

// SVTK-HeaderTest-Exclude: svtkRange.h
