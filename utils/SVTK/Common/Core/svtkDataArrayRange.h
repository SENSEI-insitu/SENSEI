/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArrayRange.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @file svtkDataArrayRange.h
 * STL-compatible iterable ranges that provide access to svtkDataArray elements.
 *
 * @note Since the term 'range' is overloaded, it's worth pointing out that to
 * determine the value-range of an array's elements (an unrelated concept to
 * the Range objects defined here), see the svtkDataArray::GetRange and
 * svtkGenericDataArray::GetValueRange methods.
 */

#ifndef svtkDataArrayRange_h
#define svtkDataArrayRange_h

#include "svtkAOSDataArrayTemplate.h"
#include "svtkDataArray.h"
#include "svtkDataArrayMeta.h"
#include "svtkDataArrayTupleRange_AOS.h"
#include "svtkDataArrayTupleRange_Generic.h"
#include "svtkDataArrayValueRange_AOS.h"
#include "svtkDataArrayValueRange_Generic.h"
#include "svtkMeta.h"
#include "svtkSmartPointer.h"

#include <cassert>
#include <iterator>
#include <type_traits>

/**
 * @file svtkDataArrayRange.h
 *
 * The svtkDataArrayRange.h header provides utilities to convert svtkDataArrays
 * into "range" objects that behave like STL ranges. There are two types of
 * ranges: TupleRange and ValueRange.
 *
 * See Testing/Cxx/ExampleDataArrayRangeAPI.cxx for an illustrative example of
 * how these ranges and their associated iterators and references are used.
 *
 * These ranges unify the different memory layouts supported by SVTK and provide
 * a consistent interface to processing them with high efficiency. Whether a
 * range is constructed from a svtkDataArray, svtkFloatArray, or even
 * svtkScaledSOADataArrayTemplate, the same range-based algorithm implementation
 * can be used to provide the best performance possible using the input array's
 * API.
 *
 * Constructing a range using a derived subclass of svtkDataArray (such as
 * svtkFloatArray) will always give better performance than a range constructed
 * from a svtkDataArray pointer, since the svtkDataArray API requires virtual
 * calls and type conversion. Using a more derived type generally allows the
 * compiler to optimize out any function calls and emit assembly that directly
 * operates on the array's raw memory buffer(s). See svtkArrayDispatch for
 * utilities to convert an unknown svtkDataArray into a more derived type.
 * Testing/Cxx/ExampleDataArrayRangeDispatch.cxx demonstrates how ranges may
 * be used with the dispatcher system.
 *
 * # TupleRanges
 *
 * A TupleRange traverses a svtkDataArray tuple-by-tuple, providing iterators
 * and reference objects that refer to conceptual tuples. The tuple references
 * themselves may be iterated upon to access individual components.
 *
 * TupleRanges are created via the function svtk::DataArrayTupleRange. See
 * that function's documentation for more information about creating
 * TupleRanges.
 *
 * # ValueRanges
 *
 * A ValueRange will traverse a svtkDataArray in "value index" order, e.g. as
 * if walking a pointer into an AOS layout array:
 *
 * ```
 * Array:    {X, X, X}, {X, X, X}, {X, X, X}, ...
 * TupleIdx:  0  0  0    1  1  1    2  2  2
 * CompIdx:   0  1  2    0  1  2    0  1  2
 * ValueIdx:  0  1  2    3  4  5    6  7  8
 * ```
 *
 * ValueRanges are created via the function svtk::DataArrayValueRange. See that
 * function's documentation for more information about creating ValueRanges.
 */

SVTK_ITER_OPTIMIZE_START

namespace svtk
{

namespace detail
{

// Internal detail: This utility is not directly needed by users of
// DataArrayRange.
//
// These classes are used to detect when specializations exist for a given
// array type. They are necessary because given:
//
// template <typename ArrayType> class SomeTemplateClass;
// template <typename T> class SomeTemplateClass<svtkAOSDataArrayTemplate<T>>;
//
// SomeTemplateClass<svtkFloatArray> will pick the generic version, as ArrayType
// is a better match than svtkAOSDataArrayTemplate<T>. This class works around
// that by using Declare[Tuple|Value]RangeSpecialization functions that map an
// input ArrayTypePtr and tuple size to a specific version of the appropriate
// Range.
template <typename ArrayTypePtr, ComponentIdType TupleSize>
struct SelectTupleRange
{
private:
  // Allow this to work with svtkNew, svtkSmartPointer, etc.
  using ArrayType = typename detail::StripPointers<ArrayTypePtr>::type;

  static_assert(detail::IsValidTupleSize<TupleSize>::value, "Invalid tuple size.");
  static_assert(detail::IsVtkDataArray<ArrayType>::value, "Invalid array type.");

public:
  using type =
    typename std::decay<decltype(svtk::detail::DeclareTupleRangeSpecialization<ArrayType, TupleSize>(
      std::declval<ArrayType*>()))>::type;
};

template <typename ArrayTypePtr, ComponentIdType TupleSize>
struct SelectValueRange
{
private:
  // Allow this to work with svtkNew, svtkSmartPointer, etc.
  using ArrayType = typename detail::StripPointers<ArrayTypePtr>::type;

  static_assert(detail::IsValidTupleSize<TupleSize>::value, "Invalid tuple size.");
  static_assert(detail::IsVtkDataArray<ArrayType>::value, "Invalid array type.");

public:
  using type =
    typename std::remove_reference<decltype(svtk::detail::DeclareValueRangeSpecialization<ArrayType,
      TupleSize>(std::declval<ArrayType*>()))>::type;
};

} // end namespace detail

/**
 * @brief Generate an stl and for-range compatible range of tuple iterators
 * from a svtkDataArray.
 *
 * This function returns a TupleRange object that is compatible with C++11
 * for-range syntax. As an example usage, consider a function that takes some
 * instance of svtkDataArray (or a subclass) and prints the magnitude of each
 * tuple:
 *
 * ```
 * template <typename ArrayType>
 * void PrintMagnitudes(ArrayType *array)
 * {
 *   using T = svtk::GetAPIType<ArrayType>;
 *
 *   for (const auto tuple : svtk::DataArrayTupleRange(array))
 *   {
 *     double mag = 0.;
 *     for (const T comp : tuple)
 *     {
 *       mag += static_cast<double>(comp) * static_cast<double>(comp);
 *     }
 *     mag = std::sqrt(mag);
 *     std::cerr << mag < "\n";
 *   }
 * }
 * ```
 *
 * Note that `ArrayType` is generic in the above function. When
 * `svtk::DataArrayTupleRange` is given a `svtkDataArray` pointer, the generated
 * code produces iterators and reference proxies that rely on the `svtkDataArray`
 * API. However, when a more derived `ArrayType` is passed in (for example,
 * `svtkFloatArray`), specialized implementations are used that generate highly
 * optimized code.
 *
 * Performance can be further improved when the number of components in the
 * array is known. By passing a compile-time-constant integer as a template
 * parameter, e.g. `svtk::DataArrayTupleRange<3>(array)`, specializations are
 * enabled that allow the compiler to perform additional optimizations.
 *
 * `svtk::DataArrayTupleRange` takes an additional two arguments that can be used
 * to restrict the range of tuples to [start, end).
 *
 * There is a compiler definition / CMake option called
 * `SVTK_DEBUG_RANGE_ITERATORS` that enables checks for proper usage of the
 * range/iterator/reference classes. This slows things down significantly, but
 * is useful for diagnosing problems.
 *
 * In some situations, developers may want to build in Debug mode while still
 * maintaining decent performance for data-heavy computations. For these
 * usecases, an additional CMake option `SVTK_ALWAYS_OPTIMIZE_ARRAY_ITERATORS`
 * may be enabled to force optimization of code using these iterators. This
 * option will force inlining and enable -O3 (or equivalent) optimization level
 * for iterator code when compiling on platforms that support these features.
 * This option has no effect when `SVTK_DEBUG_RANGE_ITERATORS` is enabled.
 *
 * @warning Use caution when using `auto` to hold values or references obtained
 * from iterators, as they may not behave as expected. This is a deficiency in
 * C++ that affects all proxy iterators (such as those from `vector<bool>`)
 * that use a reference object instead of an actual C++ reference type. When in
 * doubt, use `std::iterator_traits` (along with decltype) or the typedefs
 * listed below to determine the proper value/reference type to use. The
 * examples below show how these may be used.
 *
 *
 * To mitigate this, the following types are defined on the range object:
 * - `Range::TupleIteratorType`: Iterator that visits tuples.
 * - `Range::ConstTupleIteratorType`: Const iterator that visits tuples.
 * - `Range::TupleReferenceType`: Mutable tuple proxy reference.
 * - `Range::ConstTupleReferenceType`: Const tuple proxy reference.
 * - `Range::ComponentIteratorType`: Iterator that visits components in a tuple.
 * - `Range::ConstComponentIteratorType`: Const iterator that visits tuple components.
 * - `Range::ComponentReferenceType`: Reference proxy to a single tuple component.
 * - `Range::ConstComponentReferenceType`: Const reference proxy to a single tuple component.
 * - `Range::ComponentType`: `ValueType` of components.
 *
 * These can be accessed via the range objects, e.g.:
 *
 * ```
 * auto range = svtk::DataArrayTupleRange(array);
 *
 * using TupleRef = typename decltype(range)::TupleReferenceType;
 * using ComponentRef = typename decltype(range)::ComponentReferenceType;
 *
 * for (TupleRef tuple : range)
 * {
 *   for (ComponentRef comp : tuple)
 *   {
 *     comp = comp - 1; // Array is modified.
 *   }
 * }
 *
 * using ConstTupleRef = typename decltype(range)::ConstTupleReferenceType;
 * using ComponentType = typename decltype(range)::ComponentType;
 *
 * for (ConstTupleRef tuple : range)
 * {
 *   for (ComponentType comp : tuple)
 *   {
 *     comp = comp - 1; // Array is not modified.
 *   }
 * }
 * ```
 */
template <ComponentIdType TupleSize = detail::DynamicTupleSize,
  typename ArrayTypePtr = svtkDataArray*>
SVTK_ITER_INLINE auto DataArrayTupleRange(const ArrayTypePtr& array, TupleIdType start = -1,
  TupleIdType end = -1) -> typename detail::SelectTupleRange<ArrayTypePtr, TupleSize>::type
{
  // Lookup specializations:
  using RangeType = typename detail::SelectTupleRange<ArrayTypePtr, TupleSize>::type;

  assert(array);

  return RangeType(array, start < 0 ? 0 : start, end < 0 ? array->GetNumberOfTuples() : end);
}

/**
 * @brief Generate an stl and for-range compatible range of flat AOS iterators
 * from a svtkDataArray.
 *
 * This function returns a ValueRange object that is compatible with C++11
 * for-range syntax. The array is traversed as if calling
 * svtkGenericDataArray::GetValue with consecutive, increasing indices. As an
 * example usage, consider a function that takes some instance of svtkDataArray
 * (or a subclass) and sums the values it contains:
 *
 * ```
 * template <typename ArrayType>
 * auto ComputeSum(ArrayType *array) -> svtk::GetAPIType<ArrayType>
 * {
 *   using T = svtk::GetAPIType<ArrayType>;
 *
 *   T sum = 0.;
 *   for (const T val : svtk::DataArrayValueRange(array))
 *   {
 *     sum += val;
 *   }
 *   return sum;
 * }
 * ```
 *
 * These ranges may also be used with STL algorithms:
 *
 * ```
 * template <typename ArrayType>
 * auto ComputeSum(ArrayType *array) -> svtk::GetAPIType<ArrayType>
 * {
 *   const auto range = svtk::DataArrayValueRange(array);
 *   return std::accumulate(range.begin(), range.end(), 0);
 * }
 * ```
 *
 * Note that `ArrayType` is generic in the above function. When
 * `svtk::DataArrayValueRange` is given a `svtkDataArray` pointer, the generated
 * code produces iterators and reference proxies that rely on the `svtkDataArray`
 * API. However, when a more derived `ArrayType` is passed in (for example,
 * `svtkFloatArray`), specialized implementations are used that generate highly
 * optimized code.
 *
 * Performance can be further improved when the number of components in the
 * array is known. By passing a compile-time-constant integer as a template
 * parameter, e.g. `svtk::DataArrayValueRange<3>(array)`, specializations are
 * enabled that allow the compiler to perform additional optimizations.
 *
 * `svtk::DataArrayValueRange` takes an additional two arguments that can be used
 * to restrict the range of values to [start, end).
 *
 * There is a compiler definition / CMake option called
 * `SVTK_DEBUG_RANGE_ITERATORS` that enables checks for proper usage of the
 * range/iterator/reference classes. This slows things down significantly, but
 * is useful for diagnosing problems.
 *
 * In some situations, developers may want to build in Debug mode while still
 * maintaining decent performance for data-heavy computations. For these
 * usecases, an additional CMake option `SVTK_ALWAYS_OPTIMIZE_ARRAY_ITERATORS`
 * may be enabled to force optimization of code using these iterators. This
 * option will force inlining and enable -O3 (or equivalent) optimization level
 * for iterator code when compiling on platforms that support these features.
 * This option has no effect when `SVTK_DEBUG_RANGE_ITERATORS` is enabled.
 *
 * @warning Use caution when using `auto` to hold values or references obtained
 * from iterators, as they may not behave as expected. This is a deficiency in
 * C++ that affects all proxy iterators (such as those from `vector<bool>`)
 * that use a reference object instead of an actual C++ reference type. When in
 * doubt, use `std::iterator_traits` (along with decltype) or the typedefs
 * listed below to determine the proper value/reference type to use. The
 * examples below show how these may be used.
 *
 * To mitigate this, the following types are defined on the range object:
 * - `Range::IteratorType`: Iterator that visits values in AOS order.
 * - `Range::ConstIteratorType`: Const iterator that visits values in AOS order.
 * - `Range::ReferenceType`: Mutable value proxy reference.
 * - `Range::ConstReferenceType`: Const value proxy reference.
 * - `Range::ValueType`: `ValueType` of array's API.
 *
 * These can be accessed via the range objects, e.g.:
 *
 * ```
 * auto range = svtk::DataArrayValueRange(array);
 *
 * using RefType = typename decltype(range)::ReferenceType;
 * for (RefType ref : range)
 * { // `ref` is a reference (or reference proxy) to the data held by the array.
 *   ref -= 1; // Array is modified.
 * }
 *
 * using ValueType = typename decltype(range)::ValueType;
 * for (ValueType value : range)
 * { // implicitly converts from a reference (or proxy) to a local lvalue `value`
 *   value -= 1; // Array is not modified.
 * }
 * ```
 */
template <ComponentIdType TupleSize = detail::DynamicTupleSize,
  typename ArrayTypePtr = svtkDataArray*>
SVTK_ITER_INLINE auto DataArrayValueRange(const ArrayTypePtr& array, ValueIdType start = -1,
  ValueIdType end = -1) -> typename detail::SelectValueRange<ArrayTypePtr, TupleSize>::type
{
  using RangeType = typename detail::SelectValueRange<ArrayTypePtr, TupleSize>::type;

  assert(array);

  return RangeType(array, start < 0 ? 0 : start, end < 0 ? array->GetNumberOfValues() : end);
}

} // end namespace svtk

SVTK_ITER_OPTIMIZE_END

#endif // svtkDataArrayRange_h

// SVTK-HeaderTest-Exclude: svtkDataArrayRange.h
