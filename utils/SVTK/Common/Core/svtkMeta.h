/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMeta.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef svtkMeta_h
#define svtkMeta_h

#include <type_traits>
#include <utility>

/**
 * @file svtkMeta
 * This file contains a variety of metaprogramming constructs for working
 * with svtk types.
 */

// Forward decs for StripPointers:
template <typename ArrayType>
class svtkNew;
template <typename ArrayType>
class svtkSmartPointer;
template <typename ArrayType>
class svtkWeakPointer;

namespace svtk
{
namespace detail
{

//------------------------------------------------------------------------------
// Strip svtkNew, svtkSmartPointer, etc from a type.
template <typename T>
struct StripPointers
{
  using type = T;
};

template <typename T>
struct StripPointers<T*>
{
  using type = T;
};

template <typename ArrayType>
struct StripPointers<svtkNew<ArrayType> >
{
  using type = ArrayType;
};

template <typename ArrayType>
struct StripPointers<svtkSmartPointer<ArrayType> >
{
  using type = ArrayType;
};

template <typename ArrayType>
struct StripPointers<svtkWeakPointer<ArrayType> >
{
  using type = ArrayType;
};

//------------------------------------------------------------------------------
// Test if a type is defined (true) or just forward declared (false).
template <typename T>
struct IsComplete
{
private:
  // Can't take the sizeof an incomplete class.
  template <typename U, std::size_t = sizeof(U)>
  static std::true_type impl(U*);
  static std::false_type impl(...);
  using bool_constant = decltype(impl(std::declval<T*>()));

public:
  static constexpr bool value = bool_constant::value;
};

}
} // end namespace svtk::detail

#endif // svtkMeta_h

// SVTK-HeaderTest-Exclude: svtkMeta.h
