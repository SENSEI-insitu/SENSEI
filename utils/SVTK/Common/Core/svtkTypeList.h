/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTypeList.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

////////////////////////////////////////////////////////////////////////////////
// The Loki Library
// Copyright (c) 2001 by Andrei Alexandrescu
// This code accompanies the book:
// Alexandrescu, Andrei. "Modern C++ Design: Generic Programming and Design
//     Patterns Applied". Copyright (c) 2001. Addison-Wesley.
// Permission to use, copy, modify, distribute and sell this software for any
//     purpose is hereby granted without fee, provided that the above copyright
//     notice appear in all copies and that both that copyright notice and this
//     permission notice appear in supporting documentation.
// The author or Addison-Wesley Longman make no representations about the
//     suitability of this software for any purpose. It is provided "as is"
//     without express or implied warranty.
////////////////////////////////////////////////////////////////////////////////

/**
 * @class   svtkTypeList
 * @brief   TypeList implementation and utilities.
 *
 *
 * svtkTypeList provides a way to collect a list of types using C++ templates.
 * In SVTK, this is used heavily by the svtkArrayDispatch system to instantiate
 * templated code for specific array implementations. The book "Modern C++
 * Design: Generic Programming and Design Patterns Applied" by Andrei
 * Alexandrescu provides additional details and applications for typeLists. This
 * implementation is heavily influenced by the example code in the book.
 *
 * Note that creating a typelist in C++ is simplified greatly by using the
 * svtkTypeList::Create<T1, T2, ...> functions.
 *
 * @sa
 * svtkArrayDispatch svtkTypeListMacros
 */

#ifndef svtkTypeList_h
#define svtkTypeList_h

#ifndef __SVTK_WRAP__

#include "svtkTypeListMacros.h"

namespace svtkTypeList
{

//------------------------------------------------------------------------------
/**
 * Used to terminate a TypeList.
 */
struct NullType
{
};

//------------------------------------------------------------------------------
//@{
/**
 * Generic implementation of TypeList.
 */
template <typename T, typename U>
struct TypeList
{
  typedef T Head;
  typedef U Tail;
};
//@}

//------------------------------------------------------------------------------
/**
 * Sets Result to T if Exp is true, or F if Exp is false.
 */
template <bool Exp, typename T, typename F>
struct Select;

//------------------------------------------------------------------------------
/**
 * Sets member Result to true if a conversion exists to convert type From to
 * type To. Member SameType will be true if the types are identical.
 */
template <typename From, typename To>
struct CanConvert;

//------------------------------------------------------------------------------
/**
 * Sets the enum value Result to the index of type T in the TypeList TList.
 * Result will equal -1 if the type is not found.
 */
template <typename TList, typename T>
struct IndexOf;

//------------------------------------------------------------------------------
/**
 * Erase the first element of type T from TypeList TList, storing the new list
 * in Result.
 */
template <typename TList, typename T>
struct Erase;

//------------------------------------------------------------------------------
/**
 * Erase all type T from TypeList TList, storing the new list in Result.
 */
template <typename TList, typename T>
struct EraseAll;

//------------------------------------------------------------------------------
/**
 * Remove all duplicate types from TypeList TList, storing the new list in
 * Result.
 */
template <typename TList>
struct Unique;

//------------------------------------------------------------------------------
/**
 * Replace the first instance of Bad with Good in the TypeList TList, storing
 * the new list in Result.
 */
template <typename TList, typename Bad, typename Good>
struct Replace;

//------------------------------------------------------------------------------
/**
 * Replace all instances of Bad with Good in the TypeList TList, storing the
 * new list in Result.
 */
template <typename TList, typename Bad, typename Good>
struct ReplaceAll;

//------------------------------------------------------------------------------
/**
 * Given a type T and a TypeList TList, store the most derived type of T in
 * TList as Result. If no subclasses of T exist in TList, T will be set as
 * Result, even if T itself is not in TList.
 */
template <typename TList, typename T>
struct MostDerived;

//------------------------------------------------------------------------------
/**
 * Sort the TypeList from most-derived to least-derived type, storing the
 * sorted TypeList in Result. Note that the input TypeList cannot have duplicate
 * types (see Unique).
 */
template <typename TList>
struct DerivedToFront;

//------------------------------------------------------------------------------
/**
 * Appends type T to TypeList TList and stores the result in Result.
 */
template <typename TList, typename T>
struct Append;

} // end namespace svtkTypeList

#include "svtkTypeList.txx"

namespace svtkTypeList
{

template <typename... Ts>
using Create = typename svtkTypeList::detail::CreateImpl<Ts...>::type;

} // end namespace svtkTypeList

#endif // __SVTK_WRAP__

#endif // svtkTypeList_h
// SVTK-HeaderTest-Exclude: svtkTypeList.h
