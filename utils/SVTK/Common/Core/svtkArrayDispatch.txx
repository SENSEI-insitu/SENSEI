/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayDispatch.txx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef svtkArrayDispatch_txx
#define svtkArrayDispatch_txx

#include "svtkArrayDispatch.h"

#include "svtkConfigure.h" // For warning macro settings.
#include "svtkSetGet.h"    // For warning macros.

#include <utility> // For std::forward

class svtkDataArray;

namespace svtkArrayDispatch
{
namespace impl
{

//------------------------------------------------------------------------------
// Implementation of the single-array dispatch mechanism.
template <typename ArrayList>
struct Dispatch;

// Terminal case:
template <>
struct Dispatch<svtkTypeList::NullType>
{
  template <typename... T>
  static bool Execute(T&&...)
  {
#ifdef SVTK_WARN_ON_DISPATCH_FAILURE
    svtkGenericWarningMacro("Array dispatch failed.");
#endif
    return false;
  }
};

// Recursive case:
template <typename ArrayHead, typename ArrayTail>
struct Dispatch<svtkTypeList::TypeList<ArrayHead, ArrayTail> >
{
  template <typename Worker, typename... Params>
  static bool Execute(svtkDataArray* inArray, Worker&& worker, Params&&... params)
  {
    if (ArrayHead* array = svtkArrayDownCast<ArrayHead>(inArray))
    {
      worker(array, std::forward<Params>(params)...);
      return true;
    }
    else
    {
      return Dispatch<ArrayTail>::Execute(
        inArray, std::forward<Worker>(worker), std::forward<Params>(params)...);
    }
  }
};

//------------------------------------------------------------------------------
// Description:
// Implementation of the 2 array dispatch mechanism.
template <typename ArrayList1, typename ArrayList2>
struct Dispatch2;

//----------------------------//
// First dispatch trampoline: //
//----------------------------//
template <typename Array1T, typename ArrayList2>
struct Dispatch2Trampoline;

// Dispatch2 Terminal case:
template <typename ArrayList2>
struct Dispatch2<svtkTypeList::NullType, ArrayList2>
{
  template <typename... T>
  static bool Execute(T&&...)
  {
#ifdef SVTK_WARN_ON_DISPATCH_FAILURE
    svtkGenericWarningMacro("Dual array dispatch failed.");
#endif
    return false;
  }
};

// Dispatch2 Recursive case:
template <typename Array1Head, typename Array1Tail, typename ArrayList2>
struct Dispatch2<svtkTypeList::TypeList<Array1Head, Array1Tail>, ArrayList2>
{
  typedef Dispatch2<Array1Tail, ArrayList2> NextDispatch;
  typedef Dispatch2Trampoline<Array1Head, ArrayList2> Trampoline;

  template <typename Worker, typename... Params>
  static bool Execute(
    svtkDataArray* array1, svtkDataArray* array2, Worker&& worker, Params&&... params)
  {
    if (Array1Head* array = svtkArrayDownCast<Array1Head>(array1))
    {
      return Trampoline::Execute(
        array, array2, std::forward<Worker>(worker), std::forward<Params>(params)...);
    }
    else
    {
      return NextDispatch::Execute(
        array1, array2, std::forward<Worker>(worker), std::forward<Params>(params)...);
    }
  }
};

// Dispatch2 Trampoline terminal case:
template <typename Array1T>
struct Dispatch2Trampoline<Array1T, svtkTypeList::NullType>
{
  template <typename... T>
  static bool Execute(T&&...)
  {
#ifdef SVTK_WARN_ON_DISPATCH_FAILURE
    svtkGenericWarningMacro("Dual array dispatch failed.");
#endif
    return false;
  }
};

// Dispatch2 Trampoline recursive case:
template <typename Array1T, typename Array2Head, typename Array2Tail>
struct Dispatch2Trampoline<Array1T, svtkTypeList::TypeList<Array2Head, Array2Tail> >
{
  typedef Dispatch2Trampoline<Array1T, Array2Tail> NextDispatch;

  template <typename Worker, typename... Params>
  static bool Execute(Array1T* array1, svtkDataArray* array2, Worker&& worker, Params&&... params)
  {
    if (Array2Head* array = svtkArrayDownCast<Array2Head>(array2))
    {
      worker(array1, array, std::forward<Params>(params)...);
      return true;
    }
    else
    {
      return NextDispatch::Execute(
        array1, array2, std::forward<Worker>(worker), std::forward<Params>(params)...);
    }
  }
};

//------------------------------------------------------------------------------
// Description:
// Implementation of the 2 array same-type dispatch mechanism.
template <typename ArrayList1, typename ArrayList2>
struct Dispatch2Same;

// Terminal case:
template <typename ArrayList2>
struct Dispatch2Same<svtkTypeList::NullType, ArrayList2>
{
  template <typename... T>
  static bool Execute(T&&...)
  {
#ifdef SVTK_WARN_ON_DISPATCH_FAILURE
    svtkGenericWarningMacro("Dual array dispatch failed.");
#endif
    return false;
  }
};

// Recursive case:
template <typename ArrayHead, typename ArrayTail, typename ArrayList2>
struct Dispatch2Same<svtkTypeList::TypeList<ArrayHead, ArrayTail>, ArrayList2>
{
  typedef Dispatch2Same<ArrayTail, ArrayList2> NextDispatch;
  typedef svtkTypeList::Create<typename ArrayHead::ValueType> ValueType;
  typedef typename FilterArraysByValueType<ArrayList2, ValueType>::Result ValueArrayList;
  typedef Dispatch2Trampoline<ArrayHead, ValueArrayList> Trampoline;

  template <typename Worker, typename... Params>
  static bool Execute(
    svtkDataArray* array1, svtkDataArray* array2, Worker&& worker, Params&&... params)
  {
    if (ArrayHead* array = svtkArrayDownCast<ArrayHead>(array1))
    {
      return Trampoline::Execute(
        array, array2, std::forward<Worker>(worker), std::forward<Params>(params)...);
    }
    else
    {
      return NextDispatch::Execute(
        array1, array2, std::forward<Worker>(worker), std::forward<Params>(params)...);
    }
  }
};

//------------------------------------------------------------------------------
// Description:
// Implementation of the 3 array dispatch mechanism.
template <typename ArrayList1, typename ArrayList2, typename ArrayList3>
struct Dispatch3;

//-----------------------------//
// First dispatch trampoline: //
//---------------------------//
template <typename Array1T, typename ArrayList2, typename ArrayList3>
struct Dispatch3Trampoline1;

//------------------------------//
// Second dispatch trampoline: //
//----------------------------//
template <typename Array1T, typename Array2T, typename ArrayList3>
struct Dispatch3Trampoline2;

// Dispatch3 Terminal case:
template <typename ArrayList2, typename ArrayList3>
struct Dispatch3<svtkTypeList::NullType, ArrayList2, ArrayList3>
{
  template <typename... T>
  static bool Execute(T&&...)
  {
#ifdef SVTK_WARN_ON_DISPATCH_FAILURE
    svtkGenericWarningMacro("Triple array dispatch failed.");
#endif
    return false;
  }
};

// Dispatch3 Recursive case:
template <typename ArrayHead, typename ArrayTail, typename ArrayList2, typename ArrayList3>
struct Dispatch3<svtkTypeList::TypeList<ArrayHead, ArrayTail>, ArrayList2, ArrayList3>
{
private:
  typedef Dispatch3Trampoline1<ArrayHead, ArrayList2, ArrayList3> Trampoline;
  typedef Dispatch3<ArrayTail, ArrayList2, ArrayList3> NextDispatch;

public:
  template <typename Worker, typename... Params>
  static bool Execute(svtkDataArray* array1, svtkDataArray* array2, svtkDataArray* array3,
    Worker&& worker, Params&&... params)
  {
    if (ArrayHead* array = svtkArrayDownCast<ArrayHead>(array1))
    {
      return Trampoline::Execute(
        array, array2, array3, std::forward<Worker>(worker), std::forward<Params>(params)...);
    }
    else
    {
      return NextDispatch::Execute(
        array1, array2, array3, std::forward<Worker>(worker), std::forward<Params>(params)...);
    }
  }
};

// Dispatch3 Trampoline1 terminal case:
template <typename Array1T, typename ArrayList3>
struct Dispatch3Trampoline1<Array1T, svtkTypeList::NullType, ArrayList3>
{
  template <typename... T>
  static bool Execute(T&&...)
  {
#ifdef SVTK_WARN_ON_DISPATCH_FAILURE
    svtkGenericWarningMacro("Triple array dispatch failed.");
#endif
    return false;
  }
};

// Dispatch3 Trampoline1 recursive case:
template <typename Array1T, typename ArrayHead, typename ArrayTail, typename ArrayList3>
struct Dispatch3Trampoline1<Array1T, svtkTypeList::TypeList<ArrayHead, ArrayTail>, ArrayList3>
{
private:
  typedef Dispatch3Trampoline2<Array1T, ArrayHead, ArrayList3> Trampoline;
  typedef Dispatch3Trampoline1<Array1T, ArrayTail, ArrayList3> NextDispatch;

public:
  template <typename Worker, typename... Params>
  static bool Execute(Array1T* array1, svtkDataArray* array2, svtkDataArray* array3, Worker&& worker,
    Params&&... params)
  {
    if (ArrayHead* array = svtkArrayDownCast<ArrayHead>(array2))
    {
      return Trampoline::Execute(
        array1, array, array3, std::forward<Worker>(worker), std::forward<Params>(params)...);
    }
    else
    {
      return NextDispatch::Execute(
        array1, array2, array3, std::forward<Worker>(worker), std::forward<Params>(params)...);
    }
  }
};

// Dispatch3 Trampoline2 terminal case:
template <typename Array1T, typename Array2T>
struct Dispatch3Trampoline2<Array1T, Array2T, svtkTypeList::NullType>
{
  template <typename... T>
  static bool Execute(T&&...)
  {
#ifdef SVTK_WARN_ON_DISPATCH_FAILURE
    svtkGenericWarningMacro("Triple array dispatch failed.");
#endif
    return false;
  }
};

// Dispatch3 Trampoline2 recursive case:
template <typename Array1T, typename Array2T, typename ArrayHead, typename ArrayTail>
struct Dispatch3Trampoline2<Array1T, Array2T, svtkTypeList::TypeList<ArrayHead, ArrayTail> >
{
private:
  typedef Dispatch3Trampoline2<Array1T, Array2T, ArrayTail> NextDispatch;

public:
  template <typename Worker, typename... Params>
  static bool Execute(
    Array1T* array1, Array2T* array2, svtkDataArray* array3, Worker&& worker, Params&&... params)
  {
    if (ArrayHead* array = svtkArrayDownCast<ArrayHead>(array3))
    {
      worker(array1, array2, array, std::forward<Params>(params)...);
      return true;
    }
    else
    {
      return NextDispatch::Execute(
        array1, array2, array3, std::forward<Worker>(worker), std::forward<Params>(params)...);
    }
  }
};

//------------------------------------------------------------------------------
// Description:
// Dispatch three arrays, enforcing that all three have the same ValueType.
// Initially, set both ArraysToTest and ArrayList to the same TypeList.
// ArraysToTest is iterated through, while ArrayList is preserved for later
// dispatches.
template <typename ArrayList1, typename ArrayList2, typename ArrayList3>
struct Dispatch3Same;

// Dispatch3Same terminal case:
template <typename ArrayList2, typename ArrayList3>
struct Dispatch3Same<svtkTypeList::NullType, ArrayList2, ArrayList3>
{
  template <typename... T>
  static bool Execute(T&&...)
  {
#ifdef SVTK_WARN_ON_DISPATCH_FAILURE
    svtkGenericWarningMacro("Triple array dispatch failed.");
#endif
    return false;
  }
};

// Dispatch3Same recursive case:
template <typename ArrayHead, typename ArrayTail, typename ArrayList2, typename ArrayList3>
struct Dispatch3Same<svtkTypeList::TypeList<ArrayHead, ArrayTail>, ArrayList2, ArrayList3>
{
private:
  typedef svtkTypeList::Create<typename ArrayHead::ValueType> ValueType;
  typedef typename FilterArraysByValueType<ArrayList2, ValueType>::Result ValueArrays2;
  typedef typename FilterArraysByValueType<ArrayList3, ValueType>::Result ValueArrays3;
  typedef Dispatch3Trampoline1<ArrayHead, ValueArrays2, ValueArrays3> Trampoline;
  typedef Dispatch3Same<ArrayTail, ArrayList2, ArrayList3> NextDispatch;

public:
  template <typename Worker, typename... Params>
  static bool Execute(svtkDataArray* array1, svtkDataArray* array2, svtkDataArray* array3,
    Worker&& worker, Params&&... params)
  {
    if (ArrayHead* array = svtkArrayDownCast<ArrayHead>(array1))
    {
      return Trampoline::Execute(
        array, array2, array3, std::forward<Worker>(worker), std::forward<Params>(params)...);
    }
    else
    {
      return NextDispatch::Execute(
        array1, array2, array3, std::forward<Worker>(worker), std::forward<Params>(params)...);
    }
  }
};

} // end namespace impl

//------------------------------------------------------------------------------
// FilterArraysByValueType implementation:
//------------------------------------------------------------------------------

// Terminal case:
template <typename ValueList>
struct FilterArraysByValueType<svtkTypeList::NullType, ValueList>
{
  typedef svtkTypeList::NullType Result;
};

// Recursive case:
template <typename ArrayHead, typename ArrayTail, typename ValueList>
struct FilterArraysByValueType<svtkTypeList::TypeList<ArrayHead, ArrayTail>, ValueList>
{
private:
  typedef typename ArrayHead::ValueType ValueType;
  enum
  {
    ValueIsAllowed = svtkTypeList::IndexOf<ValueList, ValueType>::Result >= 0
  };
  typedef typename FilterArraysByValueType<ArrayTail, ValueList>::Result NewTail;

public:
  typedef typename svtkTypeList::Select<ValueIsAllowed, svtkTypeList::TypeList<ArrayHead, NewTail>,
    NewTail>::Result Result;
};

//------------------------------------------------------------------------------
// DispatchByArray implementation:
//------------------------------------------------------------------------------
// Preprocess and pass off to impl::Dispatch:
template <typename ArrayHead, typename ArrayTail>
struct DispatchByArray<svtkTypeList::TypeList<ArrayHead, ArrayTail> >
{
private:
  typedef svtkTypeList::TypeList<ArrayHead, ArrayTail> ArrayList;
  typedef typename svtkTypeList::Unique<ArrayList>::Result UniqueArrays;
  typedef typename svtkTypeList::DerivedToFront<UniqueArrays>::Result SortedUniqueArrays;
  typedef impl::Dispatch<SortedUniqueArrays> ArrayDispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(svtkDataArray* inArray, Worker&& worker, Params&&... params)
  {
    return ArrayDispatcher::Execute(
      inArray, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

//------------------------------------------------------------------------------
// Dispatch implementation:
// (defined after DispatchByArray to prevent 'incomplete type' errors)
//------------------------------------------------------------------------------
struct Dispatch
{
private:
  typedef DispatchByArray<Arrays> Dispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(svtkDataArray* array, Worker&& worker, Params&&... params)
  {
    return Dispatcher::Execute(
      array, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

//------------------------------------------------------------------------------
// DispatchByValueType implementation:
//------------------------------------------------------------------------------
// Preprocess and pass off to impl::Dispatch
template <typename ValueTypeHead, typename ValueTypeTail>
struct DispatchByValueType<svtkTypeList::TypeList<ValueTypeHead, ValueTypeTail> >
{
private:
  typedef svtkTypeList::TypeList<ValueTypeHead, ValueTypeTail> ValueTypeList;
  typedef typename FilterArraysByValueType<Arrays, ValueTypeList>::Result ArrayList;
  typedef typename svtkTypeList::Unique<ArrayList>::Result UniqueArrays;
  typedef typename svtkTypeList::DerivedToFront<UniqueArrays>::Result SortedUniqueArrays;
  typedef impl::Dispatch<SortedUniqueArrays> ArrayDispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(svtkDataArray* inArray, Worker&& worker, Params&&... params)
  {
    return ArrayDispatcher::Execute(
      inArray, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

//------------------------------------------------------------------------------
// Dispatch2ByArray implementation:
//------------------------------------------------------------------------------
// Preprocess and pass off to impl::Dispatch2:
template <typename ArrayList1, typename ArrayList2>
struct Dispatch2ByArray
{
private:
  typedef typename svtkTypeList::Unique<ArrayList1>::Result UniqueArray1;
  typedef typename svtkTypeList::Unique<ArrayList2>::Result UniqueArray2;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray1>::Result SortedUniqueArray1;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray2>::Result SortedUniqueArray2;
  typedef impl::Dispatch2<SortedUniqueArray1, SortedUniqueArray2> ArrayDispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(
    svtkDataArray* array1, svtkDataArray* array2, Worker&& worker, Params&&... params)
  {
    return ArrayDispatcher::Execute(
      array1, array2, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

//------------------------------------------------------------------------------
// Dispatch2 implementation:
//------------------------------------------------------------------------------
struct Dispatch2
{
private:
  typedef Dispatch2ByArray<svtkArrayDispatch::Arrays, svtkArrayDispatch::Arrays> Dispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(
    svtkDataArray* array1, svtkDataArray* array2, Worker&& worker, Params&&... params)
  {
    return Dispatcher::Execute(
      array1, array2, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

//------------------------------------------------------------------------------
// Dispatch2ByValueType implementation:
//------------------------------------------------------------------------------
// Preprocess and pass off to impl::Dispatch2
template <typename ValueTypeList1, typename ValueTypeList2>
struct Dispatch2ByValueType
{
private:
  typedef typename FilterArraysByValueType<Arrays, ValueTypeList1>::Result ArrayList1;
  typedef typename FilterArraysByValueType<Arrays, ValueTypeList2>::Result ArrayList2;
  typedef typename svtkTypeList::Unique<ArrayList1>::Result UniqueArray1;
  typedef typename svtkTypeList::Unique<ArrayList2>::Result UniqueArray2;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray1>::Result SortedUniqueArray1;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray2>::Result SortedUniqueArray2;
  typedef impl::Dispatch2<SortedUniqueArray1, SortedUniqueArray2> ArrayDispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(
    svtkDataArray* array1, svtkDataArray* array2, Worker&& worker, Params&&... params)
  {
    return ArrayDispatcher::Execute(
      array1, array2, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

//------------------------------------------------------------------------------
// Dispatch2BySameValueType implementation:
//------------------------------------------------------------------------------
// Preprocess and pass off to impl::Dispatch2Same
template <typename ValueTypeList>
struct Dispatch2BySameValueType
{
private:
  typedef typename FilterArraysByValueType<Arrays, ValueTypeList>::Result ArrayList;
  typedef typename svtkTypeList::Unique<ArrayList>::Result UniqueArray;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray>::Result SortedUniqueArray;
  typedef impl::Dispatch2Same<SortedUniqueArray, SortedUniqueArray> Dispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(
    svtkDataArray* array1, svtkDataArray* array2, Worker&& worker, Params&&... params)
  {
    return Dispatcher::Execute(
      array1, array2, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

//------------------------------------------------------------------------------
// Dispatch2ByArrayWithSameValueType implementation:
//------------------------------------------------------------------------------
// Preprocess and pass off to impl::Dispatch2Same
template <typename ArrayList1, typename ArrayList2>
struct Dispatch2ByArrayWithSameValueType
{
private:
  typedef typename svtkTypeList::Unique<ArrayList1>::Result UniqueArray1;
  typedef typename svtkTypeList::Unique<ArrayList2>::Result UniqueArray2;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray1>::Result SortedUniqueArray1;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray2>::Result SortedUniqueArray2;
  typedef impl::Dispatch2Same<SortedUniqueArray1, SortedUniqueArray2> Dispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(
    svtkDataArray* array1, svtkDataArray* array2, Worker&& worker, Params&&... params)
  {
    return Dispatcher::Execute(
      array1, array2, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

//------------------------------------------------------------------------------
// Dispatch2SameValueType implementation:
//------------------------------------------------------------------------------
struct Dispatch2SameValueType
{
private:
  typedef Dispatch2ByArrayWithSameValueType<svtkArrayDispatch::Arrays, svtkArrayDispatch::Arrays>
    Dispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(
    svtkDataArray* array1, svtkDataArray* array2, Worker&& worker, Params&&... params)
  {
    return Dispatcher::Execute(
      array1, array2, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

//------------------------------------------------------------------------------
// Dispatch3ByArray implementation:
//------------------------------------------------------------------------------
// Preprocess and pass off to impl::Dispatch3:
template <typename ArrayList1, typename ArrayList2, typename ArrayList3>
struct Dispatch3ByArray
{
private:
  typedef typename svtkTypeList::Unique<ArrayList1>::Result UniqueArray1;
  typedef typename svtkTypeList::Unique<ArrayList2>::Result UniqueArray2;
  typedef typename svtkTypeList::Unique<ArrayList3>::Result UniqueArray3;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray1>::Result SortedUniqueArray1;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray2>::Result SortedUniqueArray2;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray3>::Result SortedUniqueArray3;
  typedef impl::Dispatch3<SortedUniqueArray1, SortedUniqueArray2, SortedUniqueArray3>
    ArrayDispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(svtkDataArray* array1, svtkDataArray* array2, svtkDataArray* array3,
    Worker&& worker, Params&&... params)
  {
    return ArrayDispatcher::Execute(
      array1, array2, array3, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

//------------------------------------------------------------------------------
// Dispatch3 implementation:
//------------------------------------------------------------------------------
struct Dispatch3
{
private:
  typedef Dispatch3ByArray<svtkArrayDispatch::Arrays, svtkArrayDispatch::Arrays,
    svtkArrayDispatch::Arrays>
    Dispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(svtkDataArray* array1, svtkDataArray* array2, svtkDataArray* array3,
    Worker&& worker, Params&&... params)
  {
    return Dispatcher::Execute(
      array1, array2, array3, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

//------------------------------------------------------------------------------
// Dispatch3ByValueType implementation:
//------------------------------------------------------------------------------
// Preprocess and pass off to impl::Dispatch3
template <typename ValueTypeList1, typename ValueTypeList2, typename ValueTypeList3>
struct Dispatch3ByValueType
{
private:
  typedef typename FilterArraysByValueType<Arrays, ValueTypeList1>::Result ArrayList1;
  typedef typename FilterArraysByValueType<Arrays, ValueTypeList2>::Result ArrayList2;
  typedef typename FilterArraysByValueType<Arrays, ValueTypeList3>::Result ArrayList3;
  typedef typename svtkTypeList::Unique<ArrayList1>::Result UniqueArray1;
  typedef typename svtkTypeList::Unique<ArrayList2>::Result UniqueArray2;
  typedef typename svtkTypeList::Unique<ArrayList3>::Result UniqueArray3;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray1>::Result SortedUniqueArray1;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray2>::Result SortedUniqueArray2;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray3>::Result SortedUniqueArray3;
  typedef impl::Dispatch3<SortedUniqueArray1, SortedUniqueArray2, SortedUniqueArray3>
    ArrayDispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(svtkDataArray* array1, svtkDataArray* array2, svtkDataArray* array3,
    Worker&& worker, Params&&... params)
  {
    return ArrayDispatcher::Execute(
      array1, array2, array3, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

//------------------------------------------------------------------------------
// Dispatch3BySameValueType implementation:
//------------------------------------------------------------------------------
// Preprocess and pass off to impl::Dispatch3Same
template <typename ValueTypeList>
struct Dispatch3BySameValueType
{
private:
  typedef typename FilterArraysByValueType<Arrays, ValueTypeList>::Result ArrayList;
  typedef typename svtkTypeList::Unique<ArrayList>::Result UniqueArray;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray>::Result SortedUniqueArray;
  typedef impl::Dispatch3Same<SortedUniqueArray, SortedUniqueArray, SortedUniqueArray> Dispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(svtkDataArray* array1, svtkDataArray* array2, svtkDataArray* array3,
    Worker&& worker, Params&&... params)
  {
    return Dispatcher::Execute(
      array1, array2, array3, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

//------------------------------------------------------------------------------
// Dispatch3BySameValueType implementation:
//------------------------------------------------------------------------------
// Preprocess and pass off to impl::Dispatch3Same
template <typename ArrayList1, typename ArrayList2, typename ArrayList3>
struct Dispatch3ByArrayWithSameValueType
{
private:
  typedef typename svtkTypeList::Unique<ArrayList1>::Result UniqueArray1;
  typedef typename svtkTypeList::Unique<ArrayList2>::Result UniqueArray2;
  typedef typename svtkTypeList::Unique<ArrayList3>::Result UniqueArray3;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray1>::Result SortedUniqueArray1;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray2>::Result SortedUniqueArray2;
  typedef typename svtkTypeList::DerivedToFront<UniqueArray3>::Result SortedUniqueArray3;
  typedef impl::Dispatch3Same<SortedUniqueArray1, SortedUniqueArray2, SortedUniqueArray3>
    Dispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(svtkDataArray* array1, svtkDataArray* array2, svtkDataArray* array3,
    Worker&& worker, Params&&... params)
  {
    return Dispatcher::Execute(
      array1, array2, array3, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

//------------------------------------------------------------------------------
// Dispatch3SameValueType implementation:
//------------------------------------------------------------------------------
struct Dispatch3SameValueType
{
private:
  typedef Dispatch3ByArrayWithSameValueType<svtkArrayDispatch::Arrays, svtkArrayDispatch::Arrays,
    svtkArrayDispatch::Arrays>
    Dispatcher;

public:
  template <typename Worker, typename... Params>
  static bool Execute(svtkDataArray* array1, svtkDataArray* array2, svtkDataArray* array3,
    Worker&& worker, Params&&... params)
  {
    return Dispatcher::Execute(
      array1, array2, array3, std::forward<Worker>(worker), std::forward<Params>(params)...);
  }
};

} // end namespace svtkArrayDispatch

#endif // svtkArrayDispatch_txx
