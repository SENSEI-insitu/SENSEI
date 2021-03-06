/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDoubleDispatcher.h

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
 * @class   svtkDoubleDispatcher
 * @brief   Dispatch to functor based on two pointer types.
 *
 * svtkDoubleDispatcher is a class that allows calling a functor based
 * on the derived types of two pointers. This form of dynamic dispatching
 * allows the conversion of runtime polymorphism to a compile time polymorphism.
 * For example it can be used as a replacement for the svtkTemplateMacro when
 * you want to know multiple parameter types, or need to call a specialized implementation
 * for a subset
 *
 * Note: By default the return type is void.
 *
 * The functors that are passed around can contain state, and are allowed
 * to be const or non const. If you are using a functor that does have state,
 * make sure your copy constructor is correct.
 *
 * \code
 * struct functor{
 *   template<typename T,typename U>
 *   void operator()(T& t,U& u) const
 *   {
 *
 *   }
 * };
 *
 * Here is an example of using the double dispatcher.
 *  \code
 *  svtkDoubleDispatcher<svtkObject,svtkObject,svtkPoints*> dispatcher;
 *  dispatcher.Add<svtkPoints,svtkDoubleArray>(makePointsWrapperFunctor());
 *  dispatcher.Add<svtkPoints,svtkPoints>(straightCopyFunctor());
 *  dispatcher.Go(ptr1,ptr2); //this will return a svtkPoints pointer
 *  \endcode
 *
 *
 * @sa
 * svtkDispatcher
 */

#ifndef svtkDoubleDispatcher_h
#define svtkDoubleDispatcher_h

#include "svtkConfigure.h"

#ifndef SVTK_LEGACY_REMOVE

#include "svtkDispatcher_Private.h" //needed for Functor,CastingPolicy,TypeInfo
#include <map>                     //Required for the storage of template params to runtime params

template <class BaseLhs, class BaseRhs = BaseLhs, typename ReturnType = void,
  template <class, class> class CastingPolicy = svtkDispatcherCommon::svtkCaster>
class svtkDoubleDispatcher
{
public:
  /**
   * Add in a functor that is mapped to the combination of the
   * two template parameters passed in. When instances of the two parameters
   * are passed in on the Go method we will call the functor and pass along
   * the given parameters.
   * Note: This copies the functor so pass stateful functors by pointer.

   * \code
   * svtkDoubleDispatcher<svtkDataModel,svtkCell> dispatcher;
   * dispatcher.Add<svtkImageData,svtkVoxel>(exampleFunctor());
   * dispatcher.Add<svtkImageData,svtkVoxel>(&exampleFunctorWithState);
   * \endcode
   */
  template <class SomeLhs, class SomeRhs, class Functor>
  void Add(Functor fun)
  {
    SVTK_LEGACY_BODY(svtkDoubleDispatcher, "SVTK 9.0");
    this->AddInternal<SomeLhs, SomeRhs>(fun, 1);
  }

  /**
   * Remove a functor that is bound to the given parameter types. Will
   * return true if we did remove a functor.
   */
  template <class SomeLhs, class SomeRhs>
  bool Remove()
  {
    return DoRemove(typeid(SomeLhs), typeid(SomeRhs));
  }

  /**
   * Given two pointers of objects that derive from the BaseLhs and BaseRhs
   * we find the matching functor that was added, and call it passing along
   * the given parameters. It should be noted that the functor will be called
   * with the parameters being the derived type that Functor was registered with.

   * Note: This will only find exact matches. So if you add functor to find
   * svtkDataArray,svtkDataArray, it will not be called if passed with
   * svtkDoubleArray,svtkDoubleArray.

   * \code

   * svtkDoubleDispatcher<svtkDataArray,svtkDataArray> dispatcher;
   * dispatcher.Add(svtkFloatArray,svtkFloatArray>(floatFunctor())
   * dispatcher.Add(svtkFloatArray,svtkDoubleArray>(mixedTypeFunctor())
   * dispatcher.Go( dataArray1, dataArray2);
   * \endcode
   */
  ReturnType Go(BaseLhs* lhs, BaseRhs* rhs);

protected:
  typedef svtkDispatcherCommon::TypeInfo TypeInfo;
  typedef svtkDoubleDispatcherPrivate::Functor<ReturnType, BaseLhs, BaseRhs> MappedType;

  void DoAddFunctor(TypeInfo lhs, TypeInfo rhs, MappedType fun);
  bool DoRemove(TypeInfo lhs, TypeInfo rhs);

  typedef std::pair<TypeInfo, TypeInfo> KeyType;
  typedef std::map<KeyType, MappedType> MapType;
  MapType FunctorMap;

private:
  template <class SomeLhs, class SomeRhs, class Functor>
  void AddInternal(const Functor& fun, long);
  template <class SomeLhs, class SomeRhs, class Functor>
  void AddInternal(Functor* fun, int);
};

// We are making all these method non-inline to reduce compile time overhead
//----------------------------------------------------------------------------
template <class BaseLhs, class BaseRhs, typename ReturnType,
  template <class, class> class CastingPolicy>
template <class SomeLhs, class SomeRhs, class Functor>
void svtkDoubleDispatcher<BaseLhs, BaseRhs, ReturnType, CastingPolicy>::AddInternal(
  const Functor& fun, long)
{
  typedef svtkDoubleDispatcherPrivate::FunctorDoubleDispatcherHelper<BaseLhs, BaseRhs, SomeLhs,
    SomeRhs, ReturnType, CastingPolicy<SomeLhs, BaseLhs>, CastingPolicy<SomeRhs, BaseRhs>, Functor>
    Adapter;
  Adapter ada(fun);
  MappedType mt(ada);
  DoAddFunctor(typeid(SomeLhs), typeid(SomeRhs), mt);
}

//----------------------------------------------------------------------------
template <class BaseLhs, class BaseRhs, typename ReturnType,
  template <class, class> class CastingPolicy>
template <class SomeLhs, class SomeRhs, class Functor>
void svtkDoubleDispatcher<BaseLhs, BaseRhs, ReturnType, CastingPolicy>::AddInternal(
  Functor* fun, int)
{
  typedef svtkDoubleDispatcherPrivate::FunctorRefDispatcherHelper<BaseLhs, BaseRhs, SomeLhs, SomeRhs,
    ReturnType, CastingPolicy<SomeLhs, BaseLhs>, CastingPolicy<SomeRhs, BaseRhs>, Functor>
    Adapter;
  Adapter ada(*fun);
  MappedType mt(ada);
  DoAddFunctor(typeid(SomeLhs), typeid(SomeRhs), mt);
}

//----------------------------------------------------------------------------
template <class BaseLhs, class BaseRhs, typename ReturnType,
  template <class, class> class CastingPolicy>
void svtkDoubleDispatcher<BaseLhs, BaseRhs, ReturnType, CastingPolicy>::DoAddFunctor(
  TypeInfo lhs, TypeInfo rhs, MappedType fun)
{
  FunctorMap[KeyType(lhs, rhs)] = fun;
}

//----------------------------------------------------------------------------
template <class BaseLhs, class BaseRhs, typename ReturnType,
  template <class, class> class CastingPolicy>
bool svtkDoubleDispatcher<BaseLhs, BaseRhs, ReturnType, CastingPolicy>::DoRemove(
  TypeInfo lhs, TypeInfo rhs)
{
  return FunctorMap.erase(KeyType(lhs, rhs)) == 1;
}

//----------------------------------------------------------------------------
template <class BaseLhs, class BaseRhs, typename ReturnType,
  template <class, class> class CastingPolicy>
ReturnType svtkDoubleDispatcher<BaseLhs, BaseRhs, ReturnType, CastingPolicy>::Go(
  BaseLhs* lhs, BaseRhs* rhs)
{
  typename MapType::key_type k(typeid(*lhs), typeid(*rhs));
  typename MapType::iterator i = FunctorMap.find(k);
  if (i == FunctorMap.end())
  {
    // we don't want to throw exceptions so we have two options.
    // we can return the default, or make a lightweight struct for return value
    return ReturnType();
  }
  return (i->second)(*lhs, *rhs);
}

#endif // legacy
#endif // svtkDoubleDispatcher_h
// SVTK-HeaderTest-Exclude: svtkDoubleDispatcher.h
