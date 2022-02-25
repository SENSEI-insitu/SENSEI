/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArrayDispatcher.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkDataArrayDispatcher
 * @brief   Dispatch to functor svtkDataArrayType
 *
 * @warning The svtkArrayDispatch utilities should be used instead.
 *
 * svtkDataArrayDispatcher is a class that allows calling a functor based
 * on the data type of the svtkDataArray subclass. This is a wrapper
 * around the svtkTemplateMacro (SVTK_TT) to allow easier implementation and
 * readability, while at the same time the ability to use statefull functors.
 *
 * Note: By default the return type is void.
 * Note: The functor parameter must be of type svtkDataArrayDispatcherPointer
 *
 * The functors that are passed around can contain state, and are allowed
 * to be const or non const. If you are using a functor that does have state,
 * make sure your copy constructor is correct.
 *
 * \code
 * struct sizeOfFunctor{
 *   template<typename T>
 *   int operator()(const svtkDataArrayDispatcherPointer<T>& t) const
 *   {
 *   return t.NumberOfComponents * t.NumberOfTuples;
 *   }
 * };
 *
 * Here is an example of using the dispatcher.
 *  \code
 *  svtkDataArrayDispatcher<sizeOfFunctor,int> dispatcher;
 *  int arrayLength = dispatcher.Go(svtkDataArrayPtr);
 *  \endcode
 *
 *
 * @sa
 * svtkArrayDispatch
 */

#ifndef svtkDataArrayDispatcher_h
#define svtkDataArrayDispatcher_h

#include "svtkConfigure.h"

#ifndef SVTK_LEGACY_REMOVE

#include "svtkDataArray.h" //required for constructor of the svtkDataArrayFunctor
#include "svtkType.h"      //Required for svtkIdType
#include <map>            //Required for the storage of template params to runtime params

////////////////////////////////////////////////////////////////////////////////
// Object that is passed to all functor that are used with this class
// This allows the user the ability to find info about the size
////////////////////////////////////////////////////////////////////////////////
template <typename T>
struct svtkDataArrayDispatcherPointer
{
  typedef T ValueType;

  svtkIdType NumberOfTuples;
  svtkIdType NumberOfComponents;
  ValueType* RawPointer;

  explicit svtkDataArrayDispatcherPointer(svtkDataArray* array)
    : NumberOfTuples(array->GetNumberOfTuples())
    , NumberOfComponents(array->GetNumberOfComponents())
    , RawPointer(static_cast<ValueType*>(array->GetVoidPointer(0)))
  {
  }
};

////////////////////////////////////////////////////////////////////////////////
// class template FunctorDispatcher
////////////////////////////////////////////////////////////////////////////////
template <class DefaultFunctorType, typename ReturnType = void>
class svtkDataArrayDispatcher
{
public:
  /**
   * Specify the functor that is to be used when dispatching. This allows
   * you to specify a statefull functor.

   * \code

   * struct storeLengthFunctor
   * {
   * int length;
   * storeLengthFunctor() : length(0) {}

   * template<typename T>
   * void operator()(svtkDataArrayDispatcherPointer<T> array)
   * {
   * length += array.NumberOfComponents * array.NumberOfTuples;
   * }
   * };

   * storeLengthFunctor storedLength;
   * svtkDataArrayDispatcher<storeLengthFunctor> dispatcher(storedLength);
   * dispatcher.Go(exampleDataArray);

   * \endcode
   */
  svtkDataArrayDispatcher(DefaultFunctorType& f);

  /**
   * Default constructor which will create an instance of the DefaultFunctorType
   * and use that single instance for all calls.
   */
  svtkDataArrayDispatcher();

  virtual ~svtkDataArrayDispatcher();

  /**
   * Execute the default functor with the passed in svtkDataArray;
   */
  ReturnType Go(svtkDataArray* lhs);

protected:
  DefaultFunctorType* DefaultFunctor;
  bool OwnsFunctor;
};

// We are making all these method non-inline to reduce compile time overhead

//----------------------------------------------------------------------------
template <class DefaultFunctorType, typename ReturnType>
svtkDataArrayDispatcher<DefaultFunctorType, ReturnType>::svtkDataArrayDispatcher(
  DefaultFunctorType& fun)
  : DefaultFunctor(&fun)
  , OwnsFunctor(false)
{
  SVTK_LEGACY_BODY(svtkDataArrayDispatcher, "SVTK 9.0");
}

//----------------------------------------------------------------------------
template <class DefaultFunctorType, typename ReturnType>
svtkDataArrayDispatcher<DefaultFunctorType, ReturnType>::svtkDataArrayDispatcher()
  : DefaultFunctor(new DefaultFunctorType())
  , OwnsFunctor(true)
{
  SVTK_LEGACY_BODY(svtkDataArrayDispatcher, "SVTK 9.0");
}

//----------------------------------------------------------------------------
template <class DefaultFunctorType, typename ReturnType>
svtkDataArrayDispatcher<DefaultFunctorType, ReturnType>::~svtkDataArrayDispatcher()
{
  if (OwnsFunctor)
  {
    delete this->DefaultFunctor;
  }
}

//----------------------------------------------------------------------------
template <class DefaultFunctorType, typename ReturnType>
ReturnType svtkDataArrayDispatcher<DefaultFunctorType, ReturnType>::Go(svtkDataArray* lhs)
{
  switch (lhs->GetDataType())
  {
    svtkTemplateMacro(return (*this->DefaultFunctor)(svtkDataArrayDispatcherPointer<SVTK_TT>(lhs)));
  }
  return ReturnType();
}

#endif // legacy
#endif // svtkDataArrayDispatcher_h
// SVTK-HeaderTest-Exclude: svtkDataArrayDispatcher.h
