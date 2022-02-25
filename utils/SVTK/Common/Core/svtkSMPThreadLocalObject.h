/*=========================================================================

 Program:   Visualization Toolkit
 Module:    svtkSMPThreadLocalObject.h

 Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
 All rights reserved.
 See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

    This software is distributed WITHOUT ANY WARRANTY; without even
    the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
    PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkSMPThreadLocalObject
 * @brief   Thread local storage for SVTK objects.
 *
 * This class essentially does the same thing as svtkSMPThreadLocal with
 * 2 additional functions:
 * - Local() allocates an object of the template argument type using ::New
 * - The destructor calls Delete() on all objects created with Local().
 *
 * @warning
 * There is absolutely no guarantee to the order in which the local objects
 * will be stored and hence the order in which they will be traversed when
 * using iterators. You should not even assume that two svtkSMPThreadLocal
 * populated in the same parallel section will be populated in the same
 * order. For example, consider the following
 * \verbatim
 * svtkSMPThreadLocal<int> Foo;
 * svtkSMPThreadLocal<int> Bar;
 * class AFunctor
 * {
 *    void Initialize() const
 *    {
 *      int& foo = Foo.Local();
 *      int& bar = Bar.Local();
 *      foo = random();
 *      bar = foo;
 *    }
 *
 * @warning
 *    void operator()(svtkIdType, svtkIdType)
 *    {}
 *    void Finalize()
 *    {}
 * };
 *
 * @warning
 * AFunctor functor;
 * svtkSMPTools::For(0, 100000, functor);
 *
 * @warning
 * svtkSMPThreadLocal<int>::iterator itr1 = Foo.begin();
 * svtkSMPThreadLocal<int>::iterator itr2 = Bar.begin();
 * while (itr1 != Foo.end())
 * {
 *   assert(*itr1 == *itr2);
 *   ++itr1; ++itr2;
 * }
 * \endverbatim
 *
 * @warning
 * It is possible and likely that the assert() will fail using the TBB
 * backend. So if you need to store values related to each other and
 * iterate over them together, use a struct or class to group them together
 * and use a thread local of that class.
 *
 * @sa
 * svtkSMPThreadLocal
 */

#ifndef svtkSMPThreadLocalObject_h
#define svtkSMPThreadLocalObject_h

#include "svtkSMPThreadLocal.h"

template <typename T>
class svtkSMPThreadLocalObject
{
  typedef svtkSMPThreadLocal<T*> TLS;
  typedef typename svtkSMPThreadLocal<T*>::iterator TLSIter;

  // Hide the copy constructor for now and assignment
  // operator for now.
  svtkSMPThreadLocalObject(const svtkSMPThreadLocalObject&);
  void operator=(const svtkSMPThreadLocalObject&);

public:
  /**
   * Default constructor.
   */
  svtkSMPThreadLocalObject()
    : Internal(nullptr)
    , Exemplar(nullptr)
  {
  }

  svtkSMPThreadLocalObject(T* const& exemplar)
    : Internal(0)
    , Exemplar(exemplar)
  {
  }

  virtual ~svtkSMPThreadLocalObject()
  {
    iterator iter = this->begin();
    while (iter != this->end())
    {
      if (*iter)
      {
        (*iter)->Delete();
      }
      ++iter;
    }
  }

  //@{
  /**
   * Returns an object local to the current thread.
   * This object is allocated with ::New() and will
   * be deleted in the destructor of svtkSMPThreadLocalObject.
   */
  T*& Local()
  {
    T*& svtkobject = this->Internal.Local();
    if (!svtkobject)
    {
      if (this->Exemplar)
      {
        svtkobject = this->Exemplar->NewInstance();
      }
      else
      {
        svtkobject = T::SafeDownCast(T::New());
      }
    }
    return svtkobject;
  }
  //@}

  /**
   * Return the number of thread local objects that have been initialized
   */
  size_t size() const { return this->Internal.size(); }

  //@{
  /**
   * Subset of the standard iterator API.
   * The most common design pattern is to use iterators in a sequential
   * code block and to use only the thread local objects in parallel
   * code blocks.
   */
  class iterator
  {
  public:
    iterator& operator++()
    {
      ++this->Iter;
      return *this;
    }
    //@}

    iterator operator++(int)
    {
      iterator copy = *this;
      ++this->Iter;
      return copy;
    }

    bool operator==(const iterator& other) { return this->Iter == other.Iter; }

    bool operator!=(const iterator& other) { return this->Iter != other.Iter; }

    T*& operator*() { return *this->Iter; }

    T** operator->() { return &*this->Iter; }

  private:
    TLSIter Iter;

    friend class svtkSMPThreadLocalObject<T>;
  };

  iterator begin()
  {
    iterator iter;
    iter.Iter = this->Internal.begin();
    return iter;
  };

  iterator end()
  {
    iterator iter;
    iter.Iter = this->Internal.end();
    return iter;
  }

private:
  TLS Internal;
  T* Exemplar;
};

#endif
// SVTK-HeaderTest-Exclude: svtkSMPThreadLocalObject.h
