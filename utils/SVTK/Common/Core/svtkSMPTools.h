/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSMPTools.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkSMPTools
 * @brief   A set of parallel (multi-threaded) utility functions.
 *
 * svtkSMPTools provides a set of utility functions that can
 * be used to parallelize parts of SVTK code using multiple threads.
 * There are several back-end implementations of parallel functionality
 * (currently Sequential, TBB and X-Kaapi) that actual execution is
 * delegated to.
 */

#ifndef svtkSMPTools_h
#define svtkSMPTools_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

#include "svtkSMPThreadLocal.h" // For Initialized
#include "svtkSMPToolsInternal.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#ifndef __SVTK_WRAP__
namespace svtk
{
namespace detail
{
namespace smp
{
template <typename T>
class svtkSMPTools_Has_Initialize
{
  typedef char (&no_type)[1];
  typedef char (&yes_type)[2];
  template <typename U, void (U::*)()>
  struct V
  {
  };
  template <typename U>
  static yes_type check(V<U, &U::Initialize>*);
  template <typename U>
  static no_type check(...);

public:
  static bool const value = sizeof(check<T>(nullptr)) == sizeof(yes_type);
};

template <typename T>
class svtkSMPTools_Has_Initialize_const
{
  typedef char (&no_type)[1];
  typedef char (&yes_type)[2];
  template <typename U, void (U::*)() const>
  struct V
  {
  };
  template <typename U>
  static yes_type check(V<U, &U::Initialize>*);
  template <typename U>
  static no_type check(...);

public:
  static bool const value = sizeof(check<T>(0)) == sizeof(yes_type);
};

template <typename Functor, bool Init>
struct svtkSMPTools_FunctorInternal;

template <typename Functor>
struct svtkSMPTools_FunctorInternal<Functor, false>
{
  Functor& F;
  svtkSMPTools_FunctorInternal(Functor& f)
    : F(f)
  {
  }
  void Execute(svtkIdType first, svtkIdType last) { this->F(first, last); }
  void For(svtkIdType first, svtkIdType last, svtkIdType grain)
  {
    svtk::detail::smp::svtkSMPTools_Impl_For(first, last, grain, *this);
  }
  svtkSMPTools_FunctorInternal<Functor, false>& operator=(
    const svtkSMPTools_FunctorInternal<Functor, false>&);
  svtkSMPTools_FunctorInternal<Functor, false>(const svtkSMPTools_FunctorInternal<Functor, false>&);
};

template <typename Functor>
struct svtkSMPTools_FunctorInternal<Functor, true>
{
  Functor& F;
  svtkSMPThreadLocal<unsigned char> Initialized;
  svtkSMPTools_FunctorInternal(Functor& f)
    : F(f)
    , Initialized(0)
  {
  }
  void Execute(svtkIdType first, svtkIdType last)
  {
    unsigned char& inited = this->Initialized.Local();
    if (!inited)
    {
      this->F.Initialize();
      inited = 1;
    }
    this->F(first, last);
  }
  void For(svtkIdType first, svtkIdType last, svtkIdType grain)
  {
    svtk::detail::smp::svtkSMPTools_Impl_For(first, last, grain, *this);
    this->F.Reduce();
  }
  svtkSMPTools_FunctorInternal<Functor, true>& operator=(
    const svtkSMPTools_FunctorInternal<Functor, true>&);
  svtkSMPTools_FunctorInternal<Functor, true>(const svtkSMPTools_FunctorInternal<Functor, true>&);
};

template <typename Functor>
class svtkSMPTools_Lookup_For
{
  static bool const init = svtkSMPTools_Has_Initialize<Functor>::value;

public:
  typedef svtkSMPTools_FunctorInternal<Functor, init> type;
};

template <typename Functor>
class svtkSMPTools_Lookup_For<Functor const>
{
  static bool const init = svtkSMPTools_Has_Initialize_const<Functor>::value;

public:
  typedef svtkSMPTools_FunctorInternal<Functor const, init> type;
};
} // namespace smp
} // namespace detail
} // namespace svtk
#endif // __SVTK_WRAP__
#endif // DOXYGEN_SHOULD_SKIP_THIS

class SVTKCOMMONCORE_EXPORT svtkSMPTools
{
public:
  //@{
  /**
   * Execute a for operation in parallel. First and last
   * define the range over which to operate (which is defined
   * by the operator). The operation executed is defined by
   * operator() of the functor object. The grain gives the parallel
   * engine a hint about the coarseness over which to parallelize
   * the function (as defined by last-first of each execution of
   * operator() ).
   */
  template <typename Functor>
  static void For(svtkIdType first, svtkIdType last, svtkIdType grain, Functor& f)
  {
    typename svtk::detail::smp::svtkSMPTools_Lookup_For<Functor>::type fi(f);
    fi.For(first, last, grain);
  }
  //@}

  //@{
  /**
   * Execute a for operation in parallel. First and last
   * define the range over which to operate (which is defined
   * by the operator). The operation executed is defined by
   * operator() of the functor object. The grain gives the parallel
   * engine a hint about the coarseness over which to parallelize
   * the function (as defined by last-first of each execution of
   * operator() ).
   */
  template <typename Functor>
  static void For(svtkIdType first, svtkIdType last, svtkIdType grain, Functor const& f)
  {
    typename svtk::detail::smp::svtkSMPTools_Lookup_For<Functor const>::type fi(f);
    fi.For(first, last, grain);
  }
  //@}

  /**
   * Execute a for operation in parallel. First and last
   * define the range over which to operate (which is defined
   * by the operator). The operation executed is defined by
   * operator() of the functor object. The grain gives the parallel
   * engine a hint about the coarseness over which to parallelize
   * the function (as defined by last-first of each execution of
   * operator() ). Uses a default value for the grain.
   */
  template <typename Functor>
  static void For(svtkIdType first, svtkIdType last, Functor& f)
  {
    svtkSMPTools::For(first, last, 0, f);
  }

  /**
   * Execute a for operation in parallel. First and last
   * define the range over which to operate (which is defined
   * by the operator). The operation executed is defined by
   * operator() of the functor object. The grain gives the parallel
   * engine a hint about the coarseness over which to parallelize
   * the function (as defined by last-first of each execution of
   * operator() ). Uses a default value for the grain.
   */
  template <typename Functor>
  static void For(svtkIdType first, svtkIdType last, Functor const& f)
  {
    svtkSMPTools::For(first, last, 0, f);
  }

  /**
   * Initialize the underlying libraries for execution. This is
   * not required as it is automatically called before the first
   * execution of any parallel code. However, it can be used to
   * control the maximum number of threads used when the back-end
   * supports it (currently Simple and TBB only). Make sure to call
   * it before any other parallel operation.
   * When using Kaapi, use the KAAPI_CPUCOUNT env. variable to control
   * the number of threads used in the thread pool.
   */
  static void Initialize(int numThreads = 0);

  /**
   * Get the estimated number of threads being used by the backend.
   * This should be used as just an estimate since the number of threads may
   * vary dynamically and a particular task may not be executed on all the
   * available threads.
   */
  static int GetEstimatedNumberOfThreads();

  /**
   * A convenience method for sorting data. It is a drop in replacement for
   * std::sort(). Under the hood different methods are used. For example,
   * tbb::parallel_sort is used in TBB.
   */
  template <typename RandomAccessIterator>
  static void Sort(RandomAccessIterator begin, RandomAccessIterator end)
  {
    svtk::detail::smp::svtkSMPTools_Impl_Sort(begin, end);
  }

  /**
   * A convenience method for sorting data. It is a drop in replacement for
   * std::sort(). Under the hood different methods are used. For example,
   * tbb::parallel_sort is used in TBB. This version of Sort() takes a
   * comparison class.
   */
  template <typename RandomAccessIterator, typename Compare>
  static void Sort(RandomAccessIterator begin, RandomAccessIterator end, Compare comp)
  {
    svtk::detail::smp::svtkSMPTools_Impl_Sort(begin, end, comp);
  }
};

#endif
// SVTK-HeaderTest-Exclude: svtkSMPTools.h
