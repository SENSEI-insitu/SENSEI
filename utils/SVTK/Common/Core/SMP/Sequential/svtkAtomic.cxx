/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAtomic.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkAtomic.h"

#if !defined(SVTK_GCC_ATOMICS_32) && !defined(SVTK_APPLE_ATOMICS_32) &&                              \
  !defined(SVTK_WINDOWS_ATOMICS_32)
#define SVTK_LOCK_BASED_ATOMICS_32
#endif

#if !defined(SVTK_GCC_ATOMICS_64) && !defined(SVTK_APPLE_ATOMICS_64) &&                              \
  !defined(SVTK_WINDOWS_ATOMICS_64)
#define SVTK_LOCK_BASED_ATOMICS_64
#endif

#if defined(SVTK_WINDOWS_ATOMICS_32) || defined(SVTK_WINDOWS_ATOMICS_64)
#include "svtkWindows.h"
#endif

#if defined(SVTK_LOCK_BASED_ATOMICS_32) || defined(SVTK_LOCK_BASED_ATOMICS_64)

#include "svtkSimpleCriticalSection.h"

class CriticalSectionGuard
{
public:
  CriticalSectionGuard(svtkSimpleCriticalSection& cs)
    : CriticalSection(cs)
  {
    this->CriticalSection.Lock();
  }

  ~CriticalSectionGuard() { this->CriticalSection.Unlock(); }

private:
  // not copyable
  CriticalSectionGuard(const CriticalSectionGuard&);
  void operator=(const CriticalSectionGuard&);

  svtkSimpleCriticalSection& CriticalSection;
};

#if defined(SVTK_LOCK_BASED_ATOMICS_64)
detail::AtomicOps<8>::atomic_type::atomic_type(svtkTypeInt64 init)
  : var(init)
{
  this->csec = new svtkSimpleCriticalSection;
}

detail::AtomicOps<8>::atomic_type::~atomic_type()
{
  delete this->csec;
}
#endif

#if defined(SVTK_LOCK_BASED_ATOMICS_32)
detail::AtomicOps<4>::atomic_type::atomic_type(svtkTypeInt32 init)
  : var(init)
{
  this->csec = new svtkSimpleCriticalSection;
}

detail::AtomicOps<4>::atomic_type::~atomic_type()
{
  delete this->csec;
}
#endif

#endif // SVTK_LOCK_BASED_ATOMICS

namespace detail
{

#if defined(SVTK_WINDOWS_ATOMICS_64) || defined(SVTK_LOCK_BASED_ATOMICS_64)

svtkTypeInt64 AtomicOps<8>::AddAndFetch(atomic_type* ref, svtkTypeInt64 val)
{
#if defined(SVTK_WINDOWS_ATOMICS_64)
#if defined(SVTK_HAS_INTERLOCKEDADD)
  return InterlockedAdd64(ref, val);
#else
  return InterlockedExchangeAdd64(ref, val) + val;
#endif
#else
  CriticalSectionGuard csg(*ref->csec);
  return ref->var += val;
#endif
}

svtkTypeInt64 AtomicOps<8>::SubAndFetch(atomic_type* ref, svtkTypeInt64 val)
{
#if defined(SVTK_WINDOWS_ATOMICS_64)
#if defined(SVTK_HAS_INTERLOCKEDADD)
  return InterlockedAdd64(ref, -val);
#else
  return InterlockedExchangeAdd64(ref, -val) - val;
#endif
#else
  CriticalSectionGuard csg(*ref->csec);
  return ref->var -= val;
#endif
}

svtkTypeInt64 AtomicOps<8>::PreIncrement(atomic_type* ref)
{
#if defined(SVTK_WINDOWS_ATOMICS_64)
  return InterlockedIncrement64(ref);
#else
  CriticalSectionGuard csg(*ref->csec);
  return ++(ref->var);
#endif
}

svtkTypeInt64 AtomicOps<8>::PreDecrement(atomic_type* ref)
{
#if defined(SVTK_WINDOWS_ATOMICS_64)
  return InterlockedDecrement64(ref);
#else
  CriticalSectionGuard csg(*ref->csec);
  return --(ref->var);
#endif
}

svtkTypeInt64 AtomicOps<8>::PostIncrement(atomic_type* ref)
{
#if defined(SVTK_WINDOWS_ATOMICS_64)
  svtkTypeInt64 val = InterlockedIncrement64(ref);
  return --val;
#else
  CriticalSectionGuard csg(*ref->csec);
  return (ref->var)++;
#endif
}

svtkTypeInt64 AtomicOps<8>::PostDecrement(atomic_type* ref)
{
#if defined(SVTK_WINDOWS_ATOMICS_64)
  svtkTypeInt64 val = InterlockedDecrement64(ref);
  return ++val;
#else
  CriticalSectionGuard csg(*ref->csec);
  return (ref->var)--;
#endif
}

svtkTypeInt64 AtomicOps<8>::Load(const atomic_type* ref)
{
#if defined(SVTK_WINDOWS_ATOMICS_64)
  svtkTypeInt64 val;
  InterlockedExchange64(&val, *ref);
  return val;
#else
  CriticalSectionGuard csg(*ref->csec);
  return ref->var;
#endif
}

void AtomicOps<8>::Store(atomic_type* ref, svtkTypeInt64 val)
{
#if defined(SVTK_WINDOWS_ATOMICS_64)
  InterlockedExchange64(ref, val);
#else
  CriticalSectionGuard csg(*ref->csec);
  ref->var = val;
#endif
}

#endif // defined(SVTK_WINDOWS_ATOMICS_64) || defined(SVTK_LOCK_BASED_ATOMICS_64)

#if defined(SVTK_WINDOWS_ATOMICS_32) || defined(SVTK_LOCK_BASED_ATOMICS_32)

svtkTypeInt32 AtomicOps<4>::AddAndFetch(atomic_type* ref, svtkTypeInt32 val)
{
#if defined(SVTK_WINDOWS_ATOMICS_32)
#if defined(SVTK_HAS_INTERLOCKEDADD)
  return InterlockedAdd(reinterpret_cast<long*>(ref), val);
#else
  return InterlockedExchangeAdd(reinterpret_cast<long*>(ref), val) + val;
#endif
#else
  CriticalSectionGuard csg(*ref->csec);
  return ref->var += val;
#endif
}

svtkTypeInt32 AtomicOps<4>::SubAndFetch(atomic_type* ref, svtkTypeInt32 val)
{
#if defined(SVTK_WINDOWS_ATOMICS_32)
#if defined(SVTK_HAS_INTERLOCKEDADD)
  return InterlockedAdd(reinterpret_cast<long*>(ref), -val);
#else
  return InterlockedExchangeAdd(reinterpret_cast<long*>(ref), -val) - val;
#endif
#else
  CriticalSectionGuard csg(*ref->csec);
  return ref->var -= val;
#endif
}

svtkTypeInt32 AtomicOps<4>::PreIncrement(atomic_type* ref)
{
#if defined(SVTK_WINDOWS_ATOMICS_32)
  return InterlockedIncrement(reinterpret_cast<long*>(ref));
#else
  CriticalSectionGuard csg(*ref->csec);
  return ++(ref->var);
#endif
}

svtkTypeInt32 AtomicOps<4>::PreDecrement(atomic_type* ref)
{
#if defined(SVTK_WINDOWS_ATOMICS_32)
  return InterlockedDecrement(reinterpret_cast<long*>(ref));
#else
  CriticalSectionGuard csg(*ref->csec);
  return --(ref->var);
#endif
}

svtkTypeInt32 AtomicOps<4>::PostIncrement(atomic_type* ref)
{
#if defined(SVTK_WINDOWS_ATOMICS_32)
  svtkTypeInt32 val = InterlockedIncrement(reinterpret_cast<long*>(ref));
  return --val;
#else
  CriticalSectionGuard csg(*ref->csec);
  return (ref->var)++;
#endif
}

svtkTypeInt32 AtomicOps<4>::PostDecrement(atomic_type* ref)
{
#if defined(SVTK_WINDOWS_ATOMICS_32)
  svtkTypeInt32 val = InterlockedDecrement(reinterpret_cast<long*>(ref));
  return ++val;
#else
  CriticalSectionGuard csg(*ref->csec);
  return (ref->var)--;
#endif
}

svtkTypeInt32 AtomicOps<4>::Load(const atomic_type* ref)
{
#if defined(SVTK_WINDOWS_ATOMICS_32)
  long val;
  InterlockedExchange(&val, *ref);
  return val;
#else
  CriticalSectionGuard csg(*ref->csec);
  return ref->var;
#endif
}

void AtomicOps<4>::Store(atomic_type* ref, svtkTypeInt32 val)
{
#if defined(SVTK_WINDOWS_ATOMICS_32)
  InterlockedExchange(reinterpret_cast<long*>(ref), val);
#else
  CriticalSectionGuard csg(*ref->csec);
  ref->var = val;
#endif
}

#endif // defined(SVTK_WINDOWS_ATOMICS_32) || defined(SVTK_LOCK_BASED_ATOMICS_32)

} // namespace detail
