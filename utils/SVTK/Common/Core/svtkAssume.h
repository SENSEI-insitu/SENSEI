/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAssume.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   SVTK_ASSUME
 * @brief   Provide compiler hints for non-obvious conditions.
 */

#ifndef svtkAssume_h
#define svtkAssume_h

#include "svtkConfigure.h"

#include <cassert>

/**
 * SVTK_ASSUME instructs the compiler that a certain non-obvious condition will
 * *always* be true. Beware that if cond is false at runtime, the results are
 * unpredictable (and likely catastrophic). A runtime assertion is added so
 * that debugging builds may easily catch violations of the condition.

 * A useful application of this macro is when a svtkGenericDataArray subclass has
 * a known number of components at compile time. Adding, for example,
 * SVTK_ASSUME(array->GetNumberOfComponents() == 3); allows the compiler to
 * provide faster access through the GetTypedComponent method, as the fixed data
 * stride in AOS arrays allows advanced optimization of the accesses.

 * A more detailed description of this class and related tools can be found
 * \ref SVTK-7-1-ArrayDispatch "here".
 */
#define SVTK_ASSUME(cond)                                                                           \
  do                                                                                               \
  {                                                                                                \
    const bool c = cond;                                                                           \
    assert("Bad assumption in SVTK_ASSUME: " #cond&& c);                                            \
    SVTK_ASSUME_IMPL(c);                                                                            \
    (void)c;      /* Prevents unused var warnings */                                               \
  } while (false) /* do-while prevents extra semicolon warnings */

#define SVTK_ASSUME_NO_ASSERT(cond)                                                                 \
  do                                                                                               \
  {                                                                                                \
    const bool c = cond;                                                                           \
    SVTK_ASSUME_IMPL(c);                                                                            \
    (void)c;      /* Prevents unused var warnings */                                               \
  } while (false) /* do-while prevents extra semicolon warnings */

// SVTK_ASSUME_IMPL is compiler-specific:
#if defined(SVTK_COMPILER_MSVC) || defined(SVTK_COMPILER_ICC)
#define SVTK_ASSUME_IMPL(cond) __assume(cond)
#elif defined(SVTK_COMPILER_GCC) || defined(SVTK_COMPILER_CLANG)
#define SVTK_ASSUME_IMPL(cond)                                                                      \
  if (!(cond))                                                                                     \
  __builtin_unreachable()
#else
#define SVTK_ASSUME_IMPL(cond)                                                                      \
  do                                                                                               \
  {                                                                                                \
  } while (false) /* no-op */
#endif

#endif // svtkAssume_h
// SVTK-HeaderTest-Exclude: svtkAssume.h
