/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAutoInit.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef svtkAutoInit_h
#define svtkAutoInit_h

#include "svtkDebugLeaksManager.h" // DebugLeaks exists longer.
#include "svtkTimeStamp.h"         // Here so that TimeStamp Schwarz initializer works

#define SVTK_MODULE_AUTOINIT SVTK_AUTOINIT

#define SVTK_AUTOINIT(M) SVTK_AUTOINIT0(M, M##_AUTOINIT)
#define SVTK_AUTOINIT0(M, T) SVTK_AUTOINIT1(M, T)
#define SVTK_AUTOINIT1(M, T)                                                                        \
  /* Declare every <mod>_AutoInit_Construct function.  */                                          \
  SVTK_AUTOINIT_DECLARE_##T namespace                                                               \
  {                                                                                                \
    static struct M##_AutoInit                                                                     \
    {                                                                                              \
      /* Call every <mod>_AutoInit_Construct during initialization.  */                            \
      M##_AutoInit() { SVTK_AUTOINIT_CONSTRUCT_##T }                                                \
    } M##_AutoInit_Instance;                                                                       \
  }

#define SVTK_AUTOINIT_DECLARE_0()
#define SVTK_AUTOINIT_DECLARE_1(t1) SVTK_AUTOINIT_DECLARE_0() SVTK_AUTOINIT_DECLARE(t1)
#define SVTK_AUTOINIT_DECLARE_2(t1, t2) SVTK_AUTOINIT_DECLARE_1(t1) SVTK_AUTOINIT_DECLARE(t2)
#define SVTK_AUTOINIT_DECLARE_3(t1, t2, t3) SVTK_AUTOINIT_DECLARE_2(t1, t2) SVTK_AUTOINIT_DECLARE(t3)
#define SVTK_AUTOINIT_DECLARE_4(t1, t2, t3, t4)                                                     \
  SVTK_AUTOINIT_DECLARE_3(t1, t2, t3) SVTK_AUTOINIT_DECLARE(t4)
#define SVTK_AUTOINIT_DECLARE_5(t1, t2, t3, t4, t5)                                                 \
  SVTK_AUTOINIT_DECLARE_4(t1, t2, t3, t4) SVTK_AUTOINIT_DECLARE(t5)
#define SVTK_AUTOINIT_DECLARE_6(t1, t2, t3, t4, t5, t6)                                             \
  SVTK_AUTOINIT_DECLARE_5(t1, t2, t3, t4, t5) SVTK_AUTOINIT_DECLARE(t6)
#define SVTK_AUTOINIT_DECLARE_7(t1, t2, t3, t4, t5, t6, t7)                                         \
  SVTK_AUTOINIT_DECLARE_6(t1, t2, t3, t4, t5, t6) SVTK_AUTOINIT_DECLARE(t7)
#define SVTK_AUTOINIT_DECLARE_8(t1, t2, t3, t4, t5, t6, t7, t8)                                     \
  SVTK_AUTOINIT_DECLARE_7(t1, t2, t3, t4, t5, t6, t7) SVTK_AUTOINIT_DECLARE(t8)
#define SVTK_AUTOINIT_DECLARE_9(t1, t2, t3, t4, t5, t6, t7, t8, t9)                                 \
  SVTK_AUTOINIT_DECLARE_8(t1, t2, t3, t4, t5, t6, t7, t8) SVTK_AUTOINIT_DECLARE(t9)
#define SVTK_AUTOINIT_DECLARE(M) void M##_AutoInit_Construct();

#define SVTK_AUTOINIT_CONSTRUCT_0()
#define SVTK_AUTOINIT_CONSTRUCT_1(t1) SVTK_AUTOINIT_CONSTRUCT_0() SVTK_AUTOINIT_CONSTRUCT(t1)
#define SVTK_AUTOINIT_CONSTRUCT_2(t1, t2) SVTK_AUTOINIT_CONSTRUCT_1(t1) SVTK_AUTOINIT_CONSTRUCT(t2)
#define SVTK_AUTOINIT_CONSTRUCT_3(t1, t2, t3)                                                       \
  SVTK_AUTOINIT_CONSTRUCT_2(t1, t2) SVTK_AUTOINIT_CONSTRUCT(t3)
#define SVTK_AUTOINIT_CONSTRUCT_4(t1, t2, t3, t4)                                                   \
  SVTK_AUTOINIT_CONSTRUCT_3(t1, t2, t3) SVTK_AUTOINIT_CONSTRUCT(t4)
#define SVTK_AUTOINIT_CONSTRUCT_5(t1, t2, t3, t4, t5)                                               \
  SVTK_AUTOINIT_CONSTRUCT_4(t1, t2, t3, t4) SVTK_AUTOINIT_CONSTRUCT(t5)
#define SVTK_AUTOINIT_CONSTRUCT_6(t1, t2, t3, t4, t5, t6)                                           \
  SVTK_AUTOINIT_CONSTRUCT_5(t1, t2, t3, t4, t5) SVTK_AUTOINIT_CONSTRUCT(t6)
#define SVTK_AUTOINIT_CONSTRUCT_7(t1, t2, t3, t4, t5, t6, t7)                                       \
  SVTK_AUTOINIT_CONSTRUCT_6(t1, t2, t3, t4, t5, t6) SVTK_AUTOINIT_CONSTRUCT(t7)
#define SVTK_AUTOINIT_CONSTRUCT_8(t1, t2, t3, t4, t5, t6, t7, t8)                                   \
  SVTK_AUTOINIT_CONSTRUCT_7(t1, t2, t3, t4, t5, t6, t7) SVTK_AUTOINIT_CONSTRUCT(t8)
#define SVTK_AUTOINIT_CONSTRUCT_9(t1, t2, t3, t4, t5, t6, t7, t8, t9)                               \
  SVTK_AUTOINIT_CONSTRUCT_8(t1, t2, t3, t4, t5, t6, t7, t8) SVTK_AUTOINIT_CONSTRUCT(t9)
#define SVTK_AUTOINIT_CONSTRUCT(M) M##_AutoInit_Construct();

// Description:
// Initialize the named module, ensuring its object factory is correctly
// registered. This call must be made in global scope in the
// translation unit of your executable (which can include a shared library, but
// will not work as expected in a static library).
//
// @code{.cpp}
// #include "svtkAutoInit.h"
// SVTK_MODULE_INIT(svtkRenderingOpenGL2);
// @endcode
//
// The above snippet if included in the global scope will ensure the object
// factories for svtkRenderingOpenGL2 are correctly registered and unregistered.
#define SVTK_MODULE_INIT(M)                                                                         \
  SVTK_AUTOINIT_DECLARE(M) namespace                                                                \
  {                                                                                                \
    static struct M##_ModuleInit                                                                   \
    {                                                                                              \
      /* Call <mod>_AutoInit_Construct during initialization.  */                                  \
      M##_ModuleInit() { SVTK_AUTOINIT_CONSTRUCT(M) }                                               \
    } M##_ModuleInit_Instance;                                                                     \
  }

#endif
// SVTK-HeaderTest-Exclude: svtkAutoInit.h
