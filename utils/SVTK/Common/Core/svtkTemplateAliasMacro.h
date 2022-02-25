/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTemplateAliasMacro.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkTemplateAliasMacro
 * @brief   Dispatch a scalar processing template.
 *
 * svtkTemplateAliasMacro is used in a switch statement to
 * automatically generate duplicate code for all enabled scalar types.
 * The code can be written to use SVTK_TT to refer to the type, and
 * each case generated will define SVTK_TT appropriately.  The
 * difference between this and the standard svtkTemplateMacro is that
 * this version will set SVTK_TT to an "alias" for each type.  The
 * alias may be the same type or may be a different type that is the
 * same size/signedness.  This is sufficient when only the numerical
 * value associated with instances of the type is needed, and it
 * avoids unnecessary template instantiations.
 *
 * Example usage:
 *
 *   void* p = dataArray->GetVoidPointer(0);
 *   switch(dataArray->GetDataType())
 *     {
 *     svtkTemplateAliasMacro(svtkMyTemplateFunction(static_cast<SVTK_TT*>(p)));
 *     }
 */

#ifndef svtkTemplateAliasMacro_h
#define svtkTemplateAliasMacro_h

#include "svtkTypeTraits.h"

// Allow individual switching of support for each scalar size/signedness.
// These could be made advanced user options to be configured by CMake.
#define SVTK_USE_INT8 1
#define SVTK_USE_UINT8 1
#define SVTK_USE_INT16 1
#define SVTK_USE_UINT16 1
#define SVTK_USE_INT32 1
#define SVTK_USE_UINT32 1
#define SVTK_USE_INT64 1
#define SVTK_USE_UINT64 1
#define SVTK_USE_FLOAT32 1
#define SVTK_USE_FLOAT64 1

//--------------------------------------------------------------------------

// Define helper macros to switch types on and off.
#define svtkTemplateAliasMacroCase(typeN, call)                                                     \
  svtkTemplateAliasMacroCase0(typeN, call, SVTK_TYPE_SIZED_##typeN)
#define svtkTemplateAliasMacroCase0(typeN, call, sized)                                             \
  svtkTemplateAliasMacroCase1(typeN, call, sized)
#define svtkTemplateAliasMacroCase1(typeN, call, sized)                                             \
  svtkTemplateAliasMacroCase2(typeN, call, SVTK_USE_##sized)
#define svtkTemplateAliasMacroCase2(typeN, call, value)                                             \
  svtkTemplateAliasMacroCase3(typeN, call, value)
#define svtkTemplateAliasMacroCase3(typeN, call, value)                                             \
  svtkTemplateAliasMacroCase_##value(typeN, call)
#define svtkTemplateAliasMacroCase_0(typeN, call)                                                   \
  case SVTK_##typeN:                                                                                \
  {                                                                                                \
    svtkGenericWarningMacro("Support for SVTK_" #typeN " not compiled.");                            \
  };                                                                                               \
  break
#define svtkTemplateAliasMacroCase_1(typeN, call)                                                   \
  case SVTK_##typeN:                                                                                \
  {                                                                                                \
    typedef svtkTypeTraits<SVTK_TYPE_NAME_##typeN>::SizedType SVTK_TT;                                \
    call;                                                                                          \
  };                                                                                               \
  break

// Define a macro to dispatch calls to a template instantiated over
// the aliased scalar types.
#define svtkTemplateAliasMacro(call)                                                                \
  svtkTemplateAliasMacroCase(DOUBLE, call);                                                         \
  svtkTemplateAliasMacroCase(FLOAT, call);                                                          \
  svtkTemplateAliasMacroCase(LONG_LONG, call);                                                      \
  svtkTemplateAliasMacroCase(UNSIGNED_LONG_LONG, call);                                             \
  svtkTemplateAliasMacroCase(ID_TYPE, call);                                                        \
  svtkTemplateAliasMacroCase(LONG, call);                                                           \
  svtkTemplateAliasMacroCase(UNSIGNED_LONG, call);                                                  \
  svtkTemplateAliasMacroCase(INT, call);                                                            \
  svtkTemplateAliasMacroCase(UNSIGNED_INT, call);                                                   \
  svtkTemplateAliasMacroCase(SHORT, call);                                                          \
  svtkTemplateAliasMacroCase(UNSIGNED_SHORT, call);                                                 \
  svtkTemplateAliasMacroCase(CHAR, call);                                                           \
  svtkTemplateAliasMacroCase(SIGNED_CHAR, call);                                                    \
  svtkTemplateAliasMacroCase(UNSIGNED_CHAR, call)

#endif
// SVTK-HeaderTest-Exclude: svtkTemplateAliasMacro.h
