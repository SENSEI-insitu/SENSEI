/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArrayTemplate.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDataArrayTemplate
 *
 * svtkDataArrayTemplate is deprecated, use svtkAOSDataArrayTemplate instead.
 */

#ifndef svtkDataArrayTemplate_h
#define svtkDataArrayTemplate_h

#include "svtkAOSDataArrayTemplate.h"

#ifndef SVTK_LEGACY_REMOVE

template <typename ValueType>
class svtkDataArrayTemplate : public svtkAOSDataArrayTemplate<ValueType>
{
public:
  svtkTemplateTypeMacro(svtkDataArrayTemplate<ValueType>, svtkAOSDataArrayTemplate<ValueType>);

  static svtkDataArrayTemplate<ValueType>* New()
  {
    SVTK_STANDARD_NEW_BODY(svtkDataArrayTemplate<ValueType>);
  }

protected:
  svtkDataArrayTemplate() {}
  ~svtkDataArrayTemplate() override {}

private:
  svtkDataArrayTemplate(const svtkDataArrayTemplate&) = delete;
  void operator=(const svtkDataArrayTemplate&) = delete;
};

#endif // SVTK_LEGACY_REMOVE

#endif // svtkDataArrayTemplate_h

// SVTK-HeaderTest-Exclude: svtkDataArrayTemplate.h
