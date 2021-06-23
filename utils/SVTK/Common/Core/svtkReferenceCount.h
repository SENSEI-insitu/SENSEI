/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkReferenceCount.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkReferenceCount
 * @brief   Obsolete / empty subclass of object.
 *
 * svtkReferenceCount functionality has now been moved into svtkObject
 * @sa
 * svtkObject
 */

#ifndef svtkReferenceCount_h
#define svtkReferenceCount_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

class SVTKCOMMONCORE_EXPORT svtkReferenceCount : public svtkObject
{
public:
  static svtkReferenceCount* New();

  svtkTypeMacro(svtkReferenceCount, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

protected:
  svtkReferenceCount();
  ~svtkReferenceCount() override;

private:
  svtkReferenceCount(const svtkReferenceCount&) = delete;
  void operator=(const svtkReferenceCount&) = delete;
};

#endif
