/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkWeakReference.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkWeakReference
 * @brief   Utility class to hold a weak reference to a svtkObject.
 *
 * Simple Set(...)/Get(...) interface. Used in numpy support to provide a
 * reference to a svtkObject without preventing it from being collected.
 */

#ifndef svtkWeakReference_h
#define svtkWeakReference_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"
#include "svtkWeakPointer.h"

class SVTKCOMMONCORE_EXPORT svtkWeakReference : public svtkObject
{
public:
  svtkTypeMacro(svtkWeakReference, svtkObject);
  static svtkWeakReference* New();
  svtkWeakReference();
  ~svtkWeakReference() override;

  /**
   * Set the svtkObject to maintain a weak reference to.
   */
  void Set(svtkObject* object);

  /**
   * Get the svtkObject pointer or nullptr if the object has been collected.
   */
  svtkObject* Get();

private:
  svtkWeakPointer<svtkObject> Object;
};

#endif

// SVTK-HeaderTest-Exclude: svtkWeakReference.h
