/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkArrayIterator
 * @brief   Abstract superclass to iterate over elements
 * in an svtkAbstractArray.
 *
 *
 * svtkArrayIterator is used to iterate over elements in any
 * svtkAbstractArray subclass.  The svtkArrayIteratorTemplateMacro is used
 * to centralize the set of types supported by Execute methods.  It also
 * avoids duplication of long switch statement case lists.
 *
 * Note that in this macro SVTK_TT is defined to be the type of the
 * iterator for the given type of array. One must include the
 * svtkArrayIteratorIncludes.h header file to provide for extending of
 * this macro by addition of new iterators.
 *
 * Example usage:
 * \code
 * svtkArrayIter* iter = array->NewIterator();
 * switch(array->GetDataType())
 *   {
 *   svtkArrayIteratorTemplateMacro(myFunc(static_cast<SVTK_TT*>(iter), arg2));
 *   }
 * iter->Delete();
 * \endcode
 */

#ifndef svtkArrayIterator_h
#define svtkArrayIterator_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"
class svtkAbstractArray;
class SVTKCOMMONCORE_EXPORT svtkArrayIterator : public svtkObject
{
public:
  svtkTypeMacro(svtkArrayIterator, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Set the array this iterator will iterate over.
   * After Initialize() has been called, the iterator is valid
   * so long as the Array has not been modified
   * (except using the iterator itself).
   * If the array is modified, the iterator must be re-initialized.
   */
  virtual void Initialize(svtkAbstractArray* array) = 0;

  /**
   * Get the data type from the underlying array. Returns 0 if
   * no underlying array is present.
   */
  virtual int GetDataType() const = 0;

protected:
  svtkArrayIterator();
  ~svtkArrayIterator() override;

private:
  svtkArrayIterator(const svtkArrayIterator&) = delete;
  void operator=(const svtkArrayIterator&) = delete;
};

#endif
