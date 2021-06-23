/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationIterator
 * @brief   Iterates over keys of an information object
 *
 * svtkInformationIterator can be used to iterate over the keys of an
 * information object. The corresponding values can then be directly
 * obtained from the information object using the keys.
 *
 * @sa
 * svtkInformation svtkInformationKey
 */

#ifndef svtkInformationIterator_h
#define svtkInformationIterator_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

class svtkInformation;
class svtkInformationKey;
class svtkInformationIteratorInternals;

class SVTKCOMMONCORE_EXPORT svtkInformationIterator : public svtkObject
{
public:
  static svtkInformationIterator* New();
  svtkTypeMacro(svtkInformationIterator, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Set/Get the information to iterator over.
   */
  void SetInformation(svtkInformation*);
  svtkGetObjectMacro(Information, svtkInformation);
  //@}

  /**
   * Set the function to iterate over. The iterator
   * will not hold a reference to the information object.
   * Can be used to optimize certain places by avoiding
   * garbage collection.
   */
  void SetInformationWeak(svtkInformation*);

  /**
   * Move the iterator to the beginning of the collection.
   */
  void InitTraversal() { this->GoToFirstItem(); }

  /**
   * Move the iterator to the beginning of the collection.
   */
  virtual void GoToFirstItem();

  /**
   * Move the iterator to the next item in the collection.
   */
  virtual void GoToNextItem();

  /**
   * Test whether the iterator is currently pointing to a valid
   * item. Returns 1 for yes, 0 for no.
   */
  virtual int IsDoneWithTraversal();

  /**
   * Get the current item. Valid only when IsDoneWithTraversal()
   * returns 1.
   */
  virtual svtkInformationKey* GetCurrentKey();

protected:
  svtkInformationIterator();
  ~svtkInformationIterator() override;

  svtkInformation* Information;
  svtkInformationIteratorInternals* Internal;

  bool ReferenceIsWeak;

private:
  svtkInformationIterator(const svtkInformationIterator&) = delete;
  void operator=(const svtkInformationIterator&) = delete;
};

#endif
