/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationRequestKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationRequestKey
 * @brief   Key for pointer to pointer.
 *
 * svtkInformationRequestKey is used to represent keys for pointer
 * to pointer values in svtkInformation.h
 */

#ifndef svtkInformationRequestKey_h
#define svtkInformationRequestKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class SVTKCOMMONCORE_EXPORT svtkInformationRequestKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationRequestKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationRequestKey(const char* name, const char* location);
  ~svtkInformationRequestKey() override;

  /**
   * This method simply returns a new svtkInformationRequestKey, given a
   * name and a location. This method is provided for wrappers. Use the
   * constructor directly from C++ instead.
   */
  static svtkInformationRequestKey* MakeKey(const char* name, const char* location)
  {
    return new svtkInformationRequestKey(name, location);
  }

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Set(svtkInformation* info);
  void Remove(svtkInformation* info) override;
  int Has(svtkInformation* info) override;
  //@}

  /**
   * Copy the entry associated with this key from one information
   * object to another.  If there is no entry in the first information
   * object for this key, the value is removed from the second.
   */
  void ShallowCopy(svtkInformation* from, svtkInformation* to) override;

  /**
   * Print the key's value in an information object to a stream.
   */
  void Print(ostream& os, svtkInformation* info) override;

private:
  svtkInformationRequestKey(const svtkInformationRequestKey&) = delete;
  void operator=(const svtkInformationRequestKey&) = delete;
};

#endif
