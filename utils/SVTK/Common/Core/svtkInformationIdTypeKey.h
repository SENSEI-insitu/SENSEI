/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationIdTypeKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationIdTypeKey
 * @brief   Key for svtkIdType values in svtkInformation.
 *
 * svtkInformationIdTypeKey is used to represent keys for svtkIdType values
 * in svtkInformation.
 */

#ifndef svtkInformationIdTypeKey_h
#define svtkInformationIdTypeKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class SVTKCOMMONCORE_EXPORT svtkInformationIdTypeKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationIdTypeKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationIdTypeKey(const char* name, const char* location);
  ~svtkInformationIdTypeKey() override;

  /**
   * This method simply returns a new svtkInformationIdTypeKey, given a
   * name and a location. This method is provided for wrappers. Use the
   * constructor directly from C++ instead.
   */
  static svtkInformationIdTypeKey* MakeKey(const char* name, const char* location)
  {
    return new svtkInformationIdTypeKey(name, location);
  }

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Set(svtkInformation* info, svtkIdType);
  svtkIdType Get(svtkInformation* info);
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

protected:
  /**
   * Get the address at which the actual value is stored.  This is
   * meant for use from a debugger to add watches and is therefore not
   * a public method.
   */
  svtkIdType* GetWatchAddress(svtkInformation* info);

private:
  svtkInformationIdTypeKey(const svtkInformationIdTypeKey&) = delete;
  void operator=(const svtkInformationIdTypeKey&) = delete;
};

#endif
