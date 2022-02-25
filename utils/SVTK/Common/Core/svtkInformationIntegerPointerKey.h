/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationIntegerPointerKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationIntegerPointerKey
 * @brief   Key for pointer to integer.
 *
 * svtkInformationIntegerPointerKey is used to represent keys for pointer
 * to integer values in svtkInformation.h
 */

#ifndef svtkInformationIntegerPointerKey_h
#define svtkInformationIntegerPointerKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class SVTKCOMMONCORE_EXPORT svtkInformationIntegerPointerKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationIntegerPointerKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationIntegerPointerKey(const char* name, const char* location, int length = -1);
  ~svtkInformationIntegerPointerKey() override;

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Set(svtkInformation* info, int* value, int length);
  int* Get(svtkInformation* info);
  void Get(svtkInformation* info, int* value);
  int Length(svtkInformation* info);
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
  // The required length of the vector value (-1 is no restriction).
  int RequiredLength;

  /**
   * Get the address at which the actual value is stored.  This is
   * meant for use from a debugger to add watches and is therefore not
   * a public method.
   */
  int* GetWatchAddress(svtkInformation* info);

private:
  svtkInformationIntegerPointerKey(const svtkInformationIntegerPointerKey&) = delete;
  void operator=(const svtkInformationIntegerPointerKey&) = delete;
};

#endif
