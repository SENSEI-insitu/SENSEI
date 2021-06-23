/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationIntegerKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationIntegerKey
 * @brief   Key for integer values in svtkInformation.
 *
 * svtkInformationIntegerKey is used to represent keys for integer values
 * in svtkInformation.
 */

#ifndef svtkInformationIntegerKey_h
#define svtkInformationIntegerKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class SVTKCOMMONCORE_EXPORT svtkInformationIntegerKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationIntegerKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationIntegerKey(const char* name, const char* location);
  ~svtkInformationIntegerKey() override;

  /**
   * This method simply returns a new svtkInformationIntegerKey, given a
   * name and a location. This method is provided for wrappers. Use the
   * constructor directly from C++ instead.
   */
  static svtkInformationIntegerKey* MakeKey(const char* name, const char* location)
  {
    return new svtkInformationIntegerKey(name, location);
  }

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Set(svtkInformation* info, int);
  int Get(svtkInformation* info);
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
  int* GetWatchAddress(svtkInformation* info);

private:
  svtkInformationIntegerKey(const svtkInformationIntegerKey&) = delete;
  void operator=(const svtkInformationIntegerKey&) = delete;
};

#endif
