/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationUnsignedLongKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationUnsignedLongKey
 * @brief   Key for unsigned long values in svtkInformation.
 *
 * svtkInformationUnsignedLongKey is used to represent keys for unsigned long values
 * in svtkInformation.
 */

#ifndef svtkInformationUnsignedLongKey_h
#define svtkInformationUnsignedLongKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class SVTKCOMMONCORE_EXPORT svtkInformationUnsignedLongKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationUnsignedLongKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationUnsignedLongKey(const char* name, const char* location);
  ~svtkInformationUnsignedLongKey() override;

  /**
   * This method simply returns a new svtkInformationUnsignedLongKey, given a
   * name and a location. This method is provided for wrappers. Use the
   * constructor directly from C++ instead.
   */
  static svtkInformationUnsignedLongKey* MakeKey(const char* name, const char* location)
  {
    return new svtkInformationUnsignedLongKey(name, location);
  }

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Set(svtkInformation* info, unsigned long);
  unsigned long Get(svtkInformation* info);
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
  unsigned long* GetWatchAddress(svtkInformation* info);

private:
  svtkInformationUnsignedLongKey(const svtkInformationUnsignedLongKey&) = delete;
  void operator=(const svtkInformationUnsignedLongKey&) = delete;
};

#endif
