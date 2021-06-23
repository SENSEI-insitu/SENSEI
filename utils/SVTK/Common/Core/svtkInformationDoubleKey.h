/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationDoubleKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationDoubleKey
 * @brief   Key for double values in svtkInformation.
 *
 * svtkInformationDoubleKey is used to represent keys for double values
 * in svtkInformation.
 */

#ifndef svtkInformationDoubleKey_h
#define svtkInformationDoubleKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class SVTKCOMMONCORE_EXPORT svtkInformationDoubleKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationDoubleKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationDoubleKey(const char* name, const char* location);
  ~svtkInformationDoubleKey() override;

  /**
   * This method simply returns a new svtkInformationDoubleKey, given a
   * name and a location. This method is provided for wrappers. Use the
   * constructor directly from C++ instead.
   */
  static svtkInformationDoubleKey* MakeKey(const char* name, const char* location)
  {
    return new svtkInformationDoubleKey(name, location);
  }

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Set(svtkInformation* info, double);
  double Get(svtkInformation* info);
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
  double* GetWatchAddress(svtkInformation* info);

private:
  svtkInformationDoubleKey(const svtkInformationDoubleKey&) = delete;
  void operator=(const svtkInformationDoubleKey&) = delete;
};

#endif
