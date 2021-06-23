/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationVariantKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationVariantKey
 * @brief   Key for variant values in svtkInformation.
 *
 * svtkInformationVariantKey is used to represent keys for variant values
 * in svtkInformation.
 */

#ifndef svtkInformationVariantKey_h
#define svtkInformationVariantKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class svtkVariant;

class SVTKCOMMONCORE_EXPORT svtkInformationVariantKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationVariantKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationVariantKey(const char* name, const char* location);
  ~svtkInformationVariantKey() override;

  /**
   * This method simply returns a new svtkInformationVariantKey, given a
   * name and a location. This method is provided for wrappers. Use the
   * constructor directly from C++ instead.
   */
  static svtkInformationVariantKey* MakeKey(const char* name, const char* location)
  {
    return new svtkInformationVariantKey(name, location);
  }

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Set(svtkInformation* info, const svtkVariant&);
  const svtkVariant& Get(svtkInformation* info);
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
  svtkVariant* GetWatchAddress(svtkInformation* info);

private:
  svtkInformationVariantKey(const svtkInformationVariantKey&) = delete;
  void operator=(const svtkInformationVariantKey&) = delete;
};

#endif
