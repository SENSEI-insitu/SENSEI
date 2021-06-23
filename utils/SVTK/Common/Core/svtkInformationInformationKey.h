/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationInformationKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationInformationKey
 * @brief   Key for svtkInformation values.
 *
 * svtkInformationInformationKey is used to represent keys in svtkInformation
 * for other information objects.
 */

#ifndef svtkInformationInformationKey_h
#define svtkInformationInformationKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class SVTKCOMMONCORE_EXPORT svtkInformationInformationKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationInformationKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationInformationKey(const char* name, const char* location);
  ~svtkInformationInformationKey() override;

  /**
   * This method simply returns a new svtkInformationInformationKey, given a
   * name and a location. This method is provided for wrappers. Use the
   * constructor directly from C++ instead.
   */
  static svtkInformationInformationKey* MakeKey(const char* name, const char* location)
  {
    return new svtkInformationInformationKey(name, location);
  }

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Set(svtkInformation* info, svtkInformation*);
  svtkInformation* Get(svtkInformation* info);
  //@}

  /**
   * Copy the entry associated with this key from one information
   * object to another.  If there is no entry in the first information
   * object for this key, the value is removed from the second.
   */
  void ShallowCopy(svtkInformation* from, svtkInformation* to) override;

  /**
   * Duplicate (new instance created) the entry associated with this key from
   * one information object to another (new instances of any contained
   * svtkInformation and svtkInformationVector objects are created).
   */
  void DeepCopy(svtkInformation* from, svtkInformation* to) override;

private:
  svtkInformationInformationKey(const svtkInformationInformationKey&) = delete;
  void operator=(const svtkInformationInformationKey&) = delete;
};

#endif
