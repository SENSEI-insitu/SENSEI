/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationStringKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationStringKey
 * @brief   Key for string values in svtkInformation.
 *
 * svtkInformationStringKey is used to represent keys for string values
 * in svtkInformation.
 */

#ifndef svtkInformationStringKey_h
#define svtkInformationStringKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

#include <string> // for std::string compat

class SVTKCOMMONCORE_EXPORT svtkInformationStringKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationStringKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationStringKey(const char* name, const char* location);
  ~svtkInformationStringKey() override;

  /**
   * This method simply returns a new svtkInformationStringKey, given a
   * name and a location. This method is provided for wrappers. Use the
   * constructor directly from C++ instead.
   */
  static svtkInformationStringKey* MakeKey(const char* name, const char* location)
  {
    return new svtkInformationStringKey(name, location);
  }

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Set(svtkInformation* info, const char*);
  void Set(svtkInformation* info, const std::string& str);
  const char* Get(svtkInformation* info);
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
  svtkInformationStringKey(const svtkInformationStringKey&) = delete;
  void operator=(const svtkInformationStringKey&) = delete;
};

#endif
