/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationKeyVectorKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationKeyVectorKey
 * @brief   Key for vector-of-keys values.
 *
 * svtkInformationKeyVectorKey is used to represent keys for
 * vector-of-keys values in svtkInformation.
 */

#ifndef svtkInformationKeyVectorKey_h
#define svtkInformationKeyVectorKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class SVTKCOMMONCORE_EXPORT svtkInformationKeyVectorKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationKeyVectorKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationKeyVectorKey(const char* name, const char* location);
  ~svtkInformationKeyVectorKey() override;

  /**
   * This method simply returns a new svtkInformationKeyVectorKey, given a
   * name and a location. This method is provided for wrappers. Use the
   * constructor directly from C++ instead.
   */
  static svtkInformationKeyVectorKey* MakeKey(const char* name, const char* location)
  {
    return new svtkInformationKeyVectorKey(name, location);
  }

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Append(svtkInformation* info, svtkInformationKey* value);
  void AppendUnique(svtkInformation* info, svtkInformationKey* value);
  void Set(svtkInformation* info, svtkInformationKey* const* value, int length);
  void RemoveItem(svtkInformation* info, svtkInformationKey* value);
  svtkInformationKey** Get(svtkInformation* info);
  svtkInformationKey* Get(svtkInformation* info, int idx);
  void Get(svtkInformation* info, svtkInformationKey** value);
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

private:
  svtkInformationKeyVectorKey(const svtkInformationKeyVectorKey&) = delete;
  void operator=(const svtkInformationKeyVectorKey&) = delete;
};

#endif
