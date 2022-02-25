/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationVariantVectorKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationVariantVectorKey
 * @brief   Key for variant vector values.
 *
 * svtkInformationVariantVectorKey is used to represent keys for variant
 * vector values in svtkInformation.h
 */

#ifndef svtkInformationVariantVectorKey_h
#define svtkInformationVariantVectorKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class svtkVariant;

class SVTKCOMMONCORE_EXPORT svtkInformationVariantVectorKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationVariantVectorKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationVariantVectorKey(const char* name, const char* location, int length = -1);
  ~svtkInformationVariantVectorKey() override;

  /**
   * This method simply returns a new svtkInformationVariantVectorKey, given a
   * name, a location and a required length. This method is provided for
   * wrappers. Use the constructor directly from C++ instead.
   */
  static svtkInformationVariantVectorKey* MakeKey(
    const char* name, const char* location, int length = -1)
  {
    return new svtkInformationVariantVectorKey(name, location, length);
  }

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Append(svtkInformation* info, const svtkVariant& value);
  void Set(svtkInformation* info, const svtkVariant* value, int length);
  const svtkVariant* Get(svtkInformation* info) const;
  const svtkVariant& Get(svtkInformation* info, int idx) const;
  void Get(svtkInformation* info, svtkVariant* value) const;
  int Length(svtkInformation* info) const;
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

private:
  svtkInformationVariantVectorKey(const svtkInformationVariantVectorKey&) = delete;
  void operator=(const svtkInformationVariantVectorKey&) = delete;
};

#endif
