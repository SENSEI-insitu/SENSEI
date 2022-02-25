/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationDoubleVectorKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationDoubleVectorKey
 * @brief   Key for double vector values.
 *
 * svtkInformationDoubleVectorKey is used to represent keys for double
 * vector values in svtkInformation.h
 */

#ifndef svtkInformationDoubleVectorKey_h
#define svtkInformationDoubleVectorKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class SVTKCOMMONCORE_EXPORT svtkInformationDoubleVectorKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationDoubleVectorKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationDoubleVectorKey(const char* name, const char* location, int length = -1);
  ~svtkInformationDoubleVectorKey() override;

  /**
   * This method simply returns a new svtkInformationDoubleVectorKey, given a
   * name, a location and a required length. This method is provided for
   * wrappers. Use the constructor directly from C++ instead.
   */
  static svtkInformationDoubleVectorKey* MakeKey(
    const char* name, const char* location, int length = -1)
  {
    return new svtkInformationDoubleVectorKey(name, location, length);
  }

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Append(svtkInformation* info, double value);
  void Set(svtkInformation* info, const double* value, int length);
  double* Get(svtkInformation* info);
  double Get(svtkInformation* info, int idx);
  void Get(svtkInformation* info, double* value);
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

private:
  svtkInformationDoubleVectorKey(const svtkInformationDoubleVectorKey&) = delete;
  void operator=(const svtkInformationDoubleVectorKey&) = delete;
};

#endif
