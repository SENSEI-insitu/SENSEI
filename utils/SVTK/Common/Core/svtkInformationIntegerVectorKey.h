/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationIntegerVectorKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationIntegerVectorKey
 * @brief   Key for integer vector values.
 *
 * svtkInformationIntegerVectorKey is used to represent keys for integer
 * vector values in svtkInformation.h
 */

#ifndef svtkInformationIntegerVectorKey_h
#define svtkInformationIntegerVectorKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class SVTKCOMMONCORE_EXPORT svtkInformationIntegerVectorKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationIntegerVectorKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationIntegerVectorKey(const char* name, const char* location, int length = -1);
  ~svtkInformationIntegerVectorKey() override;

  /**
   * This method simply returns a new svtkInformationIntegerVectorKey, given a
   * name, a location and a required length. This method is provided for
   * wrappers. Use the constructor directly from C++ instead.
   */
  static svtkInformationIntegerVectorKey* MakeKey(
    const char* name, const char* location, int length = -1)
  {
    return new svtkInformationIntegerVectorKey(name, location, length);
  }

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Append(svtkInformation* info, int value);
  void Set(svtkInformation* info, const int* value, int length);
  void Set(svtkInformation* info);
  int* Get(svtkInformation* info);
  int Get(svtkInformation* info, int idx);
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
  svtkInformationIntegerVectorKey(const svtkInformationIntegerVectorKey&) = delete;
  void operator=(const svtkInformationIntegerVectorKey&) = delete;
};

#endif
