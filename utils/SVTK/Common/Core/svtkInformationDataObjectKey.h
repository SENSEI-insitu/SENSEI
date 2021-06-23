/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationDataObjectKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationDataObjectKey
 * @brief   Key for svtkDataObject values.
 *
 * svtkInformationDataObjectKey is used to represent keys in
 * svtkInformation for values that are svtkDataObject instances.
 */

#ifndef svtkInformationDataObjectKey_h
#define svtkInformationDataObjectKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class svtkDataObject;

class SVTKCOMMONCORE_EXPORT svtkInformationDataObjectKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationDataObjectKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationDataObjectKey(const char* name, const char* location);
  ~svtkInformationDataObjectKey() override;

  /**
   * This method simply returns a new svtkInformationDataObjectKey, given a
   * name and a location. This method is provided for wrappers. Use the
   * constructor directly from C++ instead.
   */
  static svtkInformationDataObjectKey* MakeKey(const char* name, const char* location)
  {
    return new svtkInformationDataObjectKey(name, location);
  }

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Set(svtkInformation* info, svtkDataObject*);
  svtkDataObject* Get(svtkInformation* info);
  //@}

  /**
   * Copy the entry associated with this key from one information
   * object to another.  If there is no entry in the first information
   * object for this key, the value is removed from the second.
   */
  void ShallowCopy(svtkInformation* from, svtkInformation* to) override;

  /**
   * Report a reference this key has in the given information object.
   */
  void Report(svtkInformation* info, svtkGarbageCollector* collector) override;

private:
  svtkInformationDataObjectKey(const svtkInformationDataObjectKey&) = delete;
  void operator=(const svtkInformationDataObjectKey&) = delete;
};

#endif
