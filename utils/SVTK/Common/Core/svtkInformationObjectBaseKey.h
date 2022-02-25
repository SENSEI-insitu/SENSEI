/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationObjectBaseKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationObjectBaseKey
 * @brief   Key for svtkObjectBase values.
 *
 * svtkInformationObjectBaseKey is used to represent keys in
 * svtkInformation for values that are svtkObjectBase instances.
 */

#ifndef svtkInformationObjectBaseKey_h
#define svtkInformationObjectBaseKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class svtkObjectBase;

class SVTKCOMMONCORE_EXPORT svtkInformationObjectBaseKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationObjectBaseKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationObjectBaseKey(
    const char* name, const char* location, const char* requiredClass = nullptr);
  ~svtkInformationObjectBaseKey() override;

  /**
   * This method simply returns a new svtkInformationObjectBaseKey, given a
   * name, location and optionally a required class (a classname to restrict
   * which class types can be set with this key). This method is provided
   * for wrappers. Use the constructor directly from C++ instead.
   */
  static svtkInformationObjectBaseKey* MakeKey(
    const char* name, const char* location, const char* requiredClass = nullptr)
  {
    return new svtkInformationObjectBaseKey(name, location, requiredClass);
  }

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Set(svtkInformation* info, svtkObjectBase*);
  svtkObjectBase* Get(svtkInformation* info);
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

protected:
  // The type required of all objects stored with this key.
  const char* RequiredClass;

  svtkInformationKeySetStringMacro(RequiredClass);

private:
  svtkInformationObjectBaseKey(const svtkInformationObjectBaseKey&) = delete;
  void operator=(const svtkInformationObjectBaseKey&) = delete;
};

#endif
