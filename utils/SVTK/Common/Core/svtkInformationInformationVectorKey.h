/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationInformationVectorKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationInformationVectorKey
 * @brief   Key for svtkInformation vectors.
 *
 * svtkInformationInformationVectorKey is used to represent keys in
 * svtkInformation for vectors of other svtkInformation objects.
 */

#ifndef svtkInformationInformationVectorKey_h
#define svtkInformationInformationVectorKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkInformationKey.h"

#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.

class svtkInformationVector;

class SVTKCOMMONCORE_EXPORT svtkInformationInformationVectorKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationInformationVectorKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkInformationInformationVectorKey(const char* name, const char* location);
  ~svtkInformationInformationVectorKey() override;

  //@{
  /**
   * Get/Set the value associated with this key in the given
   * information object.
   */
  void Set(svtkInformation* info, svtkInformationVector*);
  svtkInformationVector* Get(svtkInformation* info);
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

  /**
   * Report a reference this key has in the given information object.
   */
  void Report(svtkInformation* info, svtkGarbageCollector* collector) override;

private:
  svtkInformationInformationVectorKey(const svtkInformationInformationVectorKey&) = delete;
  void operator=(const svtkInformationInformationVectorKey&) = delete;
};

#endif
