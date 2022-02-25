/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkOverrideInformation.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkOverrideInformation
 * @brief   Factory object override information
 *
 * svtkOverrideInformation is used to represent the information about
 * a class which is overridden in a svtkObjectFactory.
 *
 */

#ifndef svtkOverrideInformation_h
#define svtkOverrideInformation_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

class svtkObjectFactory;

class SVTKCOMMONCORE_EXPORT svtkOverrideInformation : public svtkObject
{
public:
  static svtkOverrideInformation* New();
  svtkTypeMacro(svtkOverrideInformation, svtkObject);
  /**
   * Print ObjectFactor to stream.
   */
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Returns the name of the class being overridden.  For example,
   * if you had a factory that provided an override for
   * svtkVertex, then this function would return "svtkVertex"
   */
  const char* GetClassOverrideName() { return this->ClassOverrideName; }

  /**
   * Returns the name of the class that will override the class.
   * For example, if you had a factory that provided an override for
   * svtkVertex called svtkMyVertex, then this would return "svtkMyVertex"
   */
  const char* GetClassOverrideWithName() { return this->ClassOverrideWithName; }

  /**
   * Return a human readable or GUI displayable description of this
   * override.
   */
  const char* GetDescription() { return this->Description; }

  /**
   * Return the specific object factory that this override occurs in.
   */
  svtkObjectFactory* GetObjectFactory() { return this->ObjectFactory; }

  //@{
  /**
   * Set the class override name
   */
  svtkSetStringMacro(ClassOverrideName);

  /**
   * Set the class override with name
   */
  svtkSetStringMacro(ClassOverrideWithName);

  /**
   * Set the description
   */
  svtkSetStringMacro(Description);
  //@}

protected:
  virtual void SetObjectFactory(svtkObjectFactory*);

private:
  svtkOverrideInformation();
  ~svtkOverrideInformation() override;
  // allow the object factory to set the values in this
  // class, but only the object factory

  friend class svtkObjectFactory;

  char* ClassOverrideName;
  char* ClassOverrideWithName;
  char* Description;
  svtkObjectFactory* ObjectFactory;

private:
  svtkOverrideInformation(const svtkOverrideInformation&) = delete;
  void operator=(const svtkOverrideInformation&) = delete;
};

#endif
