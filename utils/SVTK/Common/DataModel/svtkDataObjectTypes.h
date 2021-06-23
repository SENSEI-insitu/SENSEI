/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataObjectTypes.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDataObject
 * @brief   helper class to get SVTK data object types as string and instantiate them
 *
 * svtkDataObjectTypes is a helper class that supports conversion between
 * integer types defined in svtkType.h and string names as well as creation
 * of data objects from either integer or string types. This class has
 * to be updated every time a new data type is added to SVTK.
 * @sa
 * svtkDataObject
 */

#ifndef svtkDataObjectTypes_h
#define svtkDataObjectTypes_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkDataObject;

class SVTKCOMMONDATAMODEL_EXPORT svtkDataObjectTypes : public svtkObject
{
public:
  static svtkDataObjectTypes* New();

  svtkTypeMacro(svtkDataObjectTypes, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Given an int (as defined in svtkType.h) identifier for a class
   * return it's classname.
   */
  static const char* GetClassNameFromTypeId(int typeId);

  /**
   * Given a data object classname, return it's int identified (as
   * defined in svtkType.h)
   */
  static int GetTypeIdFromClassName(const char* classname);

  /**
   * Create (New) and return a data object of the given classname.
   */
  static svtkDataObject* NewDataObject(const char* classname);

  /**
   * Create (New) and return a data object of the given type id.
   */
  static svtkDataObject* NewDataObject(int typeId);

protected:
  svtkDataObjectTypes() {}
  ~svtkDataObjectTypes() override {}

  /**
   * Method used to validate data object types, for testing purposes
   */
  static int Validate();

private:
  svtkDataObjectTypes(const svtkDataObjectTypes&) = delete;
  void operator=(const svtkDataObjectTypes&) = delete;
};

#endif
