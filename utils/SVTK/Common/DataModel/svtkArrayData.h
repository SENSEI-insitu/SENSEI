/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayData.h

-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkArrayData
 * @brief   Pipeline data object that contains multiple svtkArray objects.
 *
 *
 * Because svtkArray cannot be stored as attributes of data objects (yet), a "carrier"
 * object is needed to pass svtkArray through the pipeline.  svtkArrayData acts as a
 * container of zero-to-many svtkArray instances, which can be retrieved via a zero-based
 * index.  Note that a collection of arrays stored in svtkArrayData may-or-may-not have related
 * types, dimensions, or extents.
 *
 * @sa
 * svtkArrayDataAlgorithm, svtkArray
 *
 * @par Thanks:
 * Developed by Timothy M. Shead (tshead@sandia.gov) at Sandia National Laboratories.
 */

#ifndef svtkArrayData_h
#define svtkArrayData_h

#include "svtkArray.h"
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataObject.h"

class svtkArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkArrayData : public svtkDataObject
{
public:
  static svtkArrayData* New();
  svtkTypeMacro(svtkArrayData, svtkDataObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  static svtkArrayData* GetData(svtkInformation* info);
  static svtkArrayData* GetData(svtkInformationVector* v, int i = 0);

  /**
   * Adds a svtkArray to the collection
   */
  void AddArray(svtkArray*);

  /**
   * Clears the contents of the collection
   */
  void ClearArrays();

  /**
   * Returns the number of svtkArray instances in the collection
   */
  svtkIdType GetNumberOfArrays();

  /**
   * Returns the n-th svtkArray in the collection
   */
  svtkArray* GetArray(svtkIdType index);

  /**
   * Returns the array having called name from the collection
   */
  svtkArray* GetArrayByName(const char* name);

  /**
   * Return class name of data type (SVTK_ARRAY_DATA).
   */
  int GetDataObjectType() override { return SVTK_ARRAY_DATA; }

  void ShallowCopy(svtkDataObject* other) override;
  void DeepCopy(svtkDataObject* other) override;

protected:
  svtkArrayData();
  ~svtkArrayData() override;

private:
  svtkArrayData(const svtkArrayData&) = delete;
  void operator=(const svtkArrayData&) = delete;

  class implementation;
  implementation* const Implementation;
};

#endif

// SVTK-HeaderTest-Exclude: svtkArrayData.h
