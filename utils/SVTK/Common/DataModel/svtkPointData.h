/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPointData.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPointData
 * @brief   represent and manipulate point attribute data
 *
 * svtkPointData is a class that is used to represent and manipulate
 * point attribute data (e.g., scalars, vectors, normals, texture
 * coordinates, etc.) Most of the functionality is handled by
 * svtkDataSetAttributes
 */

#ifndef svtkPointData_h
#define svtkPointData_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataSetAttributes.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkPointData : public svtkDataSetAttributes
{
public:
  static svtkPointData* New();

  svtkTypeMacro(svtkPointData, svtkDataSetAttributes);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  void NullPoint(svtkIdType ptId);

protected:
  svtkPointData() {}
  ~svtkPointData() override {}

private:
  svtkPointData(const svtkPointData&) = delete;
  void operator=(const svtkPointData&) = delete;
};

#endif
