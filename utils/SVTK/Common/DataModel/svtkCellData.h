/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCellData.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCellData
 * @brief   represent and manipulate cell attribute data
 *
 * svtkCellData is a class that is used to represent and manipulate
 * cell attribute data (e.g., scalars, vectors, normals, texture
 * coordinates, etc.) Special methods are provided to work with filter
 * objects, such as passing data through filter, copying data from one
 * cell to another, and interpolating data given cell interpolation weights.
 */

#ifndef svtkCellData_h
#define svtkCellData_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataSetAttributes.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkCellData : public svtkDataSetAttributes
{
public:
  static svtkCellData* New();

  svtkTypeMacro(svtkCellData, svtkDataSetAttributes);
  void PrintSelf(ostream& os, svtkIndent indent) override;

protected:
  svtkCellData() {} // make sure constructor and destructor are protected
  ~svtkCellData() override {}

private:
  svtkCellData(const svtkCellData&) = delete;
  void operator=(const svtkCellData&) = delete;
};

#endif
