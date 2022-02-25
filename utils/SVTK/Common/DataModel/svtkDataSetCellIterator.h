/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataSetCellIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDataSetCellIterator
 * @brief   Implementation of svtkCellIterator using
 * svtkDataSet API.
 */

#ifndef svtkDataSetCellIterator_h
#define svtkDataSetCellIterator_h

#include "svtkCellIterator.h"
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkSmartPointer.h"          // For svtkSmartPointer

class svtkDataSet;

class SVTKCOMMONDATAMODEL_EXPORT svtkDataSetCellIterator : public svtkCellIterator
{
public:
  static svtkDataSetCellIterator* New();
  svtkTypeMacro(svtkDataSetCellIterator, svtkCellIterator);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  bool IsDoneWithTraversal() override;
  svtkIdType GetCellId() override;

protected:
  svtkDataSetCellIterator();
  ~svtkDataSetCellIterator() override;

  void ResetToFirstCell() override;
  void IncrementToNextCell() override;
  void FetchCellType() override;
  void FetchPointIds() override;
  void FetchPoints() override;

  friend class svtkDataSet;
  void SetDataSet(svtkDataSet* ds);

  svtkSmartPointer<svtkDataSet> DataSet;
  svtkIdType CellId;

private:
  svtkDataSetCellIterator(const svtkDataSetCellIterator&) = delete;
  void operator=(const svtkDataSetCellIterator&) = delete;
};

#endif // svtkDataSetCellIterator_h
