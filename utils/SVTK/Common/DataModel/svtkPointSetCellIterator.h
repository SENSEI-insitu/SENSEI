/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPointSetCellIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPointSetCellIterator
 * @brief   Implementation of svtkCellIterator using
 * svtkPointSet API.
 */

#ifndef svtkPointSetCellIterator_h
#define svtkPointSetCellIterator_h

#include "svtkCellIterator.h"
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkSmartPointer.h"          // For svtkSmartPointer

class svtkPoints;
class svtkPointSet;

class SVTKCOMMONDATAMODEL_EXPORT svtkPointSetCellIterator : public svtkCellIterator
{
public:
  static svtkPointSetCellIterator* New();
  svtkTypeMacro(svtkPointSetCellIterator, svtkCellIterator);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  bool IsDoneWithTraversal() override;
  svtkIdType GetCellId() override;

protected:
  svtkPointSetCellIterator();
  ~svtkPointSetCellIterator() override;

  void ResetToFirstCell() override;
  void IncrementToNextCell() override;
  void FetchCellType() override;
  void FetchPointIds() override;
  void FetchPoints() override;

  friend class svtkPointSet;
  void SetPointSet(svtkPointSet* ds);

  svtkSmartPointer<svtkPointSet> PointSet;
  svtkSmartPointer<svtkPoints> PointSetPoints;
  svtkIdType CellId;

private:
  svtkPointSetCellIterator(const svtkPointSetCellIterator&) = delete;
  void operator=(const svtkPointSetCellIterator&) = delete;
};

#endif // svtkPointSetCellIterator_h
