/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnstructuredGridCellIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkUnstructuredGridCellIterator
 * @brief   Implementation of svtkCellIterator
 * specialized for svtkUnstructuredGrid.
 */

#ifndef svtkUnstructuredGridCellIterator_h
#define svtkUnstructuredGridCellIterator_h

#include "svtkCellIterator.h"
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkSmartPointer.h"          // For svtkSmartPointer

class svtkCellArray;
class svtkCellArrayIterator;
class svtkIdTypeArray;
class svtkUnsignedCharArray;
class svtkUnstructuredGrid;
class svtkPoints;

class SVTKCOMMONDATAMODEL_EXPORT svtkUnstructuredGridCellIterator : public svtkCellIterator
{
public:
  static svtkUnstructuredGridCellIterator* New();
  svtkTypeMacro(svtkUnstructuredGridCellIterator, svtkCellIterator);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  bool IsDoneWithTraversal() override;
  svtkIdType GetCellId() override;

protected:
  svtkUnstructuredGridCellIterator();
  ~svtkUnstructuredGridCellIterator() override;

  void ResetToFirstCell() override;
  void IncrementToNextCell() override;
  void FetchCellType() override;
  void FetchPointIds() override;
  void FetchPoints() override;
  void FetchFaces() override;

  friend class svtkUnstructuredGrid;
  void SetUnstructuredGrid(svtkUnstructuredGrid* ug);

  svtkSmartPointer<svtkCellArrayIterator> Cells;
  svtkSmartPointer<svtkUnsignedCharArray> Types;
  svtkSmartPointer<svtkIdTypeArray> FaceConn;
  svtkSmartPointer<svtkIdTypeArray> FaceLocs;
  svtkSmartPointer<svtkPoints> Coords;

private:
  svtkUnstructuredGridCellIterator(const svtkUnstructuredGridCellIterator&) = delete;
  void operator=(const svtkUnstructuredGridCellIterator&) = delete;
};

#endif // svtkUnstructuredGridCellIterator_h
