/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMappedUnstructuredGridCellIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkMappedUnstructuredGridCellIterator
 * @brief   Default cell iterator for
 * svtkMappedUnstructuredGrid.
 *
 *
 * This class is used by default for svtkMappedUnstructedGrid instances. It
 * uses random access for data lookups. Custom svtkCellIterator implementations
 * should be used instead when random-access is inefficient.
 */

#ifndef svtkMappedUnstructuredGridCellIterator_h
#define svtkMappedUnstructuredGridCellIterator_h

#include "svtkCellIterator.h"
#include "svtkSmartPointer.h" // For svtkSmartPointer

template <class Implementation, class CellIterator>
class svtkMappedUnstructuredGrid;

template <class Implementation>
class svtkMappedUnstructuredGridCellIterator : public svtkCellIterator
{
public:
  svtkTemplateTypeMacro(svtkMappedUnstructuredGridCellIterator<Implementation>,
    svtkCellIterator) typedef Implementation ImplementationType;
  typedef svtkMappedUnstructuredGridCellIterator<ImplementationType> ThisType;
  static svtkMappedUnstructuredGridCellIterator<ImplementationType>* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  void SetMappedUnstructuredGrid(svtkMappedUnstructuredGrid<ImplementationType, ThisType>* grid);

  bool IsDoneWithTraversal() override;
  svtkIdType GetCellId() override;

protected:
  svtkMappedUnstructuredGridCellIterator();
  ~svtkMappedUnstructuredGridCellIterator() override;

  void ResetToFirstCell() override;
  void IncrementToNextCell() override;
  void FetchCellType() override;
  void FetchPointIds() override;
  void FetchPoints() override;

private:
  svtkMappedUnstructuredGridCellIterator(const svtkMappedUnstructuredGridCellIterator&) = delete;
  void operator=(const svtkMappedUnstructuredGridCellIterator&) = delete;

  svtkSmartPointer<ImplementationType> Impl;
  svtkSmartPointer<svtkPoints> GridPoints;
  svtkIdType CellId;
  svtkIdType NumberOfCells;
};

#include "svtkMappedUnstructuredGridCellIterator.txx"

#endif // svtkMappedUnstructuredGridCellIterator_h

// SVTK-HeaderTest-Exclude: svtkMappedUnstructuredGridCellIterator.h
