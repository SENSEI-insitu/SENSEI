/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCellArrayIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkCellArrayIterator
 * @brief   Encapsulate traversal logic for svtkCellArray.
 *
 * This is iterator for thread-safe traversal of a svtkCellArray. It provides
 * random access and forward iteration. Typical usage for forward iteration
 * looks like:
 *
 * ```
 * auto iter = svtk::TakeSmartPointer(cellArray->NewIterator());
 * for (iter->GoToFirstCell(); !iter->IsDoneWithTraversal(); iter->GoToNextCell())
 * {
 *   // do work with iter
 *   iter->GetCurrentCell(numCellPts, cellPts);
 * }
 * ```
 *
 * Typical usage for random access looks like:
 *
 * ```
 * auto iter = svtk::TakeSmartPointer(cellArray->NewIterator());
 * iter->GetCellAtId(cellId, numCellPts, cellPts);
 * ```
 *
 * Here @a cellId is the id of the ith cell in the svtkCellArray;
 * @a numCellPts is the number of points defining the cell represented
 * as svtkIdType; and @a cellPts is a pointer to the point ids defined
 * as svtkIdType const*&.
 *
 * Internally the iterator may copy data from the svtkCellArray, or reference
 * the internal svtkCellArray storage. This depends on the relationship of
 * svtkIdType to the type and structure of internal storage. If the type of
 * storage is the same as svtkIdType, and the storage is a single-component
 * AOS array (i.e., a 1D array), then shared access to the svtkCellArray
 * storage is provided. Otherwise, the data from storage is copied into an
 * internal iterator buffer. (Of course copying is slower and can result in
 * 3-4x reduction in traversal performance. On the other hand, the
 * svtkCellArray can use the appropriate storage to save memory, perform
 * zero-copy, and/or efficiently represent the cell connectivity
 * information.) Note that referencing internal svtkCellArray storage has
 * implications on the validity of the iterator. If the underlying
 * svtkCellArray storage changes while iterating, and the iterator is
 * referencing this storage, unpredictable and catastrophic results are
 * likely - hence do not modify the svtkCellArray while iterating.
 *
 * @sa
 * svtkCellArray
 */

#ifndef svtkCellArrayIterator_h
#define svtkCellArrayIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

#include "svtkCellArray.h"    // Needed for inline methods
#include "svtkIdList.h"       // Needed for inline methods
#include "svtkSmartPointer.h" // For svtkSmartPointer

#include <cassert>     // for assert
#include <type_traits> // for std::enable_if

class SVTKCOMMONDATAMODEL_EXPORT svtkCellArrayIterator : public svtkObject
{
public:
  //@{
  /**
   * Standard methods for instantiation, type information, and printing.
   */
  svtkTypeMacro(svtkCellArrayIterator, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  static svtkCellArrayIterator* New();
  //@}

  /**
   * Return the svtkCellArray object over which iteration is occuring.
   */
  svtkCellArray* GetCellArray() { return this->CellArray; }

  /**
   * Intialize the iterator to a specific cell. This will revalidate the
   * iterator if the underlying svtkCellArray has been modified. This method
   * can always be used to set the starting location for forward iteration,
   * and it is also used to support random access.
   */
  void GoToCell(svtkIdType cellId)
  {
    this->CurrentCellId = cellId;
    this->NumberOfCells = this->CellArray->GetNumberOfCells();
    assert(cellId <= this->NumberOfCells);
  }

  /**
   * The following are methods supporting random access iteration.
   */

  //@{
  /**
   * Initialize the iterator to a specific cell and return the cell. Note
   * that methods passing svtkIdLists always copy data from the svtkCellArray
   * storage buffer into the svtkIdList. Otherwise, a fastpath returning
   * (numCellPts,cellPts) which may return a pointer to internal svtkCellArray
   * storage is possible, if svtkIdType is the same as the svtkCellArray buffer
   * (which is typical).
   */
  void GetCellAtId(svtkIdType cellId, svtkIdType& numCellPts, svtkIdType const*& cellPts)
  {
    this->GoToCell(cellId);
    this->GetCurrentCell(numCellPts, cellPts);
  }
  void GetCellAtId(svtkIdType cellId, svtkIdList* cellIds)
  {
    this->GoToCell(cellId);
    this->GetCurrentCell(cellIds);
  }
  svtkIdList* GetCellAtId(svtkIdType cellId)
  {
    this->GoToCell(cellId);
    return this->GetCurrentCell();
  }
  //@}

  /**
   * The following are methods supporting forward iteration.
   */

  /**
   * Initialize the iterator for forward iteration. This will revalidate the
   * iterator if the underlying svtkCellArray has been modified.
   */
  void GoToFirstCell()
  {
    this->CurrentCellId = 0;
    this->NumberOfCells = this->CellArray->GetNumberOfCells();
  }

  /**
   * Advance the forward iterator to the next cell.
   */
  void GoToNextCell() { ++this->CurrentCellId; }

  /**
   * Returns true if the iterator has completed the traversal.
   */
  bool IsDoneWithTraversal() { return this->CurrentCellId >= this->NumberOfCells; }

  /**
   * Returns the id of the current cell during forward iteration.
   */
  svtkIdType GetCurrentCellId() const { return this->CurrentCellId; }

  //@}
  /**
   * Returns the definition of the current cell during forward
   * traversal. Note that methods passing svtkIdLists always copy data from
   * the svtkCellArray storage buffer into the svtkIdList. Otherwise, a
   * fastpath returning (numCellPts,cellPts) - which may return a pointer to
   * internal svtkCellArray storage - is possible, if svtkIdType is the same as
   * the svtkCellArray storage (which is typical).
   */
  void GetCurrentCell(svtkIdType& cellSize, svtkIdType const*& cellPoints)
  {
    assert(this->CurrentCellId < this->NumberOfCells);
    // Either refer to svtkCellArray storage buffer, or copy into local buffer
    if (this->CellArray->IsStorageShareable())
    {
      this->CellArray->GetCellAtId(this->CurrentCellId, cellSize, cellPoints);
    }
    else // or copy into local iterator buffer.
    {
      this->CellArray->GetCellAtId(this->CurrentCellId, this->TempCell);
      cellSize = this->TempCell->GetNumberOfIds();
      cellPoints = this->TempCell->GetPointer(0);
    }
  }
  void GetCurrentCell(svtkIdList* ids)
  {
    assert(this->CurrentCellId < this->NumberOfCells);
    this->CellArray->GetCellAtId(this->CurrentCellId, ids);
  }
  svtkIdList* GetCurrentCell()
  {
    assert(this->CurrentCellId < this->NumberOfCells);
    this->CellArray->GetCellAtId(this->CurrentCellId, this->TempCell);
    return this->TempCell;
  }
  //@}

  /**
   * Specialized methods for performing operations on the svtkCellArray.
   */

  /**
   * Replace the current cell with the ids in `list`. Note that this method
   * CANNOT change the number of points in the cell, it can only redefine the
   * ids (e.g. `list` must contain the same number of entries as the current
   * cell's points).
   */
  void ReplaceCurrentCell(svtkIdList* list)
  {
    assert(this->CurrentCellId < this->NumberOfCells);
    this->CellArray->ReplaceCellAtId(this->CurrentCellId, list);
  }

  /**
   * Replace the current cell with the ids in `pts`. Note that this method
   * CANNOT change the number of points in the cell, it can only redefine the
   * ids (e.g. `npts` must equal the current cell's number of points).
   */
  void ReplaceCurrentCell(svtkIdType npts, const svtkIdType* pts)
  {
    assert(this->CurrentCellId < this->NumberOfCells);
    this->CellArray->ReplaceCellAtId(this->CurrentCellId, npts, pts);
  }

  /**
   * Reverses the order of the point ids in the current cell.
   */
  void ReverseCurrentCell()
  {
    assert(this->CurrentCellId < this->NumberOfCells);
    this->CellArray->ReverseCellAtId(this->CurrentCellId);
  }

  friend class svtkCellArray;

protected:
  svtkCellArrayIterator() = default;
  ~svtkCellArrayIterator() override = default;

  svtkSetMacro(CellArray, svtkCellArray*);

  svtkSmartPointer<svtkCellArray> CellArray;
  svtkNew<svtkIdList> TempCell;
  svtkIdType CurrentCellId;
  svtkIdType NumberOfCells;

private:
  svtkCellArrayIterator(const svtkCellArrayIterator&) = delete;
  void operator=(const svtkCellArrayIterator&) = delete;
};

#endif // svtkCellArrayIterator_h
