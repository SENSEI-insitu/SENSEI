/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCellLinks.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCellLinks
 * @brief   object represents upward pointers from points to list of cells using each point
 *
 * svtkCellLinks is a supplemental object to svtkCellArray and svtkCellTypes,
 * enabling access from points to the cells using the points. svtkCellLinks is
 * a list of cell ids, each such link representing a dynamic list of cell ids
 * using the point. The information provided by this object can be used to
 * determine neighbors and construct other local topological information.
 *
 * @warning
 * svtkCellLinks supports incremental (i.e., "editable") operations such as
 * inserting a new cell, or deleting a point. Because of this, it is less
 * memory efficient, and slower to construct and delete than static classes
 * such as svtkStaticCellLinks or svtkStaticCellLinksTemplate. However these
 * other classes are typically meant for one-time (static) construction.
 *
 * @sa
 * svtkCellArray svtkCellTypes svtkStaticCellLinks svtkStaticCellLinksTemplate
 */

#ifndef svtkCellLinks_h
#define svtkCellLinks_h

#include "svtkAbstractCellLinks.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkDataSet;
class svtkCellArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkCellLinks : public svtkAbstractCellLinks
{
public:
  class Link
  {
  public:
    svtkIdType ncells;
    svtkIdType* cells;
  };

  //@{
  /**
   * Standard methods to instantiate, print, and obtain type information.
   */
  static svtkCellLinks* New();
  svtkTypeMacro(svtkCellLinks, svtkAbstractCellLinks);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Build the link list array. All subclasses of svtkAbstractCellLinks
   * must support this method.
   */
  void BuildLinks(svtkDataSet* data) override;

  /**
   * Allocate the specified number of links (i.e., number of points) that
   * will be built.
   */
  void Allocate(svtkIdType numLinks, svtkIdType ext = 1000);

  /**
   * Clear out any previously allocated data structures
   */
  void Initialize() override;

  /**
   * Get a link structure given a point id.
   */
  Link& GetLink(svtkIdType ptId) { return this->Array[ptId]; }

  /**
   * Get the number of cells using the point specified by ptId.
   */
  svtkIdType GetNcells(svtkIdType ptId) { return this->Array[ptId].ncells; }

  /**
   * Return a list of cell ids using the point.
   */
  svtkIdType* GetCells(svtkIdType ptId) { return this->Array[ptId].cells; }

  /**
   * Insert a new point into the cell-links data structure. The size parameter
   * is the initial size of the list.
   */
  svtkIdType InsertNextPoint(int numLinks);

  /**
   * Insert a cell id into the list of cells (at the end) using the cell id
   * provided. (Make sure to extend the link list (if necessary) using the
   * method ResizeCellList().)
   */
  void InsertNextCellReference(svtkIdType ptId, svtkIdType cellId);

  /**
   * Delete point (and storage) by destroying links to using cells.
   */
  void DeletePoint(svtkIdType ptId);

  /**
   * Delete the reference to the cell (cellId) from the point (ptId). This
   * removes the reference to the cellId from the cell list, but does not
   * resize the list (recover memory with ResizeCellList(), if necessary).
   */
  void RemoveCellReference(svtkIdType cellId, svtkIdType ptId);

  /**
   * Add the reference to the cell (cellId) from the point (ptId). This
   * adds a reference to the cellId from the cell list, but does not resize
   * the list (extend memory with ResizeCellList(), if necessary).
   */
  void AddCellReference(svtkIdType cellId, svtkIdType ptId);

  /**
   * Change the length of a point's link list (i.e., list of cells using a
   * point) by the size specified.
   */
  void ResizeCellList(svtkIdType ptId, int size);

  /**
   * Reclaim any unused memory.
   */
  void Squeeze() override;

  /**
   * Reset to a state of no entries without freeing the memory.
   */
  void Reset() override;

  /**
   * Return the memory in kibibytes (1024 bytes) consumed by this cell links array.
   * Used to support streaming and reading/writing data. The value
   * returned is guaranteed to be greater than or equal to the memory
   * required to actually represent the data represented by this object.
   * The information returned is valid only after the pipeline has
   * been updated.
   */
  unsigned long GetActualMemorySize() override;

  /**
   * Standard DeepCopy method.  Since this object contains no reference
   * to other objects, there is no ShallowCopy.
   */
  void DeepCopy(svtkAbstractCellLinks* src) override;

protected:
  svtkCellLinks()
    : Array(nullptr)
    , Size(0)
    , MaxId(-1)
    , Extend(1000)
  {
  }
  ~svtkCellLinks() override;

  /**
   * Increment the count of the number of cells using the point.
   */
  void IncrementLinkCount(svtkIdType ptId) { this->Array[ptId].ncells++; }

  void AllocateLinks(svtkIdType n);

  /**
   * Insert a cell id into the list of cells using the point.
   */
  void InsertCellReference(svtkIdType ptId, svtkIdType pos, svtkIdType cellId);

  Link* Array;                // pointer to data
  svtkIdType Size;             // allocated size of data
  svtkIdType MaxId;            // maximum index inserted thus far
  svtkIdType Extend;           // grow array by this point
  Link* Resize(svtkIdType sz); // function to resize data

private:
  svtkCellLinks(const svtkCellLinks&) = delete;
  void operator=(const svtkCellLinks&) = delete;
};

//----------------------------------------------------------------------------
inline void svtkCellLinks::InsertCellReference(svtkIdType ptId, svtkIdType pos, svtkIdType cellId)
{
  this->Array[ptId].cells[pos] = cellId;
}

//----------------------------------------------------------------------------
inline void svtkCellLinks::DeletePoint(svtkIdType ptId)
{
  this->Array[ptId].ncells = 0;
  delete[] this->Array[ptId].cells;
  this->Array[ptId].cells = nullptr;
}

//----------------------------------------------------------------------------
inline void svtkCellLinks::InsertNextCellReference(svtkIdType ptId, svtkIdType cellId)
{
  this->Array[ptId].cells[this->Array[ptId].ncells++] = cellId;
}

//----------------------------------------------------------------------------
inline void svtkCellLinks::RemoveCellReference(svtkIdType cellId, svtkIdType ptId)
{
  svtkIdType* cells = this->Array[ptId].cells;
  svtkIdType ncells = this->Array[ptId].ncells;

  for (svtkIdType i = 0; i < ncells; i++)
  {
    if (cells[i] == cellId)
    {
      for (svtkIdType j = i; j < (ncells - 1); j++)
      {
        cells[j] = cells[j + 1];
      }
      this->Array[ptId].ncells--;
      break;
    }
  }
}

//----------------------------------------------------------------------------
inline void svtkCellLinks::AddCellReference(svtkIdType cellId, svtkIdType ptId)
{
  this->Array[ptId].cells[this->Array[ptId].ncells++] = cellId;
}

//----------------------------------------------------------------------------
inline void svtkCellLinks::ResizeCellList(svtkIdType ptId, int size)
{
  svtkIdType newSize = this->Array[ptId].ncells + size;
  svtkIdType* cells = new svtkIdType[newSize];
  memcpy(cells, this->Array[ptId].cells,
    static_cast<size_t>(this->Array[ptId].ncells) * sizeof(svtkIdType));
  delete[] this->Array[ptId].cells;
  this->Array[ptId].cells = cells;
}

#endif
