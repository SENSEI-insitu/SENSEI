/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericCellIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkGenericCellIterator
 * @brief   iterator used to traverse cells
 *
 * This class (and subclasses) are used to iterate over cells. Use it
 * only in conjunction with svtkGenericDataSet (i.e., the adaptor framework).
 *
 * Typical use is:
 * <pre>
 * svtkGenericDataSet *dataset;
 * svtkGenericCellIterator *it = dataset->NewCellIterator(2);
 * for (it->Begin(); !it->IsAtEnd(); it->Next());
 *   {
 *   spec=it->GetCell();
 *   }
 * </pre>
 */

#ifndef svtkGenericCellIterator_h
#define svtkGenericCellIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkGenericAdaptorCell;

class SVTKCOMMONDATAMODEL_EXPORT svtkGenericCellIterator : public svtkObject
{
public:
  //@{
  /**
   * Standard SVTK construction and type macros.
   */
  svtkTypeMacro(svtkGenericCellIterator, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Move iterator to first position if any (loop initialization).
   */
  virtual void Begin() = 0;

  /**
   * Is the iterator at the end of traversal?
   */
  virtual svtkTypeBool IsAtEnd() = 0;

  /**
   * Create an empty cell. The user is responsible for deleting it.
   * \post result_exists: result!=0
   */
  virtual svtkGenericAdaptorCell* NewCell() = 0;

  /**
   * Get the cell at current position. The cell should be instantiated
   * with the NewCell() method.
   * \pre not_at_end: !IsAtEnd()
   * \pre c_exists: c!=0
   * THREAD SAFE
   */
  virtual void GetCell(svtkGenericAdaptorCell* c) = 0;

  /**
   * Get the cell at the current traversal position.
   * NOT THREAD SAFE
   * \pre not_at_end: !IsAtEnd()
   * \post result_exits: result!=0
   */
  virtual svtkGenericAdaptorCell* GetCell() = 0;

  /**
   * Move the iterator to the next position in the list.
   * \pre not_at_end: !IsAtEnd()
   */
  virtual void Next() = 0;

protected:
  svtkGenericCellIterator();
  ~svtkGenericCellIterator() override;

private:
  svtkGenericCellIterator(const svtkGenericCellIterator&) = delete;
  void operator=(const svtkGenericCellIterator&) = delete;
};

#endif
