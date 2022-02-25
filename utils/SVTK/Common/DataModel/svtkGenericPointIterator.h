/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericPointIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkGenericPointIterator
 * @brief   iterator used to traverse points
 *
 * This class (and subclasses) are used to iterate over points. Use it
 * only in conjunction with svtkGenericDataSet (i.e., the adaptor framework).
 *
 * Typical use is:
 * <pre>
 * svtkGenericDataSet *dataset;
 * svtkGenericPointIterator *it = dataset->NewPointIterator();
 * for (it->Begin(); !it->IsAtEnd(); it->Next());
 *   {
 *   x=it->GetPosition();
 *   }
 * </pre>
 */

#ifndef svtkGenericPointIterator_h
#define svtkGenericPointIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkGenericPointIterator : public svtkObject
{
public:
  //@{
  /**
   * Standard SVTK construction and type macros.
   */
  svtkTypeMacro(svtkGenericPointIterator, svtkObject);
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
   * Move the iterator to the next position in the list.
   * \pre not_off: !IsAtEnd()
   */
  virtual void Next() = 0;

  /**
   * Get the coordinates of the point at the current iterator position.
   * \pre not_off: !IsAtEnd()
   * \post result_exists: result!=0
   */
  virtual double* GetPosition() = 0;

  /**
   * Get the coordinates of the point at the current iterator position.
   * \pre not_off: !IsAtEnd()
   * \pre x_exists: x!=0
   */
  virtual void GetPosition(double x[3]) = 0;

  /**
   * Return the unique identifier for the point, could be non-contiguous.
   * \pre not_off: !IsAtEnd()
   */
  virtual svtkIdType GetId() = 0;

protected:
  //@{
  /**
   * Destructor.
   */
  svtkGenericPointIterator();
  ~svtkGenericPointIterator() override;
  //@}

private:
  svtkGenericPointIterator(const svtkGenericPointIterator&) = delete;
  void operator=(const svtkGenericPointIterator&) = delete;
};

#endif
