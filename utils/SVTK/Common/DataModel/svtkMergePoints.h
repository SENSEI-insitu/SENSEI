/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMergePoints.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkMergePoints
 * @brief   merge exactly coincident points
 *
 * svtkMergePoints is a locator object to quickly locate points in 3D.
 * The primary difference between svtkMergePoints and its superclass
 * svtkPointLocator is that svtkMergePoints merges precisely coincident points
 * and is therefore much faster.
 * @sa
 * svtkCleanPolyData
 */

#ifndef svtkMergePoints_h
#define svtkMergePoints_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkPointLocator.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkMergePoints : public svtkPointLocator
{
public:
  static svtkMergePoints* New();
  svtkTypeMacro(svtkMergePoints, svtkPointLocator);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Determine whether point given by x[3] has been inserted into points list.
   * Return id of previously inserted point if this is true, otherwise return
   * -1.
   */
  svtkIdType IsInsertedPoint(const double x[3]) override;
  svtkIdType IsInsertedPoint(double x, double y, double z) override
  {
    return this->svtkPointLocator::IsInsertedPoint(x, y, z);
  }
  //@}

  /**
   * Determine whether point given by x[3] has been inserted into points list.
   * Return 0 if point was already in the list, otherwise return 1. If the
   * point was not in the list, it will be ADDED.  In either case, the id of
   * the point (newly inserted or not) is returned in the ptId argument.
   * Note this combines the functionality of IsInsertedPoint() followed
   * by a call to InsertNextPoint().
   */
  int InsertUniquePoint(const double x[3], svtkIdType& ptId) override;

protected:
  svtkMergePoints() {}
  ~svtkMergePoints() override {}

private:
  svtkMergePoints(const svtkMergePoints&) = delete;
  void operator=(const svtkMergePoints&) = delete;
};

#endif
