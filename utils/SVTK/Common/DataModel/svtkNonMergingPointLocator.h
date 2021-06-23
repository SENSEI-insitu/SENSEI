/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkNonMergingPointLocator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkNonMergingPointLocator
 * @brief   direct / check-free point insertion.
 *
 *
 *  As a special sub-class of svtkPointLocator, svtkNonMergingPointLocator is
 *  intended for direct / check-free insertion of points into a svtkPoints
 *  object. In other words, any given point is always directly inserted.
 *  The name emphasizes the difference between this class and its sibling
 *  class svtkMergePoints in that the latter class performs check-based zero
 *  tolerance point insertion (or to 'merge' exactly duplicate / coincident
 *  points) by exploiting the uniform bin mechanism employed by the parent
 *  class svtkPointLocator. svtkPointLocator allows for generic (zero and non-
 *  zero) tolerance point insertion as well as point location.
 *
 * @sa
 *  svtkIncrementalPointLocator svtkPointLocator svtkMergePoints
 */

#ifndef svtkNonMergingPointLocator_h
#define svtkNonMergingPointLocator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkPointLocator.h"

class svtkPoints;

class SVTKCOMMONDATAMODEL_EXPORT svtkNonMergingPointLocator : public svtkPointLocator
{
public:
  static svtkNonMergingPointLocator* New();

  svtkTypeMacro(svtkNonMergingPointLocator, svtkPointLocator);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Determine whether a given point x has been inserted into the points list.
   * Return the id of the already inserted point if it is true, or -1 else.
   * Note this function always returns -1 since any point is always inserted.
   */
  svtkIdType IsInsertedPoint(const double[3]) override { return -1; }
  svtkIdType IsInsertedPoint(double, double, double) override { return -1; }

  /**
   * Determine whether a given point x has been inserted into the points list.
   * Return 0 if a duplicate has been inserted in the list, or 1 else. Note
   * this function always returns 1 since any point is always inserted. The
   * index of the point is returned via ptId.
   */
  int InsertUniquePoint(const double x[3], svtkIdType& ptId) override;

protected:
  svtkNonMergingPointLocator() {}
  ~svtkNonMergingPointLocator() override {}

private:
  svtkNonMergingPointLocator(const svtkNonMergingPointLocator&) = delete;
  void operator=(const svtkNonMergingPointLocator&) = delete;
};

#endif
