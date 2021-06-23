/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkClosestNPointsStrategy.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkClosestNPointsStrategy
 * @brief   implement a specific svtkPointSet::FindCell() strategy based
 *          on the N closest points
 *
 * svtkClosestNPointsStrategy is implements a FindCell() strategy based on
 * locating the closest N points in a dataset, and then searching attached
 * cells. This class extends its superclass svtkClosestPointStrategy by looking
 * at the additional N points.
 *
 * @sa
 * svtkFindCellStrategy svtkPointSet
 */

#ifndef svtkClosestNPointsStrategy_h
#define svtkClosestNPointsStrategy_h

#include "svtkClosestPointStrategy.h"
#include "svtkCommonDataModelModule.h" // For export macro

class SVTKCOMMONDATAMODEL_EXPORT svtkClosestNPointsStrategy : public svtkClosestPointStrategy
{
public:
  /**
   * Construct a svtkFindCellStrategy subclass.
   */
  static svtkClosestNPointsStrategy* New();

  //@{
  /**
   * Standard methods for type information and printing.
   */
  svtkTypeMacro(svtkClosestNPointsStrategy, svtkClosestPointStrategy);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  // Re-use any superclass signatures that we don't override.
  using svtkClosestPointStrategy::Initialize;

  /**
   * Implement the specific strategy.
   */
  svtkIdType FindCell(double x[3], svtkCell* cell, svtkGenericCell* gencell, svtkIdType cellId,
    double tol2, int& subId, double pcoords[3], double* weights) override;

  //@{
  /**
   * Set / get the value for the N closest points.
   */
  svtkSetClampMacro(ClosestNPoints, int, 1, 100);
  svtkGetMacro(ClosestNPoints, int);
  //@}

protected:
  svtkClosestNPointsStrategy();
  ~svtkClosestNPointsStrategy() override;

  int ClosestNPoints;

private:
  svtkClosestNPointsStrategy(const svtkClosestNPointsStrategy&) = delete;
  void operator=(const svtkClosestNPointsStrategy&) = delete;
};

#endif
