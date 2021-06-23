/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCellLocatorStrategy.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCellLocatorStrategy
 * @brief   implement a specific svtkPointSet::FindCell() strategy based
 *          on using a cell locator
 *
 * svtkCellLocatorStrategy is implements a FindCell() strategy based on
 * using the FindCell() method in a cell locator. This is often the
 * slowest strategy, but the most robust.
 *
 * @sa
 * svtkFindCellStrategy svtkPointSet
 */

#ifndef svtkCellLocatorStrategy_h
#define svtkCellLocatorStrategy_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkFindCellStrategy.h"

class svtkAbstractCellLocator;

class SVTKCOMMONDATAMODEL_EXPORT svtkCellLocatorStrategy : public svtkFindCellStrategy
{
public:
  /**
   * Construct a svtkFindCellStrategy subclass.
   */
  static svtkCellLocatorStrategy* New();

  //@{
  /**
   * Standard methods for type information and printing.
   */
  svtkTypeMacro(svtkCellLocatorStrategy, svtkFindCellStrategy);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Provide necessary initialization method (see superclass for more
   * information).
   */
  int Initialize(svtkPointSet* ps) override;

  /**
   * Implement the specific strategy.
   */
  svtkIdType FindCell(double x[3], svtkCell* cell, svtkGenericCell* gencell, svtkIdType cellId,
    double tol2, int& subId, double pcoords[3], double* weights) override;

  //@{
  /**
   * Set / get an instance of svtkAbstractCellLocator which is used to
   * implement the strategy for FindCell(). The locator is required to
   * already be built and non-NULL.
   */
  virtual void SetCellLocator(svtkAbstractCellLocator*);
  svtkGetObjectMacro(CellLocator, svtkAbstractCellLocator);
  //@}

protected:
  svtkCellLocatorStrategy();
  ~svtkCellLocatorStrategy() override;

  svtkAbstractCellLocator* CellLocator;
  bool OwnsLocator; // was the locator specified? or taken from associated point set

private:
  svtkCellLocatorStrategy(const svtkCellLocatorStrategy&) = delete;
  void operator=(const svtkCellLocatorStrategy&) = delete;
};

#endif
