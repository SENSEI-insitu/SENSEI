/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkClosestPointStrategy.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkClosestPointStrategy
 * @brief   implement a specific svtkPointSet::FindCell() strategy based
 *          on closest point
 *
 * svtkClosestPointStrategy is implements a FindCell() strategy based on
 * locating the closest point in a dataset, and then searching the attached
 * cells. While relatively fast, it does not always return the correct result
 * (it may not find a cell, since the closest cell may not be connected to the
 * closest point). svtkCellLocatorStrategy or svtkClosestNPointsStrategy will
 * produce better results at the cost of speed.
 *
 * @sa
 * svtkFindCellStrategy svtkPointSet svtkCellLocatorStrategy
 * svtkClosestNPointsStrategy
 */

#ifndef svtkClosestPointStrategy_h
#define svtkClosestPointStrategy_h

#include "svtkCell.h"                  //inline SelectCell
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkFindCellStrategy.h"
#include "svtkGenericCell.h" //inline SelectCell
#include "svtkPointSet.h"    //inline SelectCell

#include <set> // For tracking visited cells

class svtkIdList;
class svtkAbstractPointLocator;

class SVTKCOMMONDATAMODEL_EXPORT svtkClosestPointStrategy : public svtkFindCellStrategy
{
public:
  /**
   * Construct a svtkFindCellStrategy subclass.
   */
  static svtkClosestPointStrategy* New();

  //@{
  /**
   * Standard methods for type information and printing.
   */
  svtkTypeMacro(svtkClosestPointStrategy, svtkFindCellStrategy);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Provide the necessary initialization method (see superclass for more
   * information). This method sets up the point locator, svtkPointSet relationship.
   * It will use the svtkPointSet's default locator if not defined by
   * SetPointLocator() below.
   */
  int Initialize(svtkPointSet* ps) override;

  /**
   * Implement the specific strategy. This method should only be called
   * after the Initialize() method has been invoked.
   */
  svtkIdType FindCell(double x[3], svtkCell* cell, svtkGenericCell* gencell, svtkIdType cellId,
    double tol2, int& subId, double pcoords[3], double* weights) override;

  //@{
  /**
   * Set / get an instance of svtkAbstractPointLocator which is used to
   * implement the strategy for FindCell(). Note if a locator is not
   * specified, then the default locator instantiated by the svtkPointSet
   * provided in the Initialize() method is used.
   */
  virtual void SetPointLocator(svtkAbstractPointLocator*);
  svtkGetObjectMacro(PointLocator, svtkAbstractPointLocator);
  //@}

  /**
   * Subclasses use this method to select the current cell.
   */
  svtkCell* SelectCell(svtkPointSet* self, svtkIdType cellId, svtkCell* cell, svtkGenericCell* gencell);

protected:
  svtkClosestPointStrategy();
  ~svtkClosestPointStrategy() override;

  std::set<svtkIdType> VisitedCells;
  svtkIdList* PointIds;
  svtkIdList* Neighbors;
  svtkIdList* CellIds;
  svtkIdList* NearPointIds;

  svtkAbstractPointLocator* PointLocator;
  bool OwnsLocator; // was the locator specified? or taken from associated point set

private:
  svtkClosestPointStrategy(const svtkClosestPointStrategy&) = delete;
  void operator=(const svtkClosestPointStrategy&) = delete;
};

// Handle cases where starting cell is provided or not
inline svtkCell* svtkClosestPointStrategy::SelectCell(
  svtkPointSet* self, svtkIdType cellId, svtkCell* cell, svtkGenericCell* gencell)
{
  if (!cell)
  {
    if (gencell)
    {
      self->GetCell(cellId, gencell);
      cell = gencell;
    }
    else
    {
      cell = self->GetCell(cellId);
    }
  }
  return cell;
}

#endif
