/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkFindCellStrategy.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkFindCellStrategy
 * @brief   helper class to manage the svtkPointSet::FindCell() METHOD
 *
 * svtkFindCellStrategy is a helper class to manage the use of locators for
 * locating cells containing a query point x[3], the so-called FindCell()
 * method. The use of svtkDataSet::FindCell() is a common operation in
 * applications such as streamline generation and probing. However, in some
 * dataset types FindCell() can be implemented very simply (e.g.,
 * svtkImageData) while in other datasets it is a complex operation requiring
 * supplemental objects like locators to perform efficiently. In particular,
 * svtkPointSet and its subclasses (like svtkUnstructuredGrid) require complex
 * strategies to efficiently implement the FindCell() operation. Subclasses
 * of the abstract svtkFindCellStrategy implement several of these strategies.
 *
 * The are two key methods to this class and subclasses. The Initialize()
 * method negotiates with an input dataset to define the locator to use:
 * either a locator associated with the inout dataset, or possibly an
 * alternative locator defined by the strategy (subclasses do this). The
 * second important method, FindCell() mimics svtkDataSet::FindCell() and
 * can be used in place of it.
 *
 * @sa
 * svtkPointSet svtkPolyData svtkStructuredGrid svtkUnstructuredGrid
 */

#ifndef svtkFindCellStrategy_h
#define svtkFindCellStrategy_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkCell;
class svtkGenericCell;
class svtkPointSet;

class SVTKCOMMONDATAMODEL_EXPORT svtkFindCellStrategy : public svtkObject
{
public:
  //@{
  /**
   * Standard methdos for type information and printing.
   */
  svtkTypeMacro(svtkFindCellStrategy, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * All subclasses of this class must provide an initialize method.  This
   * method performs handshaking and setup between the svtkPointSet dataset
   * and associated locator(s). A return value==0 means the initialization
   * process failed.
   */
  virtual int Initialize(svtkPointSet* ps);

  /**
   * Virtual method for finding a cell. Subclasses must satisfy this API.
   * This method is of the same signature as svtkDataSet::FindCell().
   */
  virtual svtkIdType FindCell(double x[3], svtkCell* cell, svtkGenericCell* gencell, svtkIdType cellId,
    double tol2, int& subId, double pcoords[3], double* weights) = 0;

protected:
  svtkFindCellStrategy();
  ~svtkFindCellStrategy() override;

  svtkPointSet* PointSet; // svtkPointSet which this strategy is associated with
  double Bounds[6];      // bounding box of svtkPointSet

  svtkTimeStamp InitializeTime; // time at which strategy was initialized

private:
  svtkFindCellStrategy(const svtkFindCellStrategy&) = delete;
  void operator=(const svtkFindCellStrategy&) = delete;
};

#endif
