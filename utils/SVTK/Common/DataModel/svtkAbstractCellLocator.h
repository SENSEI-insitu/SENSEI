/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAbstractCellLocator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkAbstractCellLocator
 * @brief   an abstract base class for locators which find cells
 *
 * svtkAbstractCellLocator is a spatial search object to quickly locate cells in 3D.
 * svtkAbstractCellLocator supplies a basic interface which concrete subclasses
 * should implement.
 *
 * @warning
 * When deriving a class from svtkAbstractCellLocator, one should include the
 * 'hidden' member functions by the following construct in the derived class
 * \verbatim
 *  using svtkAbstractCellLocator::IntersectWithLine;
 *  using svtkAbstractCellLocator::FindClosestPoint;
 *  using svtkAbstractCellLocator::FindClosestPointWithinRadius;
 * \endverbatim
 *
 *
 * @sa
 * svtkLocator svtkPointLocator svtkOBBTree svtkCellLocator
 */

#ifndef svtkAbstractCellLocator_h
#define svtkAbstractCellLocator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkLocator.h"

class svtkCellArray;
class svtkGenericCell;
class svtkIdList;
class svtkPoints;

class SVTKCOMMONDATAMODEL_EXPORT svtkAbstractCellLocator : public svtkLocator
{
public:
  svtkTypeMacro(svtkAbstractCellLocator, svtkLocator);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Specify the preferred/maximum number of cells in each node/bucket.
   * Default 32. Locators generally operate by subdividing space into
   * smaller regions until the number of cells in each region (or node)
   * reaches the desired level.
   */
  svtkSetClampMacro(NumberOfCellsPerNode, int, 1, SVTK_INT_MAX);
  svtkGetMacro(NumberOfCellsPerNode, int);
  //@}

  //@{
  /**
   * Boolean controls whether the bounds of each cell are computed only
   * once and then saved.  Should be 10 to 20% faster if repeatedly
   * calling any of the Intersect/Find routines and the extra memory
   * won't cause disk caching (24 extra bytes per cell are required to
   * save the bounds).
   */
  svtkSetMacro(CacheCellBounds, svtkTypeBool);
  svtkGetMacro(CacheCellBounds, svtkTypeBool);
  svtkBooleanMacro(CacheCellBounds, svtkTypeBool);
  //@}

  //@{
  /**
   * Boolean controls whether to maintain list of cells in each node.
   * not applicable to all implementations, but if the locator is being used
   * as a geometry simplification technique, there is no need to keep them.
   */
  svtkSetMacro(RetainCellLists, svtkTypeBool);
  svtkGetMacro(RetainCellLists, svtkTypeBool);
  svtkBooleanMacro(RetainCellLists, svtkTypeBool);
  //@}

  //@{
  /**
   * Most Locators build their search structures during BuildLocator
   * but some may delay construction until it is actually needed.
   * If LazyEvaluation is supported, this turns on/off the feature.
   * if not supported, it is ignored.
   */
  svtkSetMacro(LazyEvaluation, svtkTypeBool);
  svtkGetMacro(LazyEvaluation, svtkTypeBool);
  svtkBooleanMacro(LazyEvaluation, svtkTypeBool);
  //@}

  //@{
  /**
   * Some locators support querying a new dataset without rebuilding
   * the search structure (typically this may occur when a dataset
   * changes due to a time update, but is actually the same topology)
   * Turning on this flag enables some locators to skip the rebuilding
   * phase
   */
  svtkSetMacro(UseExistingSearchStructure, svtkTypeBool);
  svtkGetMacro(UseExistingSearchStructure, svtkTypeBool);
  svtkBooleanMacro(UseExistingSearchStructure, svtkTypeBool);
  //@}

  /**
   * Return intersection point (if any) of finite line with cells contained
   * in cell locator. See svtkCell.h parameters documentation.
   */
  virtual int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t,
    double x[3], double pcoords[3], int& subId);

  /**
   * Return intersection point (if any) AND the cell which was intersected by
   * the finite line.
   */
  virtual int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t,
    double x[3], double pcoords[3], int& subId, svtkIdType& cellId);

  /**
   * Return intersection point (if any) AND the cell which was intersected by
   * the finite line. The cell is returned as a cell id and as a generic
   * cell.
   */
  virtual int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t,
    double x[3], double pcoords[3], int& subId, svtkIdType& cellId, svtkGenericCell* cell);

  /**
   * Take the passed line segment and intersect it with the data set.
   * This method assumes that the data set is a svtkPolyData that describes
   * a closed surface, and the intersection points that are returned in
   * 'points' alternate between entrance points and exit points.
   * The return value of the function is 0 if no intersections were found,
   * -1 if point 'a0' lies inside the closed surface, or +1 if point 'a0'
   * lies outside the closed surface.
   * Either 'points' or 'cellIds' can be set to nullptr if you don't want
   * to receive that information. This method is currently only implemented
   * in svtkOBBTree.
   */
  virtual int IntersectWithLine(
    const double p1[3], const double p2[3], svtkPoints* points, svtkIdList* cellIds);

  /**
   * Return the closest point and the cell which is closest to the point x.
   * The closest point is somewhere on a cell, it need not be one of the
   * vertices of the cell.
   */
  virtual void FindClosestPoint(
    const double x[3], double closestPoint[3], svtkIdType& cellId, int& subId, double& dist2);

  /**
   * Return the closest point and the cell which is closest to the point x.
   * The closest point is somewhere on a cell, it need not be one of the
   * vertices of the cell.  This version takes in a svtkGenericCell
   * to avoid allocating and deallocating the cell.  This is much faster than
   * the version which does not take a *cell, especially when this function is
   * called many times in a row such as by a for loop, where the allocation and
   * deallocation can be done only once outside the for loop.  If a cell is
   * found, "cell" contains the points and ptIds for the cell "cellId" upon
   * exit.
   */
  virtual void FindClosestPoint(const double x[3], double closestPoint[3], svtkGenericCell* cell,
    svtkIdType& cellId, int& subId, double& dist2);

  /**
   * Return the closest point within a specified radius and the cell which is
   * closest to the point x. The closest point is somewhere on a cell, it
   * need not be one of the vertices of the cell. This method returns 1 if
   * a point is found within the specified radius. If there are no cells within
   * the specified radius, the method returns 0 and the values of closestPoint,
   * cellId, subId, and dist2 are undefined.
   */
  virtual svtkIdType FindClosestPointWithinRadius(double x[3], double radius, double closestPoint[3],
    svtkIdType& cellId, int& subId, double& dist2);

  /**
   * Return the closest point within a specified radius and the cell which is
   * closest to the point x. The closest point is somewhere on a cell, it
   * need not be one of the vertices of the cell. This method returns 1 if a
   * point is found within the specified radius. If there are no cells within
   * the specified radius, the method returns 0 and the values of
   * closestPoint, cellId, subId, and dist2 are undefined. This version takes
   * in a svtkGenericCell to avoid allocating and deallocating the cell.  This
   * is much faster than the version which does not take a *cell, especially
   * when this function is called many times in a row such as by a for loop,
   * where the allocation and deallocation can be done only once outside the
   * for loop.  If a closest point is found, "cell" contains the points and
   * ptIds for the cell "cellId" upon exit.
   */
  virtual svtkIdType FindClosestPointWithinRadius(double x[3], double radius, double closestPoint[3],
    svtkGenericCell* cell, svtkIdType& cellId, int& subId, double& dist2);

  /**
   * Return the closest point within a specified radius and the cell which is
   * closest to the point x. The closest point is somewhere on a cell, it
   * need not be one of the vertices of the cell. This method returns 1 if a
   * point is found within the specified radius. If there are no cells within
   * the specified radius, the method returns 0 and the values of
   * closestPoint, cellId, subId, and dist2 are undefined. This version takes
   * in a svtkGenericCell to avoid allocating and deallocating the cell.  This
   * is much faster than the version which does not take a *cell, especially
   * when this function is called many times in a row such as by a for loop,
   * where the allocation and dealloction can be done only once outside the
   * for loop.  If a closest point is found, "cell" contains the points and
   * ptIds for the cell "cellId" upon exit.  If a closest point is found,
   * inside returns the return value of the EvaluatePosition call to the
   * closest cell; inside(=1) or outside(=0).
   */
  virtual svtkIdType FindClosestPointWithinRadius(double x[3], double radius, double closestPoint[3],
    svtkGenericCell* cell, svtkIdType& cellId, int& subId, double& dist2, int& inside);

  /**
   * Return a list of unique cell ids inside of a given bounding box. The
   * user must provide the svtkIdList to populate. This method returns data
   * only after the locator has been built.
   */
  virtual void FindCellsWithinBounds(double* bbox, svtkIdList* cells);

  /**
   * Given a finite line defined by the two points (p1,p2), return the list
   * of unique cell ids in the buckets containing the line. It is possible
   * that an empty cell list is returned. The user must provide the svtkIdList
   * to populate. This method returns data only after the locator has been
   * built.
   */
  virtual void FindCellsAlongLine(
    const double p1[3], const double p2[3], double tolerance, svtkIdList* cells);

  /**
   * Returns the Id of the cell containing the point,
   * returns -1 if no cell found. This interface uses a tolerance of zero
   */
  virtual svtkIdType FindCell(double x[3]);

  /**
   * Find the cell containing a given point. returns -1 if no cell found
   * the cell parameters are copied into the supplied variables, a cell must
   * be provided to store the information.
   */
  virtual svtkIdType FindCell(
    double x[3], double tol2, svtkGenericCell* GenCell, double pcoords[3], double* weights);

  /**
   * Quickly test if a point is inside the bounds of a particular cell.
   * Some locators cache cell bounds and this function can make use
   * of fast access to the data.
   */
  virtual bool InsideCellBounds(double x[3], svtkIdType cell_ID);

protected:
  svtkAbstractCellLocator();
  ~svtkAbstractCellLocator() override;

  //@{
  /**
   * This command is used internally by the locator to copy
   * all cell Bounds into the internal CellBounds array. Subsequent
   * calls to InsideCellBounds(...) can make use of the data
   * A valid dataset must be present for this to work. Returns true
   * if bounds wre copied, false otherwise.
   */
  virtual bool StoreCellBounds();
  virtual void FreeCellBounds();
  //@}

  int NumberOfCellsPerNode;
  svtkTypeBool RetainCellLists;
  svtkTypeBool CacheCellBounds;
  svtkTypeBool LazyEvaluation;
  svtkTypeBool UseExistingSearchStructure;
  svtkGenericCell* GenericCell;
  double (*CellBounds)[6];

private:
  svtkAbstractCellLocator(const svtkAbstractCellLocator&) = delete;
  void operator=(const svtkAbstractCellLocator&) = delete;
};

#endif
