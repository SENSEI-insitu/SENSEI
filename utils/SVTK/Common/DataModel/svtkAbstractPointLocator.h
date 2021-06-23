/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAbstractPointLocator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkAbstractPointLocator
 * @brief   abstract class to quickly locate points in 3-space
 *
 * svtkAbstractPointLocator is an abstract spatial search object to quickly locate points
 * in 3D. svtkAbstractPointLocator works by dividing a specified region of space into
 * "rectangular" buckets, and then keeping a list of points that
 * lie in each bucket. Typical operation involves giving a position in 3D
 * and finding the closest point.  The points are provided from the specified
 * dataset input.
 *
 * @sa
 * svtkPointLocator svtkStaticPointLocator svtkMergePoints
 */

#ifndef svtkAbstractPointLocator_h
#define svtkAbstractPointLocator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkLocator.h"

class svtkIdList;

class SVTKCOMMONDATAMODEL_EXPORT svtkAbstractPointLocator : public svtkLocator
{
public:
  //@{
  /**
   * Standard type and print methods.
   */
  svtkTypeMacro(svtkAbstractPointLocator, svtkLocator);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  //@{
  /**
   * Given a position x, return the id of the point closest to it. Alternative
   * method requires separate x-y-z values.
   * These methods are thread safe if BuildLocator() is directly or
   * indirectly called from a single thread first.
   */
  virtual svtkIdType FindClosestPoint(const double x[3]) = 0;
  svtkIdType FindClosestPoint(double x, double y, double z);
  //@}

  /**
   * Given a position x and a radius r, return the id of the point
   * closest to the point in that radius.
   * dist2 returns the squared distance to the point.
   */
  virtual svtkIdType FindClosestPointWithinRadius(
    double radius, const double x[3], double& dist2) = 0;

  //@{
  /**
   * Find the closest N points to a position. This returns the closest
   * N points to a position. A faster method could be created that returned
   * N close points to a position, but necessarily the exact N closest.
   * The returned points are sorted from closest to farthest.
   * These methods are thread safe if BuildLocator() is directly or
   * indirectly called from a single thread first.
   */
  virtual void FindClosestNPoints(int N, const double x[3], svtkIdList* result) = 0;
  void FindClosestNPoints(int N, double x, double y, double z, svtkIdList* result);
  //@}

  //@{
  /**
   * Find all points within a specified radius R of position x.
   * The result is not sorted in any specific manner.
   * These methods are thread safe if BuildLocator() is directly or
   * indirectly called from a single thread first.
   */
  virtual void FindPointsWithinRadius(double R, const double x[3], svtkIdList* result) = 0;
  void FindPointsWithinRadius(double R, double x, double y, double z, svtkIdList* result);
  //@}

  //@{
  /**
   * Provide an accessor to the bounds. Valid after the locator is built.
   */
  virtual double* GetBounds() { return this->Bounds; }
  virtual void GetBounds(double*);
  //@}

  //@{
  /**
   * Return the total number of buckets in the locator. This has meaning only
   * after the locator is constructed.
   */
  svtkGetMacro(NumberOfBuckets, svtkIdType);
  //@}

protected:
  svtkAbstractPointLocator();
  ~svtkAbstractPointLocator() override;

  double Bounds[6];          // bounds of points
  svtkIdType NumberOfBuckets; // total size of locator

private:
  svtkAbstractPointLocator(const svtkAbstractPointLocator&) = delete;
  void operator=(const svtkAbstractPointLocator&) = delete;
};

#endif
