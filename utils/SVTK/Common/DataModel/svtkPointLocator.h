/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPointLocator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPointLocator
 * @brief   quickly locate points in 3-space
 *
 * svtkPointLocator is a spatial search object to quickly locate points in 3D.
 * svtkPointLocator works by dividing a specified region of space into a regular
 * array of "rectangular" buckets, and then keeping a list of points that
 * lie in each bucket. Typical operation involves giving a position in 3D
 * and finding the closest point.
 *
 * svtkPointLocator has two distinct methods of interaction. In the first
 * method, you supply it with a dataset, and it operates on the points in
 * the dataset. In the second method, you supply it with an array of points,
 * and the object operates on the array.
 *
 * @warning
 * Many other types of spatial locators have been developed such as
 * octrees and kd-trees. These are often more efficient for the
 * operations described here.
 *
 * @sa
 * svtkCellPicker svtkPointPicker svtkStaticPointLocator
 */

#ifndef svtkPointLocator_h
#define svtkPointLocator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkIncrementalPointLocator.h"

class svtkCellArray;
class svtkIdList;
class svtkNeighborPoints;
class svtkPoints;

class SVTKCOMMONDATAMODEL_EXPORT svtkPointLocator : public svtkIncrementalPointLocator
{
public:
  /**
   * Construct with automatic computation of divisions, averaging
   * 25 points per bucket.
   */
  static svtkPointLocator* New();

  //@{
  /**
   * Standard methods for type management and printing.
   */
  svtkTypeMacro(svtkPointLocator, svtkIncrementalPointLocator);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  //@{
  /**
   * Set the number of divisions in x-y-z directions.
   */
  svtkSetVector3Macro(Divisions, int);
  svtkGetVectorMacro(Divisions, int, 3);
  //@}

  //@{
  /**
   * Specify the average number of points in each bucket.
   */
  svtkSetClampMacro(NumberOfPointsPerBucket, int, 1, SVTK_INT_MAX);
  svtkGetMacro(NumberOfPointsPerBucket, int);
  //@}

  // Re-use any superclass signatures that we don't override.
  using svtkAbstractPointLocator::FindClosestPoint;

  /**
   * Given a position x, return the id of the point closest to it. Alternative
   * method requires separate x-y-z values.
   * These methods are thread safe if BuildLocator() is directly or
   * indirectly called from a single thread first.
   */
  svtkIdType FindClosestPoint(const double x[3]) override;

  //@{
  /**
   * Given a position x and a radius r, return the id of the point
   * closest to the point in that radius.
   * These methods are thread safe if BuildLocator() is directly or
   * indirectly called from a single thread first. dist2 returns the squared
   * distance to the point.
   */
  svtkIdType FindClosestPointWithinRadius(double radius, const double x[3], double& dist2) override;
  virtual svtkIdType FindClosestPointWithinRadius(
    double radius, const double x[3], double inputDataLength, double& dist2);
  //@}

  /**
   * Initialize the point insertion process. The newPts is an object
   * representing point coordinates into which incremental insertion methods
   * place their data. Bounds are the box that the points lie in.
   * Not thread safe.
   */
  int InitPointInsertion(svtkPoints* newPts, const double bounds[6]) override;

  /**
   * Initialize the point insertion process. The newPts is an object
   * representing point coordinates into which incremental insertion methods
   * place their data. Bounds are the box that the points lie in.
   * Not thread safe.
   */
  int InitPointInsertion(svtkPoints* newPts, const double bounds[6], svtkIdType estSize) override;

  /**
   * Incrementally insert a point into search structure with a particular
   * index value. You should use the method IsInsertedPoint() to see whether
   * this point has already been inserted (that is, if you desire to prevent
   * duplicate points). Before using this method you must make sure that
   * newPts have been supplied, the bounds has been set properly, and that
   * divs are properly set. (See InitPointInsertion().)
   * Not thread safe.
   */
  void InsertPoint(svtkIdType ptId, const double x[3]) override;

  /**
   * Incrementally insert a point into search structure. The method returns
   * the insertion location (i.e., point id). You should use the method
   * IsInsertedPoint() to see whether this point has already been
   * inserted (that is, if you desire to prevent duplicate points).
   * Before using this method you must make sure that newPts have been
   * supplied, the bounds has been set properly, and that divs are
   * properly set. (See InitPointInsertion().)
   * Not thread safe.
   */
  svtkIdType InsertNextPoint(const double x[3]) override;

  //@{
  /**
   * Determine whether point given by x[3] has been inserted into points list.
   * Return id of previously inserted point if this is true, otherwise return
   * -1. This method is thread safe.
   */
  svtkIdType IsInsertedPoint(double x, double y, double z) override
  {
    double xyz[3];
    xyz[0] = x;
    xyz[1] = y;
    xyz[2] = z;
    return this->IsInsertedPoint(xyz);
  };
  svtkIdType IsInsertedPoint(const double x[3]) override;
  //@}

  /**
   * Determine whether point given by x[3] has been inserted into points list.
   * Return 0 if point was already in the list, otherwise return 1. If the
   * point was not in the list, it will be ADDED.  In either case, the id of
   * the point (newly inserted or not) is returned in the ptId argument.
   * Note this combines the functionality of IsInsertedPoint() followed
   * by a call to InsertNextPoint().
   * This method is not thread safe.
   */
  int InsertUniquePoint(const double x[3], svtkIdType& ptId) override;

  /**
   * Given a position x, return the id of the point closest to it. This method
   * is used when performing incremental point insertion. Note that -1
   * indicates that no point was found.
   * This method is thread safe if BuildLocator() is directly or
   * indirectly called from a single thread first.
   */
  svtkIdType FindClosestInsertedPoint(const double x[3]) override;

  /**
   * Find the closest N points to a position. This returns the closest
   * N points to a position. A faster method could be created that returned
   * N close points to a position, but necessarily the exact N closest.
   * The returned points are sorted from closest to farthest.
   * These methods are thread safe if BuildLocator() is directly or
   * indirectly called from a single thread first.
   */
  void FindClosestNPoints(int N, const double x[3], svtkIdList* result) override;

  //@{
  /**
   * Find the closest points to a position such that each octant of
   * space around the position contains at least N points. Loosely
   * limit the search to a maximum number of points evaluated, M.
   * These methods are thread safe if BuildLocator() is directly or
   * indirectly called from a single thread first.
   */
  virtual void FindDistributedPoints(int N, const double x[3], svtkIdList* result, int M);
  virtual void FindDistributedPoints(int N, double x, double y, double z, svtkIdList* result, int M);
  //@}

  /**
   * Find all points within a specified radius R of position x.
   * The result is not sorted in any specific manner.
   * These methods are thread safe if BuildLocator() is directly or
   * indirectly called from a single thread first.
   */
  void FindPointsWithinRadius(double R, const double x[3], svtkIdList* result) override;

  /**
   * Given a position x, return the list of points in the bucket that
   * contains the point. It is possible that nullptr is returned. The user
   * provides an ijk array that is the indices into the locator.
   * This method is thread safe.
   */
  virtual svtkIdList* GetPointsInBucket(const double x[3], int ijk[3]);

  //@{
  /**
   * Provide an accessor to the points.
   */
  svtkGetObjectMacro(Points, svtkPoints);
  //@}

  //@{
  /**
   * See svtkLocator interface documentation.
   * These methods are not thread safe.
   */
  void Initialize() override;
  void FreeSearchStructure() override;
  void BuildLocator() override;
  void GenerateRepresentation(int level, svtkPolyData* pd) override;
  //@}

protected:
  svtkPointLocator();
  ~svtkPointLocator() override;

  // place points in appropriate buckets
  void GetBucketNeighbors(
    svtkNeighborPoints* buckets, const int ijk[3], const int ndivs[3], int level);
  void GetOverlappingBuckets(
    svtkNeighborPoints* buckets, const double x[3], const int ijk[3], double dist, int level);
  void GetOverlappingBuckets(svtkNeighborPoints* buckets, const double x[3], double dist,
    int prevMinLevel[3], int prevMaxLevel[3]);
  void GenerateFace(int face, int i, int j, int k, svtkPoints* pts, svtkCellArray* polys);
  double Distance2ToBucket(const double x[3], const int nei[3]);
  double Distance2ToBounds(const double x[3], const double bounds[6]);

  svtkPoints* Points;           // Used for merging points
  int Divisions[3];            // Number of sub-divisions in x-y-z directions
  int NumberOfPointsPerBucket; // Used with previous boolean to control subdivide
  svtkIdList** HashTable;       // lists of point ids in buckets
  double H[3];                 // width of each bucket in x-y-z directions

  double InsertionTol2;
  svtkIdType InsertionPointId;
  double InsertionLevel;

  // These are inlined methods and data members for performance reasons
  double HX, HY, HZ;
  double FX, FY, FZ, BX, BY, BZ;
  svtkIdType XD, YD, ZD, SliceSize;

  void GetBucketIndices(const double* x, int ijk[3]) const
  {
    // Compute point index. Make sure it lies within range of locator.
    svtkIdType tmp0 = static_cast<svtkIdType>(((x[0] - this->BX) * this->FX));
    svtkIdType tmp1 = static_cast<svtkIdType>(((x[1] - this->BY) * this->FY));
    svtkIdType tmp2 = static_cast<svtkIdType>(((x[2] - this->BZ) * this->FZ));

    ijk[0] = tmp0 < 0 ? 0 : (tmp0 >= this->XD ? this->XD - 1 : tmp0);
    ijk[1] = tmp1 < 0 ? 0 : (tmp1 >= this->YD ? this->YD - 1 : tmp1);
    ijk[2] = tmp2 < 0 ? 0 : (tmp2 >= this->ZD ? this->ZD - 1 : tmp2);
  }

  svtkIdType GetBucketIndex(const double* x) const
  {
    int ijk[3];
    this->GetBucketIndices(x, ijk);
    return ijk[0] + ijk[1] * this->XD + ijk[2] * this->SliceSize;
  }

  void ComputePerformanceFactors();

private:
  svtkPointLocator(const svtkPointLocator&) = delete;
  void operator=(const svtkPointLocator&) = delete;
};

#endif
