/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStaticCellLocator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkStaticCellLocator
 * @brief   perform fast cell location operations
 *
 * svtkStaticCellLocator is a type of svtkAbstractCellLocator that accelerates
 * certain operations when performing spatial operations on cells. These
 * operations include finding a point that contains a cell, and intersecting
 * cells with a line.
 *
 * svtkStaticCellLocator is an accelerated version of svtkCellLocator. It is
 * threaded (via svtkSMPTools), and supports one-time static construction
 * (i.e., incremental cell insertion is not supported).
 *
 * @warning
 * This class is templated. It may run slower than serial execution if the code
 * is not optimized during compilation. Build in Release or ReleaseWithDebugInfo.
 *
 * @warning
 * This class *always* caches cell bounds.
 *
 * @sa
 * svtkLocator vakAbstractCellLocator svtkCellLocator svtkCellTreeLocator
 * svtkModifiedBSPTree
 */

#ifndef svtkStaticCellLocator_h
#define svtkStaticCellLocator_h

#include "svtkAbstractCellLocator.h"
#include "svtkCommonDataModelModule.h" // For export macro

// Forward declarations for PIMPL
struct svtkCellBinner;
struct svtkCellProcessor;

class SVTKCOMMONDATAMODEL_EXPORT svtkStaticCellLocator : public svtkAbstractCellLocator
{
  friend struct svtkCellBinner;
  friend struct svtkCellProcessor;

public:
  //@{
  /**
   * Standard methods to instantiate, print and obtain type-related information.
   */
  static svtkStaticCellLocator* New();
  svtkTypeMacro(svtkStaticCellLocator, svtkAbstractCellLocator);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  //@{
  /**
   * Set the number of divisions in x-y-z directions. If the Automatic data
   * member is enabled, the Divisions are set according to the
   * NumberOfCellsPerNode and MaxNumberOfBuckets data members. The number
   * of divisions must be >= 1 in each direction.
   */
  svtkSetVector3Macro(Divisions, int);
  svtkGetVectorMacro(Divisions, int, 3);
  //@}

  using svtkAbstractCellLocator::FindClosestPoint;
  using svtkAbstractCellLocator::FindClosestPointWithinRadius;

  /**
   * Test a point to find if it is inside a cell. Returns the cellId if inside
   * or -1 if not.
   */
  svtkIdType FindCell(double pos[3], double svtkNotUsed, svtkGenericCell* cell, double pcoords[3],
    double* weights) override;

  /**
   * Reimplemented from svtkAbstractCellLocator to support bad compilers.
   */
  svtkIdType FindCell(double x[3]) override { return this->Superclass::FindCell(x); }

  /**
   * Return a list of unique cell ids inside of a given bounding box. The
   * user must provide the svtkIdList to populate. This method returns data
   * only after the locator has been built.
   */
  void FindCellsWithinBounds(double* bbox, svtkIdList* cells) override;

  /**
   * Given a finite line defined by the two points (p1,p2), return the list
   * of unique cell ids in the buckets containing the line. It is possible
   * that an empty cell list is returned. The user must provide the svtkIdList
   * cell list to populate. This method returns data only after the locator
   * has been built.
   */
  void FindCellsAlongLine(
    const double p1[3], const double p2[3], double tolerance, svtkIdList* cells) override;

  //@{
  /**
   * Given an unbounded plane defined by an origin o[3] and unit normal n[3],
   * return the list of unique cell ids in the buckets containing the
   * plane. It is possible that an empty cell list is returned. The user must
   * provide the svtkIdList cell list to populate. This method returns data
   * only after the locator has been built.
   */
  void FindCellsAlongPlane(
    const double o[3], const double n[3], double tolerance, svtkIdList* cells);
  //@}

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
  void FindClosestPoint(const double x[3], double closestPoint[3], svtkGenericCell* cell,
    svtkIdType& cellId, int& subId, double& dist2) override;

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
  svtkIdType FindClosestPointWithinRadius(double x[3], double radius, double closestPoint[3],
    svtkGenericCell* cell, svtkIdType& cellId, int& subId, double& dist2, int& inside) override;

  /**
   * Return intersection point (if any) AND the cell which was intersected by
   * the finite line. The cell is returned as a cell id and as a generic cell.
   */
  int IntersectWithLine(const double a0[3], const double a1[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId, svtkIdType& cellId, svtkGenericCell* cell) override;

  /**
   * Reimplemented from svtkAbstractCellLocator to support bad compilers.
   */
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override
  {
    return this->Superclass::IntersectWithLine(p1, p2, tol, t, x, pcoords, subId);
  }

  /**
   * Reimplemented from svtkAbstractCellLocator to support bad compilers.
   */
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId, svtkIdType& cellId) override
  {
    return this->Superclass::IntersectWithLine(p1, p2, tol, t, x, pcoords, subId, cellId);
  }

  /**
   * Reimplemented from svtkAbstractCellLocator to support bad compilers.
   */
  int IntersectWithLine(
    const double p1[3], const double p2[3], svtkPoints* points, svtkIdList* cellIds) override
  {
    return this->Superclass::IntersectWithLine(p1, p2, points, cellIds);
  }

  //@{
  /**
   * Satisfy svtkLocator abstract interface.
   */
  void GenerateRepresentation(int level, svtkPolyData* pd) override;
  void FreeSearchStructure() override;
  void BuildLocator() override;
  //@}

  //@{
  /**
   * Set the maximum number of buckets in the locator. By default the value
   * is set to SVTK_INT_MAX. Note that there are significant performance
   * implications at work here. If the number of buckets is set very large
   * (meaning > SVTK_INT_MAX) then internal sorting may be performed using
   * 64-bit integers (which is much slower than using a 32-bit int). Of
   * course, memory requirements may dramatically increase as well.  It is
   * recommended that the default value be used; but for extremely large data
   * it may be desired to create a locator with an exceptionally large number
   * of buckets. Note also that during initialization of the locator if the
   * MaxNumberOfBuckets threshold is exceeded, the Divisions are scaled down
   * in such a way as not to exceed the MaxNumberOfBuckets proportionally to
   * the size of the bounding box in the x-y-z directions.
   */
  svtkSetClampMacro(MaxNumberOfBuckets, svtkIdType, 1000, SVTK_ID_MAX);
  svtkGetMacro(MaxNumberOfBuckets, svtkIdType);
  //@}

  /**
   * Inform the user as to whether large ids are being used. This flag only
   * has meaning after the locator has been built. Large ids are used when the
   * number of binned points, or the number of bins, is >= the maximum number
   * of buckets (specified by the user). Note that LargeIds are only available
   * on 64-bit architectures.
   */
  bool GetLargeIds() { return this->LargeIds; }

protected:
  svtkStaticCellLocator();
  ~svtkStaticCellLocator() override;

  double Bounds[6]; // Bounding box of the whole dataset
  int Divisions[3]; // Number of sub-divisions in x-y-z directions
  double H[3];      // Width of each bin in x-y-z directions

  svtkIdType MaxNumberOfBuckets; // Maximum number of buckets in locator
  bool LargeIds;                // indicate whether integer ids are small or large

  // Support PIMPLd implementation
  svtkCellBinner* Binner;       // Does the binning
  svtkCellProcessor* Processor; // Invokes methods (templated subclasses)

  // Support query operations
  unsigned char* CellHasBeenVisited;
  unsigned char QueryNumber;

private:
  svtkStaticCellLocator(const svtkStaticCellLocator&) = delete;
  void operator=(const svtkStaticCellLocator&) = delete;
};

#endif
