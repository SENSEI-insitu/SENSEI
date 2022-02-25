/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBSPIntersections.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*----------------------------------------------------------------------------
 Copyright (c) Sandia Corporation
 See Copyright.txt or http://www.paraview.org/HTML/Copyright.html for details.
----------------------------------------------------------------------------*/

/**
 * @class   svtkBSPIntersections
 * @brief   Perform calculations (mostly intersection
 *   calculations) on regions of a 3D binary spatial partitioning.
 *
 *
 *    Given an axis aligned binary spatial partitioning described by a
 *    svtkBSPCuts object, perform intersection queries on various
 *    geometric entities with regions of the spatial partitioning.
 *
 * @sa
 *    svtkBSPCuts  svtkKdTree
 */

#ifndef svtkBSPIntersections_h
#define svtkBSPIntersections_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkTimeStamp;
class svtkCell;
class svtkKdNode;
class svtkBSPCuts;

class SVTKCOMMONDATAMODEL_EXPORT svtkBSPIntersections : public svtkObject
{
public:
  svtkTypeMacro(svtkBSPIntersections, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  static svtkBSPIntersections* New();

  /**
   * Define the binary spatial partitioning.
   */

  void SetCuts(svtkBSPCuts* cuts);
  svtkGetObjectMacro(Cuts, svtkBSPCuts);

  /**
   * Get the bounds of the whole space (xmin, xmax, ymin, ymax, zmin, zmax)
   * Return 0 if OK, 1 on error.
   */

  int GetBounds(double* bounds);

  /**
   * The number of regions in the binary spatial partitioning
   */

  int GetNumberOfRegions();

  /**
   * Get the spatial bounds of a particular region
   * Return 0 if OK, 1 on error.
   */

  int GetRegionBounds(int regionID, double bounds[6]);

  /**
   * Get the bounds of the data within the k-d tree region, possibly
   * smaller than the bounds of the region.
   * Return 0 if OK, 1 on error.
   */

  int GetRegionDataBounds(int regionID, double bounds[6]);

  //@{
  /**
   * Determine whether a region of the spatial decomposition
   * intersects an axis aligned box.
   */
  int IntersectsBox(int regionId, double* x);
  int IntersectsBox(
    int regionId, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax);
  //@}

  //@{
  /**
   * Compute a list of the Ids of all regions that
   * intersect the specified axis aligned box.
   * Returns: the number of ids in the list.
   */
  int IntersectsBox(int* ids, int len, double* x);
  int IntersectsBox(
    int* ids, int len, double x0, double x1, double y0, double y1, double z0, double z1);
  //@}

  /**
   * Determine whether a region of the spatial decomposition
   * intersects a sphere, given the center of the sphere
   * and the square of it's radius.
   */
  int IntersectsSphere2(int regionId, double x, double y, double z, double rSquared);

  /**
   * Compute a list of the Ids of all regions that
   * intersect the specified sphere.  The sphere is given
   * by it's center and the square of it's radius.
   * Returns: the number of ids in the list.
   */
  int IntersectsSphere2(int* ids, int len, double x, double y, double z, double rSquared);

  /**
   * Determine whether a region of the spatial decomposition
   * intersects the given cell.  If you already
   * know the region that the cell centroid lies in, provide
   * that as the last argument to make the computation quicker.
   */
  int IntersectsCell(int regionId, svtkCell* cell, int cellRegion = -1);

  /**
   * Compute a list of the Ids of all regions that
   * intersect the given cell.  If you already
   * know the region that the cell centroid lies in, provide
   * that as the last argument to make the computation quicker.
   * Returns the number of regions the cell intersects.
   */
  int IntersectsCell(int* ids, int len, svtkCell* cell, int cellRegion = -1);

  /**
   * When computing the intersection of k-d tree regions with other
   * objects, we use the spatial bounds of the region.  To use the
   * tighter bound of the bounding box of the data within the region,
   * set this variable ON.  (Specifying data bounds in the svtkBSPCuts
   * object is optional.  If data bounds were not specified, this
   * option has no meaning.)
   */

  svtkGetMacro(ComputeIntersectionsUsingDataBounds, int);
  void SetComputeIntersectionsUsingDataBounds(int c);
  void ComputeIntersectionsUsingDataBoundsOn();
  void ComputeIntersectionsUsingDataBoundsOff();

protected:
  svtkBSPIntersections();
  ~svtkBSPIntersections() override;

  svtkGetMacro(RegionListBuildTime, svtkMTimeType);

  int BuildRegionList();

  svtkKdNode** GetRegionList() { return this->RegionList; }

  double CellBoundsCache[6]; // to speed cell intersection queries

  enum
  {
    XDIM = 0, // don't change these values
    YDIM = 1,
    ZDIM = 2
  };

private:
  static int NumberOfLeafNodes(svtkKdNode* kd);
  static void SetIDRanges(svtkKdNode* kd, int& min, int& max);

  int SelfRegister(svtkKdNode* kd);

  static void SetCellBounds(svtkCell* cell, double* bounds);

  int _IntersectsBox(svtkKdNode* node, int* ids, int len, double x0, double x1, double y0, double y1,
    double z0, double z1);

  int _IntersectsSphere2(
    svtkKdNode* node, int* ids, int len, double x, double y, double z, double rSquared);

  int _IntersectsCell(svtkKdNode* node, int* ids, int len, svtkCell* cell, int cellRegion = -1);

  svtkBSPCuts* Cuts;

  int NumberOfRegions;
  svtkKdNode** RegionList;

  svtkTimeStamp RegionListBuildTime;

  int ComputeIntersectionsUsingDataBounds;

  svtkBSPIntersections(const svtkBSPIntersections&) = delete;
  void operator=(const svtkBSPIntersections&) = delete;
};
#endif
