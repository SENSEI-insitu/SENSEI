/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkOctreePointLocator.h

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
 * @class   svtkOctreePointLocator
 * @brief   an octree spatial decomposition of a set of points
 *
 *
 * Given a svtkDataSet, create an octree that is locally refined
 * such that all leaf octants contain less than a certain
 * amount of points.  Note that there is no size constraint that
 * a leaf octant in relation to any of its neighbors.
 *
 * This class can also generate a PolyData representation of
 * the boundaries of the spatial regions in the decomposition.
 *
 * @sa
 * svtkLocator svtkPointLocator svtkOctreePointLocatorNode
 */

#ifndef svtkOctreePointLocator_h
#define svtkOctreePointLocator_h

#include "svtkAbstractPointLocator.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkCellArray;
class svtkIdTypeArray;
class svtkOctreePointLocatorNode;
class svtkPoints;
class svtkPolyData;

class SVTKCOMMONDATAMODEL_EXPORT svtkOctreePointLocator : public svtkAbstractPointLocator
{
public:
  svtkTypeMacro(svtkOctreePointLocator, svtkAbstractPointLocator);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  static svtkOctreePointLocator* New();

  //@{
  /**
   * Maximum number of points per spatial region.  Default is 100.
   */
  svtkSetMacro(MaximumPointsPerRegion, int);
  svtkGetMacro(MaximumPointsPerRegion, int);
  //@}

  //@{
  /**
   * Get/Set macro for CreateCubicOctants.
   */
  svtkSetMacro(CreateCubicOctants, int);
  svtkGetMacro(CreateCubicOctants, int);
  //@}

  //@{
  /**
   * Some algorithms on octrees require a value that is a very
   * small distance relative to the diameter of the entire space
   * divided by the octree.  This factor is the maximum axis-aligned
   * width of the space multiplied by 10e-6.
   */
  svtkGetMacro(FudgeFactor, double);
  svtkSetMacro(FudgeFactor, double);
  //@}

  //@{
  /**
   * Get the spatial bounds of the entire octree space. Sets
   * bounds array to xmin, xmax, ymin, ymax, zmin, zmax.
   */
  double* GetBounds() override;
  void GetBounds(double* bounds) override;
  //@}

  //@{
  /**
   * The number of leaf nodes of the tree, the spatial regions
   */
  svtkGetMacro(NumberOfLeafNodes, int);
  //@}

  /**
   * Get the spatial bounds of octree region
   */
  void GetRegionBounds(int regionID, double bounds[6]);

  /**
   * Get the bounds of the data within the leaf node
   */
  void GetRegionDataBounds(int leafNodeID, double bounds[6]);

  /**
   * Get the id of the leaf region containing the specified location.
   */
  int GetRegionContainingPoint(double x, double y, double z);

  /**
   * Create the octree decomposition of the cells of the data set
   * or data sets.  Cells are assigned to octree spatial regions
   * based on the location of their centroids.
   */
  void BuildLocator() override;

  //@{
  /**
   * Return the Id of the point that is closest to the given point.
   * Set the square of the distance between the two points.
   */
  svtkIdType FindClosestPoint(const double x[3]) override;
  svtkIdType FindClosestPoint(double x, double y, double z, double& dist2);
  //@}

  /**
   * Given a position x and a radius r, return the id of the point
   * closest to the point in that radius.
   * dist2 returns the squared distance to the point.
   */
  svtkIdType FindClosestPointWithinRadius(double radius, const double x[3], double& dist2) override;

  //@{
  /**
   * Find the Id of the point in the given leaf region which is
   * closest to the given point.  Return the ID of the point,
   * and set the square of the distance of between the points.
   */
  svtkIdType FindClosestPointInRegion(int regionId, double* x, double& dist2);
  svtkIdType FindClosestPointInRegion(int regionId, double x, double y, double z, double& dist2);
  //@}

  /**
   * Find all points within a specified radius of position x.
   * The result is not sorted in any specific manner.
   */
  void FindPointsWithinRadius(double radius, const double x[3], svtkIdList* result) override;

  /**
   * Find the closest N points to a position. This returns the closest
   * N points to a position. A faster method could be created that returned
   * N close points to a position, but not necessarily the exact N closest.
   * The returned points are sorted from closest to farthest.
   * These methods are thread safe if BuildLocator() is directly or
   * indirectly called from a single thread first.
   */
  void FindClosestNPoints(int N, const double x[3], svtkIdList* result) override;

  /**
   * Get a list of the original IDs of all points in a leaf node.
   */
  svtkIdTypeArray* GetPointsInRegion(int leafNodeId);

  /**
   * Delete the octree data structure.
   */
  void FreeSearchStructure() override;

  /**
   * Create a polydata representation of the boundaries of
   * the octree regions.
   */
  void GenerateRepresentation(int level, svtkPolyData* pd) override;

  /**
   * Fill ids with points found in area.  The area is a 6-tuple containing
   * (xmin, xmax, ymin, ymax, zmin, zmax).
   * This method will clear the array by default.  To append ids to an array,
   * set clearArray to false.
   */
  void FindPointsInArea(double* area, svtkIdTypeArray* ids, bool clearArray = true);

protected:
  svtkOctreePointLocator();
  ~svtkOctreePointLocator() override;

  svtkOctreePointLocatorNode* Top;
  svtkOctreePointLocatorNode** LeafNodeList; // indexed by region/node ID

  void BuildLeafNodeList(svtkOctreePointLocatorNode* node, int& index);

  //@{
  /**
   * Given a point and a node return the leaf node id that contains the
   * point.  The function returns -1 if no nodes contain the point.
   */
  int FindRegion(svtkOctreePointLocatorNode* node, float x, float y, float z);
  int FindRegion(svtkOctreePointLocatorNode* node, double x, double y, double z);
  //@}

  static void SetDataBoundsToSpatialBounds(svtkOctreePointLocatorNode* node);

  static void DeleteAllDescendants(svtkOctreePointLocatorNode* octant);

  /**
   * Recursive helper for public FindPointsWithinRadius.  radiusSquared
   * is the square of the radius and is used in order to avoid the
   * expensive square root calculation.
   */
  void FindPointsWithinRadius(
    svtkOctreePointLocatorNode* node, double radiusSquared, const double x[3], svtkIdList* ids);

  // Recursive helper for public FindPointsWithinRadius
  void AddAllPointsInRegion(svtkOctreePointLocatorNode* node, svtkIdList* ids);

  // Recursive helper for public FindPointsInArea
  void FindPointsInArea(svtkOctreePointLocatorNode* node, double* area, svtkIdTypeArray* ids);

  // Recursive helper for public FindPointsInArea
  void AddAllPointsInRegion(svtkOctreePointLocatorNode* node, svtkIdTypeArray* ids);

  void DivideRegion(svtkOctreePointLocatorNode* node, int* ordering, int level);

  int DivideTest(int size, int level);

  void AddPolys(svtkOctreePointLocatorNode* node, svtkPoints* pts, svtkCellArray* polys);

  /**
   * Given a leaf node id and point, return the local id and the squared distance
   * between the closest point and the given point.
   */
  int _FindClosestPointInRegion(int leafNodeId, double x, double y, double z, double& dist2);

  /**
   * Given a location and a radiues, find the closest point within
   * this radius.  The function does not examine the region with Id
   * equal to skipRegion (do not set skipRegion to -1 as all non-leaf
   * octants have -1 as their Id).  The Id is returned along with
   * the distance squared for success and -1 is returned for failure.
   */
  int FindClosestPointInSphere(
    double x, double y, double z, double radius, int skipRegion, double& dist2);

  //@{
  /**
   * The maximum number of points in a region/octant before it is subdivided.
   */
  int MaximumPointsPerRegion;
  int NumberOfLeafNodes;
  //@}

  double FudgeFactor; // a very small distance, relative to the dataset's size
  int NumberOfLocatorPoints;
  float* LocatorPoints;
  int* LocatorIds;

  float MaxWidth;

  /**
   * If CreateCubicOctants is non-zero, the bounding box of the points will
   * be expanded such that all octants that are created will be cube-shaped
   * (e.g. have equal lengths on each side).  This may make the tree deeper
   * but also results in better shaped octants for doing searches. The
   * default is to have this set on.
   */
  int CreateCubicOctants;

  svtkOctreePointLocator(const svtkOctreePointLocator&) = delete;
  void operator=(const svtkOctreePointLocator&) = delete;
};
#endif
