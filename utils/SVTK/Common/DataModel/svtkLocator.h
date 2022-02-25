/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLocator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkLocator
 * @brief   abstract base class for objects that accelerate spatial searches
 *
 * svtkLocator is an abstract base class for spatial search objects, or
 * locators. The principle behind locators is that they divide 3-space into
 * small regions (or "buckets") that can be quickly found in response to
 * queries about point location, line intersection, or object-object
 * intersection.
 *
 * The purpose of this base class is to provide data members and methods
 * shared by all locators. The GenerateRepresentation() is one such
 * interesting method.  This method works in conjunction with
 * svtkLocatorFilter to create polygonal representations for the locator. For
 * example, if the locator is an OBB tree (i.e., svtkOBBTree.h), then the
 * representation is a set of one or more oriented bounding boxes, depending
 * upon the specified level.
 *
 * Locators typically work as follows. One or more "entities", such as points
 * or cells, are inserted into the locator structure. These entities are
 * associated with one or more buckets. Then, when performing geometric
 * operations, the operations are performed first on the buckets, and then if
 * the operation tests positive, then on the entities in the bucket. For
 * example, during collision tests, the locators are collided first to
 * identify intersecting buckets. If an intersection is found, more expensive
 * operations are then carried out on the entities in the bucket.
 *
 * To obtain good performance, locators are often organized in a tree
 * structure.  In such a structure, there are frequently multiple "levels"
 * corresponding to different nodes in the tree. So the word level (in the
 * context of the locator) can be used to specify a particular representation
 * in the tree.  For example, in an octree (which is a tree with 8 children),
 * level 0 is the bounding box, or root octant, and level 1 consists of its
 * eight children.
 *
 * @warning
 * There is a concept of static and incremental locators. Static locators are
 * constructed one time, and then support appropriate queries. Incremental
 * locators may have data inserted into them over time (e.g., adding new
 * points during the process of isocontouring).
 *
 * @sa
 * svtkPointLocator svtkCellLocator svtkOBBTree svtkMergePoints
 */

#ifndef svtkLocator_h
#define svtkLocator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkDataSet;
class svtkPolyData;

class SVTKCOMMONDATAMODEL_EXPORT svtkLocator : public svtkObject
{
public:
  //@{
  /**
   * Standard type and print methods.
   */
  svtkTypeMacro(svtkLocator, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  //@{
  /**
   * Build the locator from the points/cells defining this dataset.
   */
  virtual void SetDataSet(svtkDataSet*);
  svtkGetObjectMacro(DataSet, svtkDataSet);
  //@}

  //@{
  /**
   * Set the maximum allowable level for the tree. If the Automatic ivar is
   * off, this will be the target depth of the locator.
   * Initial value is 8.
   */
  svtkSetClampMacro(MaxLevel, int, 0, SVTK_INT_MAX);
  svtkGetMacro(MaxLevel, int);
  //@}

  //@{
  /**
   * Get the level of the locator (determined automatically if Automatic is
   * true). The value of this ivar may change each time the locator is built.
   * Initial value is 8.
   */
  svtkGetMacro(Level, int);
  //@}

  //@{
  /**
   * Boolean controls whether locator depth/resolution of locator is computed
   * automatically from average number of entities in bucket. If not set,
   * there will be an explicit method to control the construction of the
   * locator (found in the subclass).
   */
  svtkSetMacro(Automatic, svtkTypeBool);
  svtkGetMacro(Automatic, svtkTypeBool);
  svtkBooleanMacro(Automatic, svtkTypeBool);
  //@}

  //@{
  /**
   * Specify absolute tolerance (in world coordinates) for performing
   * geometric operations.
   */
  svtkSetClampMacro(Tolerance, double, 0.0, SVTK_DOUBLE_MAX);
  svtkGetMacro(Tolerance, double);
  //@}

  /**
   * Cause the locator to rebuild itself if it or its input dataset has
   * changed.
   */
  virtual void Update();

  /**
   * Initialize locator. Frees memory and resets object as appropriate.
   */
  virtual void Initialize();

  /**
   * Build the locator from the input dataset.
   */
  virtual void BuildLocator() = 0;

  /**
   * Free the memory required for the spatial data structure.
   */
  virtual void FreeSearchStructure() = 0;

  /**
   * Method to build a representation at a particular level. Note that the
   * method GetLevel() returns the maximum number of levels available for
   * the tree. You must provide a svtkPolyData object into which to place the
   * data.
   */
  virtual void GenerateRepresentation(int level, svtkPolyData* pd) = 0;

  //@{
  /**
   * Return the time of the last data structure build.
   */
  svtkGetMacro(BuildTime, svtkMTimeType);
  //@}

  //@{
  /**
   * Handle the PointSet <-> Locator loop.
   */
  void Register(svtkObjectBase* o) override;
  void UnRegister(svtkObjectBase* o) override;
  //@}

protected:
  svtkLocator();
  ~svtkLocator() override;

  svtkDataSet* DataSet;
  svtkTypeBool Automatic; // boolean controls automatic subdivision (or uses user spec.)
  double Tolerance;      // for performing merging
  int MaxLevel;
  int Level;

  svtkTimeStamp BuildTime; // time at which locator was built

  void ReportReferences(svtkGarbageCollector*) override;

private:
  svtkLocator(const svtkLocator&) = delete;
  void operator=(const svtkLocator&) = delete;
};

#endif
