/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHyperTreeGridGeometryEntry.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkHyperTreeGridGeometryEntry
JB
 * @brief   GeometryEntry is a cache data for cursors requiring coordinates
 *
 * cf. svtkHyperTreeGridEntry
 *
 * @sa
 * svtkHyperTreeGridEntry
 * svtkHyperTreeGridLevelEntry
 * svtkHyperTreeGridGeometryEntry
 * svtkHyperTreeGridGeometryLevelEntry
 * svtkHyperTreeGridNonOrientedGeometryCursor
 * svtkHyperTreeGridNonOrientedSuperCursor
 * svtkHyperTreeGridNonOrientedSuperCursorLight
 *
 * @par Thanks:
 * This class was written by Jacques-Bernard Lekien, Jerome Dubois and
 * Guenole Harel, CEA 2018.
 * This work was supported by Commissariat a l'Energie Atomique
 * CEA, DAM, DIF, F-91297 Arpajon, France.
 */

#ifndef svtkHyperTreeGridGeometryEntry_h
#define svtkHyperTreeGridGeometryEntry_h

#ifndef __SVTK_WRAP__

#include "svtkObject.h"

class svtkHyperTree;
class svtkHyperTreeGrid;

class svtkHyperTreeGridGeometryEntry
{
public:
  /**
   * Display info about the entry
   */
  void PrintSelf(ostream& os, svtkIndent indent);

  /**
   * Constructor
   */
  svtkHyperTreeGridGeometryEntry();

  /**
   * Constructor
   */
  svtkHyperTreeGridGeometryEntry(svtkIdType index, const double* origin)
  {
    this->Index = index;
    for (unsigned int d = 0; d < 3; ++d)
    {
      this->Origin[d] = origin[d];
    }
  }

  /**
   * Destructor
   */
  ~svtkHyperTreeGridGeometryEntry() = default;

  /**
   * Dump information
   */
  void Dump(ostream& os);

  /**
   * Initialize cursor at root of given tree index in grid.
   */
  svtkHyperTree* Initialize(svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create = false);

  /**
   * Initialize cursor from explicit required data
   */
  void Initialize(svtkIdType index, const double* origin)
  {
    this->Index = index;
    for (unsigned int d = 0; d < 3; ++d)
    {
      this->Origin[d] = origin[d];
    }
  }

  /**
   * Copy function
   */
  void Copy(const svtkHyperTreeGridGeometryEntry* entry)
  {
    this->Index = entry->Index;
    for (unsigned int d = 0; d < 3; ++d)
    {
      this->Origin[d] = entry->Origin[d];
    }
  }

  /**
   * Return the index of the current vertex in the tree.
   */
  svtkIdType GetVertexId() const { return this->Index; }

  /**
   * Return the global index (relative to the grid) of the
   * current vertex in the tree.
   * \pre not_tree: tree
   */
  svtkIdType GetGlobalNodeIndex(const svtkHyperTree* tree) const;

  /**
   * Set the global index for the root cell of the HyperTree.
   * \pre not_tree: tree
   */
  void SetGlobalIndexStart(svtkHyperTree* tree, svtkIdType index);

  /**
   * Set the global index for the current cell of the HyperTree.
   * \pre not_tree: tree
   */
  void SetGlobalIndexFromLocal(svtkHyperTree* tree, svtkIdType index);

  /**
   * Set the blanking mask is empty or not
   * \pre not_tree: tree
   */
  void SetMask(const svtkHyperTreeGrid* grid, const svtkHyperTree* tree, bool state);

  /**
   * Determine whether blanking mask is empty or not
   * \pre not_tree: tree
   */
  bool IsMasked(const svtkHyperTreeGrid* grid, const svtkHyperTree* tree) const;

  /**
   * Is the cursor pointing to a leaf?
   * \pre not_tree: tree
   * Return true if level == grid->GetDepthLimiter()
   */
  bool IsLeaf(const svtkHyperTreeGrid* grid, const svtkHyperTree* tree, unsigned int level) const;

  /**
   * Change the current cell's status: if leaf then becomes coarse and
   * all its children are created, cf. HyperTree.
   * \pre not_tree: tree
   * \pre depth_limiter: level == grid->GetDepthLimiter()
   * \pre is_masked: IsMasked
   */
  void SubdivideLeaf(const svtkHyperTreeGrid* grid, svtkHyperTree* tree, unsigned int level);

  /**
   * Is the cursor pointing to a coarse with all childrens leaves ?
   * \pre not_tree: tree
   */
  bool IsTerminalNode(
    const svtkHyperTreeGrid* grid, const svtkHyperTree* tree, unsigned int level) const;

  /**
   * Is the cursor at tree root?
   */
  bool IsRoot() const { return (this->Index == 0); }

  /**
   * Move the cursor to child `child' of the current vertex.
   * \pre not_tree: tree
   * \pre not_leaf: !IsLeaf()
   * \pre valid_child: ichild>=0 && ichild<this->GetNumberOfChildren()
   * \pre depth_limiter: level == grid->GetDepthLimiter()
   * \pre is_masked: !IsMasked()
   */
  void ToChild(const svtkHyperTreeGrid* grid, const svtkHyperTree* tree, unsigned int level,
    const double* sizeChild, unsigned char ichild);

  /**
   * Getter for origin coordinates of the current cell.
   */
  double* GetOrigin() { return this->Origin; }
  const double* GetOrigin() const { return this->Origin; }

  /**
   * Getter for bounding box of the current cell.
   */
  void GetBounds(const double* sizeChild, double bounds[6]) const
  {
    // Compute bounds
    bounds[0] = this->Origin[0];
    bounds[1] = this->Origin[0] + sizeChild[0];
    bounds[2] = this->Origin[1];
    bounds[3] = this->Origin[1] + sizeChild[1];
    bounds[4] = this->Origin[2];
    bounds[5] = this->Origin[2] + sizeChild[2];
  }

  /**
   * Getter for center of the current cell.
   */
  void GetPoint(const double* sizeChild, double point[3]) const
  {
    // Compute center point coordinates
    point[0] = this->Origin[0] + sizeChild[0] / 2.;
    point[1] = this->Origin[1] + sizeChild[1] / 2.;
    point[2] = this->Origin[2] + sizeChild[2] / 2.;
  }

private:
  /**
   * index of the current cell in the HyperTree.
   */
  svtkIdType Index;

  /**
   * origin coiordinates of the current cell
   */
  double Origin[3];
};

#endif // __SVTK_WRAP__

#endif // svtkHyperTreeGridGeometryEntry_h
// SVTK-HeaderTest-Exclude: svtkHyperTreeGridGeometryEntry.h
