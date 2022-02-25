/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHyperTreeGridEntry.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkHyperTreeGridEntry
 * @brief   Entries are cache data for cursors
 *
 * Entries are relevant for cursor/supercursor developers. Filters
 * developers should have a look at cursors/supercursors documentation.
 * (cf. svtkHyperTreeGridNonOrientedCursor). When writing a new cursor or
 * supercursor the choice of the entry is very important: it will drive
 * the performance and memory cost. This is even more important for
 * supercursors which have several neighbors: 6x for VonNeuman and 26x for
 * Moore.
 *
 * Several types of Entries exist:
 * 1. svtkHyperTreeGridEntry
 * This cache only memorizes the current cell index in one HyperTree.
 * Using the index, this entry provides several services such as:
 * is the cell coarse or leaf, get or set global index (to access
 * field value, cf. svtkHyperTree), descend into selected child,
 * subdivise the cell. Equivalent services are available for all entries.
 *
 * 2. svtkHyperTreeGridGeometryEntry
 * This cache adds the origin coordinates of the cell atop
 * svtkHyperTreeGridEntry. Getter is provided, as well as services related
 * to the bounding box and cell center.
 *
 * 3. svtkHyperTreeGridLevelEntry
 * This cache adds the following information with their getters atop
 * svtkHyperTreeGridEntry: pointer to the HyperTree, level of the current
 * cell.
 *
 * 4. svtkHyperTreeGridGeometryLevelEntry
 * This cache is a combination of svtkHyperTreeGridLevelEntry and
 * svtkHyperTreeGridLevelEntry: it provides all combined services.
 *
 * @sa
 * svtkHyperTreeGridEntry
 * svtkHyperTreeGridLevelEntry
 * svtkHyperTreeGridGeometryEntry
 * svtkHyperTreeGridGeometryLevelEntry
 * svtkHyperTreeGridOrientedCursor
 * svtkHyperTreeGridNonOrientedCursor
 *
 * @par Thanks:
 * This class was written by Jacques-Bernard Lekien, Jerome Dubois and
 * Guenole Harel, CEA 2018.
 * This work was supported by Commissariat a l'Energie Atomique
 * CEA, DAM, DIF, F-91297 Arpajon, France.
 */

#ifndef svtkHyperTreeGridEntry_h
#define svtkHyperTreeGridEntry_h

#ifndef __SVTK_WRAP__

#include "svtkObject.h"

class svtkHyperTree;
class svtkHyperTreeGrid;

class svtkHyperTreeGridEntry
{
public:
  /**
   * Display info about the entry
   */
  void PrintSelf(ostream& os, svtkIndent indent);

  /**
   * Dump information
   */
  void Dump(ostream& os);

  /**
   * Constructor
   */
  svtkHyperTreeGridEntry() { this->Index = 0; }

  /**
   * Constructor
   */
  svtkHyperTreeGridEntry(svtkIdType index) { this->Index = index; }

  /**
   * Destructor
   */
  ~svtkHyperTreeGridEntry() = default;

  /**
   * Initialize cursor at root of given tree index in grid.
   */
  svtkHyperTree* Initialize(svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create = false);

  /**
   * Initialize cursor at root of given tree index in grid.
   */
  void Initialize(svtkIdType index) { this->Index = index; }

  /**
   * Copy function
   */
  void Copy(const svtkHyperTreeGridEntry* entry) { this->Index = entry->Index; }

  /**
   * Return the index of the current vertex in the tree.
   */
  svtkIdType GetVertexId() const { return this->Index; }

  /**
   * Return the global index for the current cell (cf. svtkHyperTree).
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
   * Is the cursor pointing to a coarse with all childrens being leaves?
   * \pre not_tree: tree
   */
  bool IsTerminalNode(
    const svtkHyperTreeGrid* grid, const svtkHyperTree* tree, unsigned int level) const;

  /**
   * Is the cursor at HyperTree root?
   */
  bool IsRoot() const { return (this->Index == 0); }

  /**
   * Move the cursor to i-th child of the current cell.
   * \pre not_tree: tree
   * \pre not_leaf: !IsLeaf()
   * \pre valid_child: ichild>=0 && ichild<this->GetNumberOfChildren()
   * \pre depth_limiter: level == grid->GetDepthLimiter()
   * \pre is_masked: !IsMasked()
   */
  void ToChild(const svtkHyperTreeGrid* grid, const svtkHyperTree* tree, unsigned int level,
    unsigned char ichild);

protected:
  /**
   * index of the current cell in the HyperTree.
   */
  svtkIdType Index;
};

#endif // __SVTK_WRAP__

#endif // svtkHyperTreeGridEntry_h
// SVTK-HeaderTest-Exclude: svtkHyperTreeGridEntry.h
