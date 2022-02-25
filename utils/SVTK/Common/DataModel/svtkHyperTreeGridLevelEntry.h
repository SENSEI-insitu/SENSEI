/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHyperTreeGridLevelEntry.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkHyperTreeGridLevelEntry
 *
 * @brief   LevelEntry is a cache data for cursors requiring level info
 *
 * cf. svtkHyperTreeGridEntry
 *
 * @sa
 * svtkHyperTreeGridEntry
 * svtkHyperTreeGridLevelEntry
 * svtkHyperTreeGridGeometryEntry
 * svtkHyperTreeGridGeometryLevelEntry
 * svtkHyperTreeGridNonOrientedSuperCursor
 * svtkHyperTreeGridNonOrientedSuperCursorLight
 *
 * @par Thanks:
 * This class was written by Jacques-Bernard Lekien, Jerome Dubois and
 * Guenole Harel, CEA 2018.
 * This work was supported by Commissariat a l'Energie Atomique
 * CEA, DAM, DIF, F-91297 Arpajon, France.
 */

#ifndef svtkHyperTreeGridLevelEntry_h
#define svtkHyperTreeGridLevelEntry_h

#ifndef __SVTK_WRAP__

#include "svtkObject.h"
#include "svtkSmartPointer.h" // Used internally

class svtkHyperTree;
class svtkHyperTreeGrid;
class svtkHyperTreeGridNonOrientedCursor;

class svtkHyperTreeGridLevelEntry
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
  svtkHyperTreeGridLevelEntry()
    : Tree(nullptr)
    , Level(0)
    , Index(0)
  {
  }

  /**
   * Constructor
   */
  svtkHyperTreeGridLevelEntry(svtkHyperTreeGridLevelEntry* entry)
    : Tree(entry->Tree)
    , Level(entry->Level)
    , Index(entry->Index)
  {
  }

  /**
   * Constructor
   */
  svtkHyperTreeGridLevelEntry(svtkHyperTree* tree, unsigned int level, svtkIdType index)
    : Tree(tree)
    , Level(level)
    , Index(index)
  {
  }

  /**
   * Constructor
   */
  svtkHyperTreeGridLevelEntry(svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create = false);

  /**
   * Destructor
   */
  ~svtkHyperTreeGridLevelEntry() = default;

  /**
   * Reset function
   */
  void Reset()
  {
    this->Tree = nullptr;
    this->Level = 0;
    this->Index = 0;
  }

  /**
   * Initialize cursor from explicit required data
   */
  void Initialize(svtkHyperTree* tree, unsigned int level, svtkIdType index)
  {
    this->Tree = tree;
    this->Level = level;
    this->Index = index;
  }

  /**
   * Initialize cursor at root of given tree index in grid.
   */
  svtkHyperTree* Initialize(svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create = false);

  /**
   * Copy function
   */
  void Copy(const svtkHyperTreeGridLevelEntry* entry)
  {
    this->Tree = entry->Tree;
    this->Level = entry->Level;
    this->Index = entry->Index;
  }

  /**
   * Create a svtkHyperTreeGridNonOrientedCursor from input grid and
   * current entry data
   */
  svtkSmartPointer<svtkHyperTreeGridNonOrientedCursor> GetHyperTreeGridNonOrientedCursor(
    svtkHyperTreeGrid* grid);

  /**
   * Return the index of the current vertex in the tree.
   */
  svtkIdType GetVertexId() const { return this->Index; }

  /**
   * Return the global index (relative to the grid) of the
   * current vertex in the tree.
   */
  svtkIdType GetGlobalNodeIndex() const;

  /**
   * Set the global index for the root cell of the HyperTree.
   * \pre not_tree: tree
   */
  void SetGlobalIndexStart(svtkIdType index);

  /**
   * Set the global index for the current cell of the HyperTree.
   * \pre not_tree: tree
   */
  void SetGlobalIndexFromLocal(svtkIdType index);

  /**
   * Set the blanking mask is empty or not
   * \pre not_tree: tree
   */
  void SetMask(const svtkHyperTreeGrid* grid, bool state);

  /**
   * Determine whether blanking mask is empty or not
   * \pre not_tree: tree
   */
  bool IsMasked(const svtkHyperTreeGrid* grid) const;

  /**
   * Is the cursor pointing to a leaf?
   * \pre not_tree: tree
   * Return true if level == grid->GetDepthLimiter()
   */
  bool IsLeaf(const svtkHyperTreeGrid* grid) const;

  /**
   * Change the current cell's status: if leaf then becomes coarse and
   * all its children are created, cf. HyperTree.
   * \pre not_tree: tree
   * \pre depth_limiter: level == grid->GetDepthLimiter()
   * \pre is_masked: IsMasked
   */
  void SubdivideLeaf(const svtkHyperTreeGrid* grid);

  /**
   * Is the cursor pointing to a coarse with all childrens being leaves ?
   * \pre not_tree: tree
   */
  bool IsTerminalNode(const svtkHyperTreeGrid* grid) const;

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
  void ToChild(const svtkHyperTreeGrid* grid, unsigned char ichild);

  /**
   * Get HyperTree from current cache entry.
   */
  svtkHyperTree* GetTree() const { return this->Tree; }

  /**
   * Get level info from current cache entry.
   */
  unsigned int GetLevel() const { return this->Level; }

protected:
  /**
   * pointer to the HyperTree containing the current cell.
   */
  svtkHyperTree* Tree;

  /**
   * level of the current cell in the HyperTree.
   */
  unsigned int Level;

  /**
   * index of the current cell in the HyperTree.
   */
  svtkIdType Index;
};

#endif // __SVTK_WRAP__

#endif // svtkHyperTreeGridLevelEntry_h
// SVTK-HeaderTest-Exclude: svtkHyperTreeGridLevelEntry.h
