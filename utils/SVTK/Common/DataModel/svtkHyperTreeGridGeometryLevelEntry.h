/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHyperTreeGridGeometryLevelEntry.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkHyperTreeGridGeometryLevelEntry
 *
 * @brief   Cursor cache data with coordinates and level info
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

#ifndef svtkHyperTreeGridGeometryLevelEntry_h
#define svtkHyperTreeGridGeometryLevelEntry_h

#ifndef __SVTK_WRAP__

#include "assert.h"

#include "svtkObject.h"
#include "svtkSmartPointer.h"

#include "svtkHyperTreeGridNonOrientedGeometryCursor.h"
#include "svtkHyperTreeGridOrientedGeometryCursor.h"

class svtkHyperTree;
class svtkHyperTreeGrid;

class svtkHyperTreeGridGeometryLevelEntry
{
public:
  /**
   * Display info about the entry
   */
  void PrintSelf(ostream& os, svtkIndent indent);

  /**
   * Constructor
   */
  svtkHyperTreeGridGeometryLevelEntry()
  {
    this->Tree = nullptr;
    this->Level = 0;
    this->Index = 0;
    for (unsigned int d = 0; d < 3; ++d)
    {
      this->Origin[d] = 0.;
    }
  }

  /**
   * Destructor
   */
  ~svtkHyperTreeGridGeometryLevelEntry() = default;

  /**
   * Dump information
   */
  void Dump(ostream& os);

  /**
   * Initialize cache entry from explicit required data
   */
  void Initialize(svtkHyperTree* tree, unsigned int level, svtkIdType index, const double* origin)
  {
    this->Tree = tree;
    this->Level = level;
    this->Index = index;
    for (unsigned int d = 0; d < 3; ++d)
    {
      this->Origin[d] = origin[d];
    }
  }

  /**
   * Initialize cache entry at root of given tree index in grid.
   */
  svtkHyperTree* Initialize(svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create = false);

  /**
   * Reset function
   */
  void Reset()
  {
    this->Tree = nullptr;
    this->Index = 0;
  }

  /**
   * Copy function
   */
  void Copy(const svtkHyperTreeGridGeometryLevelEntry* entry)
  {
    this->Initialize(entry->Tree, entry->Level, entry->Index, entry->Origin);
  }

  /**
   * Create a svtkHyperTreeGridOrientedCursor from input grid and
   * current entry data.
   */
  svtkSmartPointer<svtkHyperTreeGridOrientedGeometryCursor> GetHyperTreeGridOrientedGeometryCursor(
    svtkHyperTreeGrid* grid)
  {
    svtkSmartPointer<svtkHyperTreeGridOrientedGeometryCursor> cursor =
      svtkSmartPointer<svtkHyperTreeGridOrientedGeometryCursor>::New();
    cursor->Initialize(grid, this->Tree, this->Level, this->Index, this->Origin);
    return cursor;
  }

  /**
   * Create a svtkHyperTreeGridNonOrientedCursor from input grid and
   * current entry data.
   */
  svtkSmartPointer<svtkHyperTreeGridNonOrientedGeometryCursor>
  GetHyperTreeGridNonOrientedGeometryCursor(svtkHyperTreeGrid* grid)
  {
    assert("pre: level==0" && this->Level == 0);
    svtkSmartPointer<svtkHyperTreeGridNonOrientedGeometryCursor> cursor =
      svtkSmartPointer<svtkHyperTreeGridNonOrientedGeometryCursor>::New();
    cursor->Initialize(grid, this->Tree, this->Level, this->Index, this->Origin);
    return cursor;
  }

  /**
   * Return the index of the current vertex in the tree.
   * \pre not_tree: tree
   */
  svtkIdType GetVertexId() const { return this->Index; }

  /**
   * Return the global index (relative to the grid) of the
   * current vertex in the tree.
   * \pre not_tree: tree
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
  bool IsRoot() { return (this->Index == 0); }

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

  /**
   * Getter for origin coordinates of the current cell.
   */
  double* GetOrigin() { return this->Origin; }
  const double* GetOrigin() const { return this->Origin; }

  /**
   * Getter for bounding box of the current cell.
   */
  void GetBounds(double bounds[6]) const;

  /**
   * Getter for center of the current cell.
   */
  void GetPoint(double point[3]) const;

private:
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

  /**
   * origin coiordinates of the current cell
   */
  double Origin[3];
};

#endif // __SVTK_WRAP__

#endif // svtkHyperTreeGridGeometryLevelEntry
// SVTK-HeaderTest-Exclude: svtkHyperTreeGridGeometryLevelEntry.h
