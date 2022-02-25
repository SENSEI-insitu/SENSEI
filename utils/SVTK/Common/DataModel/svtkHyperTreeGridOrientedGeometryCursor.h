/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHyperTreeGridOrientedGeometryCursor.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright Nonice for more information.

=========================================================================*/
/**
 * @class   svtkHyperTreeGridOrientedGeometryCursor
 * @brief   Objects for traversal a HyperTreeGrid.
 *
 * JB A REVOIR
 * NonOriented ne peut pas remonter plus haut qu'a sa creation.
 * Objects that can perform depth traversal of a hyper tree grid,
 * take into account more parameters (related to the grid structure) than
 * the compact hyper tree cursor implemented in svtkHyperTree can.
 * This is an abstract class.
 * Cursors are created by the HyperTreeGrid implementation.
 *
 * @sa
 * svtkHyperTreeCursor svtkHyperTree svtkHyperTreeGrid
 *
 * @par Thanks:
 * This class was written by Guenole Harel and Jacques-Bernard Lekien, 2014.
 * This class was re-written by Philippe Pebay, 2016.
 * JB This class was re-written for more optimisation by Jacques-Bernard Lekien,
 * Guenole Harel and Jerome Dubois, 2018.
 * This work was supported by Commissariat a l'Energie Atomique
 * CEA, DAM, DIF, F-91297 Arpajon, France.
 */

#ifndef svtkHyperTreeGridOrientedGeometryCursor_h
#define svtkHyperTreeGridOrientedGeometryCursor_h

#include "svtkCommonDataModelModule.h"      // For export macro
#include "svtkHyperTreeGridGeometryEntry.h" // Used internally
#include "svtkHyperTreeGridTools.h"         // for HasTree
#include "svtkObject.h"

#include <memory> // For std::shared_ptr
#include <vector> // For std::vector

class svtkHyperTree;
class svtkHyperTreeGrid;
class svtkHyperTreeGridScales;

class SVTKCOMMONDATAMODEL_EXPORT svtkHyperTreeGridOrientedGeometryCursor : public svtkObject
{
public:
  svtkTypeMacro(svtkHyperTreeGridOrientedGeometryCursor, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  static svtkHyperTreeGridOrientedGeometryCursor* New();

  void Dump(ostream& os);

  /**
   * Create a copy of `this'.
   * \post results_exists:result!=0
   */
  virtual svtkHyperTreeGridOrientedGeometryCursor* Clone();

  /**
   * Initialize cursor at root of given tree index in grid.
   */
  void Initialize(svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create = false);

  /**
   * Initialize cursor at root of given tree index in grid.
   */
  void Initialize(svtkHyperTreeGrid* grid, svtkHyperTree* tree, unsigned int level,
    svtkHyperTreeGridGeometryEntry& entry);

  /**
   * JB
   */
  void Initialize(svtkHyperTreeGrid* grid, svtkHyperTree* tree, unsigned int level, svtkIdType index,
    double* origin);

  /**
   * JB
   */
  void Initialize(svtkHyperTreeGridOrientedGeometryCursor* cursor);

  //@{
  /**
   * Return if a Tree pointing exist
   */
  bool HasTree() const { return svtk::hypertreegrid::HasTree(*this); }
  //@}

  //@{
  /**
   * Set the hyper tree to which the cursor is pointing.
   */
  svtkHyperTree* GetTree() const { return this->Tree; }
  //@}

  /**
   * Return the index of the current vertex in the tree.
   */
  svtkIdType GetVertexId();

  /**
   * Return the global index (relative to the grid) of the
   * current vertex in the tree.
   */
  svtkIdType GetGlobalNodeIndex();

  /**
   * Return the dimension of the tree.
   * \post positive_result: result>0
   */
  unsigned char GetDimension();

  /**
   * Return the number of children for each node (non-vertex leaf) of the tree.
   * \post positive_number: result>0
   */
  unsigned char GetNumberOfChildren();

  /**
   * JB
   */
  void SetGlobalIndexStart(svtkIdType index);

  /**
   * JB
   */
  void SetGlobalIndexFromLocal(svtkIdType index);

  /**
   * JB
   */
  double* GetOrigin();
  double* GetSize();

  /**
   * JB
   */
  void GetBounds(double bounds[6]);
  void GetPoint(double point[3]);

  /**
   * Set the blanking mask is empty or not
   * \pre not_tree: tree
   */
  void SetMask(bool state);

  /**
   * Determine whether blanking mask is empty or not
   */
  bool IsMasked();

  /**
   * Is the cursor pointing to a leaf?
   */
  bool IsLeaf();

  /**
   * JB Fait chier normalement on devrait passer par GetEntry
   */
  void SubdivideLeaf();

  /**
   * Is the cursor at tree root?
   */
  bool IsRoot();

  /**
   * Get the level of the tree vertex pointed by the cursor.
   */
  unsigned int GetLevel();

  /**
   * Move the cursor to child `child' of the current vertex.
   * \pre not_tree: HasTree()
   * \pre not_leaf: !IsLeaf()
   * \pre valid_child: ichild>=0 && ichild<GetNumberOfChildren()
   * \pre depth_limiter: GetLevel() <= GetDepthLimiter()
   */
  void ToChild(unsigned char ichild);

protected:
  /**
   * Constructor
   * JB Just pour svtkHyperTreeGridNonOrientedVonNeumannSuperCursor et Moore
   */
  svtkHyperTreeGridOrientedGeometryCursor();

  /**
   * Destructor
   * JB Just pour svtkHyperTreeGridNonOrientedVonNeumannSuperCursor et Moore
   */
  ~svtkHyperTreeGridOrientedGeometryCursor() override;

  /**
   * JB Reference sur l'hyper tree grid parcouru actuellement.
   */
  svtkHyperTreeGrid* Grid;

  /**
   * JB
   */
  svtkHyperTree* Tree;

  /**
   * JB Storage of pre-computed per-level cell scales
   */
  std::shared_ptr<svtkHyperTreeGridScales> Scales;

  /**
   * JB
   */
  unsigned int Level;

  // Hyper tree grid to which the cursor is attached
  svtkHyperTreeGridGeometryEntry Entry;

private:
  svtkHyperTreeGridOrientedGeometryCursor(const svtkHyperTreeGridOrientedGeometryCursor&) = delete;
  void operator=(const svtkHyperTreeGridOrientedGeometryCursor&) = delete;
};
#endif
