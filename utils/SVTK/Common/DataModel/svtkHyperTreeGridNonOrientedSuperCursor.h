/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHyperTreeGridNonOrientedSuperCursor.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright Nonice for more information.

=========================================================================*/
/**
 * @class   svtkHyperTreeGridNonOrientedSuperCursor
 * @brief   Objects for traversal a HyperTreeGrid.
 *
 * JB A REVOIR
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
 * This class was re-written and optimized by Jacques-Bernard Lekien,
 * Guenole Harel and Jerome Dubois, 2018.
 * This work was supported by Commissariat a l'Energie Atomique
 * CEA, DAM, DIF, F-91297 Arpajon, France.
 */

#ifndef svtkHyperTreeGridNonOrientedSuperCursor_h
#define svtkHyperTreeGridNonOrientedSuperCursor_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"
#include "svtkSmartPointer.h" // Used internally

#include "svtkHyperTreeGridGeometryLevelEntry.h" // Used Internally

#include <vector> // For std::vector

class svtkHyperTree;
class svtkHyperTreeGrid;
class svtkHyperTreeGridNonOrientedGeometryCursor;
class svtkHyperTreeGridOrientedGeometryCursor;

class SVTKCOMMONDATAMODEL_EXPORT svtkHyperTreeGridNonOrientedSuperCursor : public svtkObject
{
public:
  svtkTypeMacro(svtkHyperTreeGridNonOrientedSuperCursor, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Create a copy of `this'.
   * \post results_exists:result!=0
   */
  virtual svtkHyperTreeGridNonOrientedSuperCursor* Clone();

  /**
   * Initialize cursor at root of given tree index in grid.
   * JB Le create ne s'applique que sur le HT central.
   */
  virtual void Initialize(svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create = false) = 0;

  //@{
  /**
   * Set the hyper tree grid to which the cursor is pointing.
   */
  svtkHyperTreeGrid* GetGrid();
  //@}

  //@{
  /**
   * Return if a Tree pointing exist
   */
  bool HasTree();
  //@}

  /**
   * JB Return if a Tree pointing exist
   */
  bool HasTree(unsigned int icursor);

  //@{
  /**
   * Set the hyper tree to which the cursor is pointing.
   */
  svtkHyperTree* GetTree();
  svtkHyperTree* GetTree(unsigned int icursor);
  //@}

  /**
   * Return the index of the current vertex in the tree.
   */
  svtkIdType GetVertexId();
  svtkIdType GetVertexId(unsigned int icursor);

  /**
   * Return the global index (relative to the grid) of the
   * current vertex in the tree.
   */
  svtkIdType GetGlobalNodeIndex();

  /**
   * JB Return the global index (relative to the grid) of the
   * neighboor icursor current vertex in the tree.
   */
  svtkIdType GetGlobalNodeIndex(unsigned int icursor);

  /**
   * JB
   */
  svtkHyperTree* GetInformation(
    unsigned int icursor, unsigned int& level, bool& leaf, svtkIdType& id);

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
   * Set the blanking mask is empty or not
   * \pre not_tree: tree
   */
  void SetMask(bool state);
  void SetMask(unsigned int icursor, bool state);

  /**
   * Determine whether blanking mask is empty or not
   */
  bool IsMasked();
  bool IsMasked(unsigned int icursor);

  /**
   * JB Coordonnees de la boite englobante
   */
  void GetBounds(double bounds[6]);
  void GetBounds(unsigned int icursor, double bounds[6]);

  /**
   * JB Coordonnees du centre de la maille
   */
  void GetPoint(double point[3]);
  void GetPoint(unsigned int icursor, double point[3]);

  /**
   * Is the cursor pointing to a leaf?
   */
  bool IsLeaf();
  bool IsLeaf(unsigned int icursor);

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
  unsigned int GetLevel(unsigned int icursor);

  /**
   * Move the cursor to child `child' of the current vertex.
   * \pre not_tree: HasTree()
   * \pre not_leaf: !IsLeaf()
   * \pre valid_child: ichild>=0 && ichild<GetNumberOfChildren()
   * \pre depth_limiter: GetLevel() <= GetDepthLimiter()
   */
  void ToChild(unsigned char ichild);

  /**
   * Move the cursor to the root vertex.
   * \pre can be root
   * \post is_root: IsRoot()
   */
  void ToRoot();

  /**
   * Move the cursor to the parent of the current vertex.
   * Authorized if HasHistory return true.
   * \pre Non_root: !IsRoot()
   */
  void ToParent();

  /**
   * JB
   */
  unsigned int GetNumberOfCursors() { return this->NumberOfCursors; }

  /**
   * Return the cursor pointing into i-th neighbor.
   * The neighborhood definition depends on the type of cursor.
   * NB: Only super cursors keep track of neighborhoods.
   */
  svtkSmartPointer<svtkHyperTreeGridOrientedGeometryCursor> GetOrientedGeometryCursor(
    unsigned int icursor);

  /**
   * Return the cursor pointing into i-th neighbor.
   * The neighborhood definition depends on the type of cursor.
   * NB: Only super cursors keep track of neighborhoods.
   */
  svtkSmartPointer<svtkHyperTreeGridNonOrientedGeometryCursor> GetNonOrientedGeometryCursor(
    unsigned int icursor);

protected:
  /**
   * Constructor
   */
  svtkHyperTreeGridNonOrientedSuperCursor();

  /**
   * Destructor
   */
  ~svtkHyperTreeGridNonOrientedSuperCursor() override;

  /**
   * JB Reference sur l'hyper tree grid parcouru actuellement.
   */
  svtkHyperTreeGrid* Grid;

  /**
   * JB
   */
  // JB  svtkNew< svtkHyperTreeGridNonOrientedGeometryCursor > CentralCursor;
  svtkSmartPointer<svtkHyperTreeGridNonOrientedGeometryCursor> CentralCursor;

  /**
   * JB Hyper tree grid to which the cursor is attached
   */
  unsigned int CurrentFirstNonValidEntryByLevel;
  std::vector<unsigned int> FirstNonValidEntryByLevel;
  std::vector<svtkHyperTreeGridGeometryLevelEntry> Entries;

  /**
   * JB La derniere reference valide pour decrire tous les voisins.
   * C'est donc aussi l'offset du premier voisin du dernier niveau.
   */
  unsigned int FirstCurrentNeighboorReferenceEntry;
  std::vector<unsigned int> ReferenceEntries;

  /**
   * JB
   */
  unsigned int GetIndiceEntry(unsigned int icursor);

  /**
   * JB La valeur precedente. Dans le voisinage, ce n'est pas forcement un parent.
   */
  unsigned int GetIndicePreviousEntry(unsigned int icursor);

  /**
   * JB
   */
  unsigned int IndiceCentralCursor;

  // Number of cursors in supercursor
  unsigned int NumberOfCursors;

  // Super cursor traversal table to go retrieve the parent index for each cursor
  // of the child node. There are f^d * NumberOfCursors entries in the table.
  const unsigned int* ChildCursorToParentCursorTable;

  // Super cursor traversal table to go retrieve the child index for each cursor
  // of the child node. There are f^d * NumberOfCursors entries in the table.
  const unsigned int* ChildCursorToChildTable;

private:
  svtkHyperTreeGridNonOrientedSuperCursor(const svtkHyperTreeGridNonOrientedSuperCursor&) = delete;
  void operator=(const svtkHyperTreeGridNonOrientedSuperCursor&) = delete;
};

#endif
