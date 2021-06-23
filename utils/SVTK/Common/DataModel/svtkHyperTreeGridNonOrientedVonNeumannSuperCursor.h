/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHyperTreeGridNonOrientedVonNeumannSuperCursor.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright Nonice for more information.

=========================================================================*/
/**
 * @class   svtkHyperTreeGridNonOrientedVonNeumannSuperCursor
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

#ifndef svtkHyperTreeGridNonOrientedVonNeumannSuperCursor_h
#define svtkHyperTreeGridNonOrientedVonNeumannSuperCursor_h

#include "svtkHyperTreeGridNonOrientedSuperCursor.h"

class svtkHyperTreeGrid;

class SVTKCOMMONDATAMODEL_EXPORT svtkHyperTreeGridNonOrientedVonNeumannSuperCursor
  : public svtkHyperTreeGridNonOrientedSuperCursor
{
public:
  svtkTypeMacro(
    svtkHyperTreeGridNonOrientedVonNeumannSuperCursor, svtkHyperTreeGridNonOrientedSuperCursor);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  static svtkHyperTreeGridNonOrientedVonNeumannSuperCursor* New();

  /**
   * Initialize cursor at root of given tree index in grid.
   * JB Le create ne s'applique que sur le HT central.
   */
  void Initialize(svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create = false) override;

protected:
  /**
   * Constructor
   */
  svtkHyperTreeGridNonOrientedVonNeumannSuperCursor() {}

  /**
   * Destructor
   */
  ~svtkHyperTreeGridNonOrientedVonNeumannSuperCursor() override;

private:
  svtkHyperTreeGridNonOrientedVonNeumannSuperCursor(
    const svtkHyperTreeGridNonOrientedVonNeumannSuperCursor&) = delete;
  void operator=(const svtkHyperTreeGridNonOrientedVonNeumannSuperCursor&) = delete;
};

#endif
