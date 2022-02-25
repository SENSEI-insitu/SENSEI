/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHyperTreeGridNonOrientedMooreSuperCursorLight.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright Nonice for more information.

=========================================================================*/
/**
 * @class   svtkHyperTreeGridNonOrientedMooreSuperCursorLight
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

#ifndef svtkHyperTreeGridNonOrientedMooreSuperCursorLight_h
#define svtkHyperTreeGridNonOrientedMooreSuperCursorLight_h

#include "svtkCommonDataModelModule.h" // For export macro

#include "svtkHyperTreeGridNonOrientedSuperCursorLight.h"

class svtkIdList;
class svtkHyperTree;
class svtkHyperTreeGrid;

class SVTKCOMMONDATAMODEL_EXPORT svtkHyperTreeGridNonOrientedMooreSuperCursorLight
  : public svtkHyperTreeGridNonOrientedSuperCursorLight
{
public:
  svtkTypeMacro(
    svtkHyperTreeGridNonOrientedMooreSuperCursorLight, svtkHyperTreeGridNonOrientedSuperCursorLight);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  static svtkHyperTreeGridNonOrientedMooreSuperCursorLight* New();

  /**
   * Initialize cursor at root of given tree index in grid.
   * JB Le create ne s'applique que sur le HT central.
   */
  void Initialize(svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create = false) override;

  /**
   * Return the list of cursors pointing to the leaves touching a
   * given corner of the cell.
   * Return whether the considered cell is the owner of said corner.
   * JB Utilise aujourd'hui dans les filtres svtkHyperTreeGridContour et svtkHyperTreeGridPlaneCutter.
   */
  bool GetCornerCursors(unsigned int, unsigned int, svtkIdList*);

protected:
  /**
   * Constructor
   */
  svtkHyperTreeGridNonOrientedMooreSuperCursorLight(){};

  /**
   * Destructor
   */
  ~svtkHyperTreeGridNonOrientedMooreSuperCursorLight() override;

private:
  svtkHyperTreeGridNonOrientedMooreSuperCursorLight(
    const svtkHyperTreeGridNonOrientedMooreSuperCursorLight&) = delete;
  void operator=(const svtkHyperTreeGridNonOrientedMooreSuperCursorLight&) = delete;
};

#endif
