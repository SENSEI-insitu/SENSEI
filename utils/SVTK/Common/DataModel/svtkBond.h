/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBond.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkBond
 * @brief   convenience proxy for svtkMolecule
 *
 */

#ifndef svtkBond_h
#define svtkBond_h

#include "svtkAtom.h"                  // For svtkAtom
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"                // For macros, etc

class svtkMolecule;

class SVTKCOMMONDATAMODEL_EXPORT svtkBond
{
public:
  void PrintSelf(ostream& os, svtkIndent indent);

  /**
   * Return the Id used to identify this bond in the parent molecule.
   */
  svtkIdType GetId() const;

  /**
   * Return the parent molecule of this bond.
   */
  svtkMolecule* GetMolecule();

  //@{
  /**
   * Get the starting / ending atom ids for this bond.
   */
  svtkIdType GetBeginAtomId() const;
  svtkIdType GetEndAtomId() const;
  //@}

  //@{
  /**
   * Get a svtkAtom object that refers to the starting / ending atom
   * for this bond.
   */
  svtkAtom GetBeginAtom();
  svtkAtom GetEndAtom();
  svtkAtom GetBeginAtom() const;
  svtkAtom GetEndAtom() const;
  //@}

  /**
   * Get the bond order for this bond.
   */
  unsigned short GetOrder();

  /**
   * Get the distance between the bonded atoms.

   * @note This function is faster than svtkMolecule::GetBondLength and
   * should be used when possible.
   */
  double GetLength() const;

protected:
  friend class svtkMolecule;

  svtkBond(svtkMolecule* parent, svtkIdType id, svtkIdType beginAtomId, svtkIdType endAtomId);

  svtkMolecule* Molecule;
  svtkIdType Id;
  svtkIdType BeginAtomId;
  svtkIdType EndAtomId;
};

inline svtkIdType svtkBond::GetId() const
{
  return this->Id;
}

inline svtkMolecule* svtkBond::GetMolecule()
{
  return this->Molecule;
}

#endif
// SVTK-HeaderTest-Exclude: svtkBond.h
