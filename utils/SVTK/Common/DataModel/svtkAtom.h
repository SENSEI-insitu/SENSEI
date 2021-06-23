/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAtom.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkAtom
 * @brief   convenience proxy for svtkMolecule
 *
 */

#ifndef svtkAtom_h
#define svtkAtom_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"                // For macros, defines, etc

class svtkMolecule;
class svtkVector3d;
class svtkVector3f;

class SVTKCOMMONDATAMODEL_EXPORT svtkAtom
{
public:
  void PrintSelf(ostream& os, svtkIndent indent);

  /**
   * Return the Id used to identify this atom in the parent molecule.
   */
  svtkIdType GetId() const;

  /**
   * Return the parent molecule of this atom.
   */
  svtkMolecule* GetMolecule();

  //@{
  /**
   * Get/Set the atomic number of this atom
   */
  unsigned short GetAtomicNumber() const;
  void SetAtomicNumber(unsigned short atomicNum);
  //@}

  //@{
  /**
   * Get/Set the position of this atom
   */
  void GetPosition(float pos[3]) const;
  void GetPosition(double pos[3]) const;
  void SetPosition(const float pos[3]);
  void SetPosition(float x, float y, float z);
  svtkVector3f GetPosition() const;
  void SetPosition(const svtkVector3f& pos);
  //@}

protected:
  friend class svtkMolecule;

  svtkAtom(svtkMolecule* parent, svtkIdType id);

  svtkMolecule* Molecule;
  svtkIdType Id;
};

inline svtkIdType svtkAtom::GetId() const
{
  return this->Id;
}

inline svtkMolecule* svtkAtom::GetMolecule()
{
  return this->Molecule;
}

#endif
// SVTK-HeaderTest-Exclude: svtkAtom.h
