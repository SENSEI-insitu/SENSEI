/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkBond.cxx
Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkBond.h"

#include "svtkAtom.h"
#include "svtkMolecule.h"
#include "svtkVector.h"
#include "svtkVectorOperators.h"

#include <cassert>

//----------------------------------------------------------------------------
svtkBond::svtkBond(svtkMolecule* parent, svtkIdType id, svtkIdType beginAtomId, svtkIdType endAtomId)
  : Molecule(parent)
  , Id(id)
  , BeginAtomId(beginAtomId)
  , EndAtomId(endAtomId)
{
  assert(parent != nullptr);
  assert(id < parent->GetNumberOfBonds());
  assert(beginAtomId < parent->GetNumberOfAtoms());
  assert(endAtomId < parent->GetNumberOfAtoms());
}

//----------------------------------------------------------------------------
void svtkBond::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "Molecule: " << this->Molecule << " Id: " << this->Id
     << " Order: " << this->GetOrder() << " Length: " << this->GetLength()
     << " BeginAtomId: " << this->BeginAtomId << " EndAtomId: " << this->EndAtomId << endl;
}

//----------------------------------------------------------------------------
double svtkBond::GetLength() const
{
  // Reimplement here to avoid the potential cost of building the EdgeList
  // (We already know the atomIds, no need to look them up)
  svtkVector3f pos1 = this->Molecule->GetAtomPosition(this->BeginAtomId);
  svtkVector3f pos2 = this->Molecule->GetAtomPosition(this->EndAtomId);

  return (pos2 - pos1).Norm();
}

//----------------------------------------------------------------------------
svtkIdType svtkBond::GetBeginAtomId() const
{
  return this->BeginAtomId;
}

//----------------------------------------------------------------------------
svtkIdType svtkBond::GetEndAtomId() const
{
  return this->EndAtomId;
}

//----------------------------------------------------------------------------
svtkAtom svtkBond::GetBeginAtom()
{
  return this->Molecule->GetAtom(this->BeginAtomId);
}

//----------------------------------------------------------------------------
svtkAtom svtkBond::GetEndAtom()
{
  return this->Molecule->GetAtom(this->EndAtomId);
}

//----------------------------------------------------------------------------
svtkAtom svtkBond::GetBeginAtom() const
{
  return this->Molecule->GetAtom(this->BeginAtomId);
}

//----------------------------------------------------------------------------
svtkAtom svtkBond::GetEndAtom() const
{
  return this->Molecule->GetAtom(this->EndAtomId);
}

//----------------------------------------------------------------------------
unsigned short svtkBond::GetOrder()
{
  return this->Molecule->GetBondOrder(this->Id);
}
