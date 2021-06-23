/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkAtom.cxx
Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkAtom.h"

#include "svtkMolecule.h"
#include "svtkVector.h"
#include "svtkVectorOperators.h"

#include <cassert>

//----------------------------------------------------------------------------
svtkAtom::svtkAtom(svtkMolecule* parent, svtkIdType id)
  : Molecule(parent)
  , Id(id)
{
  assert(parent != nullptr);
  assert(id < parent->GetNumberOfAtoms());
}

//----------------------------------------------------------------------------
void svtkAtom::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "Molecule: " << this->Molecule << " Id: " << this->Id
     << " Element: " << this->GetAtomicNumber() << " Position: " << this->GetPosition() << endl;
}

//----------------------------------------------------------------------------
unsigned short svtkAtom::GetAtomicNumber() const
{
  return this->Molecule->GetAtomAtomicNumber(this->Id);
}

//----------------------------------------------------------------------------
void svtkAtom::SetAtomicNumber(unsigned short atomicNum)
{
  this->Molecule->SetAtomAtomicNumber(this->Id, atomicNum);
}

//----------------------------------------------------------------------------
void svtkAtom::GetPosition(float pos[3]) const
{
  this->Molecule->GetAtomPosition(this->Id, pos);
}

//----------------------------------------------------------------------------
void svtkAtom::GetPosition(double pos[3]) const
{
  svtkVector3f position = this->GetPosition();
  pos[0] = position.GetX();
  pos[1] = position.GetY();
  pos[2] = position.GetZ();
}

//----------------------------------------------------------------------------
void svtkAtom::SetPosition(const float pos[3])
{
  this->Molecule->SetAtomPosition(this->Id, svtkVector3f(pos));
}

//----------------------------------------------------------------------------
void svtkAtom::SetPosition(float x, float y, float z)
{
  this->Molecule->SetAtomPosition(this->Id, x, y, z);
}

//----------------------------------------------------------------------------
svtkVector3f svtkAtom::GetPosition() const
{
  return this->Molecule->GetAtomPosition(this->Id);
}

//----------------------------------------------------------------------------
void svtkAtom::SetPosition(const svtkVector3f& pos)
{
  this->Molecule->SetAtomPosition(this->Id, pos);
}
