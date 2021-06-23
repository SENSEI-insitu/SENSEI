/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkHyperTreeGridNonOrientedGeometryCursor.cxx

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright Nonice for more information.

=========================================================================*/
#include "svtkHyperTreeGridNonOrientedGeometryCursor.h"

#include "svtkHyperTree.h"
#include "svtkHyperTreeGrid.h"
#include "svtkHyperTreeGridGeometryEntry.h"
#include "svtkHyperTreeGridScales.h"

#include "svtkObjectFactory.h"

#include <cassert>

#include "svtkHyperTreeGridOrientedGeometryCursor.h"

svtkStandardNewMacro(svtkHyperTreeGridNonOrientedGeometryCursor);

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedGeometryCursor* svtkHyperTreeGridNonOrientedGeometryCursor::Clone()
{
  svtkHyperTreeGridNonOrientedGeometryCursor* clone = this->NewInstance();
  assert("post: clone_exists" && clone != nullptr);
  // Copy
  clone->Grid = this->Grid;
  clone->Tree = this->Tree;
  clone->Scales = this->Scales;
  clone->Level = this->Level;
  clone->LastValidEntry = this->LastValidEntry;
  clone->Entries.resize(this->Entries.size());
  std::vector<svtkHyperTreeGridGeometryEntry>::iterator in = this->Entries.begin();
  std::vector<svtkHyperTreeGridGeometryEntry>::iterator out = clone->Entries.begin();
  for (; in != this->Entries.end(); ++in, ++out)
  {
    (*out).Copy(&(*in));
  }
  // Return clone
  return clone;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::Initialize(
  svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create)
{
  this->Grid = grid;
  this->LastValidEntry = 0;
  if (this->Entries.size() <= static_cast<size_t>(this->LastValidEntry))
  {
    this->Entries.resize(1);
  }
  this->Tree = this->Entries[0].Initialize(grid, treeIndex, create);
  if (this->Tree)
  {
    this->Scales = this->Tree->GetScales();
    assert(this->Scales);
  }
  else
  {
    this->Scales = nullptr;
  }
  this->Level = 0;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::Initialize(svtkHyperTreeGrid* grid,
  svtkHyperTree* tree, unsigned int level, svtkHyperTreeGridGeometryEntry& entry)
{
  this->Grid = grid;
  this->Tree = tree;
  if (this->Tree)
  {
    this->Scales = this->Tree->GetScales();
    assert(this->Scales);
  }
  else
  {
    this->Scales = nullptr;
  }
  this->Level = level;
  this->LastValidEntry = 0;
  this->Entries.resize(1);
  this->Entries[0].Copy(&entry);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::Initialize(
  svtkHyperTreeGrid* grid, svtkHyperTree* tree, unsigned int level, svtkIdType index, double* origin)
{
  this->Grid = grid;
  this->Tree = tree;
  if (this->Tree)
  {
    this->Scales = this->Tree->GetScales();
    assert(this->Scales);
  }
  else
  {
    this->Scales = nullptr;
  }
  this->Level = level;
  this->LastValidEntry = 0;
  this->Entries.resize(1);
  this->Entries[0].Initialize(index, origin);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::Initialize(
  svtkHyperTreeGridNonOrientedGeometryCursor* cursor)
{
  this->Grid = cursor->Grid;
  this->Tree = cursor->Tree;
  this->Scales = cursor->Scales;
  this->Level = cursor->Level;
  this->LastValidEntry = cursor->LastValidEntry;
  this->Entries.resize(cursor->Entries.size());
  std::vector<svtkHyperTreeGridGeometryEntry>::iterator in = this->Entries.begin();
  std::vector<svtkHyperTreeGridGeometryEntry>::iterator out = cursor->Entries.begin();
  for (; in != this->Entries.end(); ++in, ++out)
  {
    (*out).Copy(&(*in));
  }
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridNonOrientedGeometryCursor::GetVertexId()
{
  return this->Entries[this->LastValidEntry].GetVertexId();
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridNonOrientedGeometryCursor::GetGlobalNodeIndex()
{
  return this->Entries[this->LastValidEntry].GetGlobalNodeIndex(this->Tree);
}

//-----------------------------------------------------------------------------
unsigned char svtkHyperTreeGridNonOrientedGeometryCursor::GetDimension()
{
  return this->Grid->GetDimension();
}

//-----------------------------------------------------------------------------
unsigned char svtkHyperTreeGridNonOrientedGeometryCursor::GetNumberOfChildren()
{
  return this->Tree->GetNumberOfChildren();
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::SetGlobalIndexStart(svtkIdType index)
{
  this->Entries[this->LastValidEntry].SetGlobalIndexStart(this->Tree, index);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::SetGlobalIndexFromLocal(svtkIdType index)
{
  this->Entries[this->LastValidEntry].SetGlobalIndexFromLocal(this->Tree, index);
}

//-----------------------------------------------------------------------------
double* svtkHyperTreeGridNonOrientedGeometryCursor::GetOrigin()
{
  return this->Entries[this->LastValidEntry].GetOrigin();
}

//-----------------------------------------------------------------------------
double* svtkHyperTreeGridNonOrientedGeometryCursor::GetSize()
{
  return this->Scales->GetScale(this->Level);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::GetBounds(double bounds[6])
{
  this->Entries[this->LastValidEntry].GetBounds(this->GetSize(), bounds);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::GetPoint(double point[3])
{
  this->Entries[this->LastValidEntry].GetPoint(this->GetSize(), point);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::SetMask(bool state)
{
  this->Entries[this->LastValidEntry].SetMask(this->Grid, this->Tree, state);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedGeometryCursor::IsMasked()
{
  return this->Entries[this->LastValidEntry].IsMasked(this->Grid, this->Tree);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedGeometryCursor::IsLeaf()
{
  return this->Entries[this->LastValidEntry].IsLeaf(this->Grid, this->Tree, this->Level);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::SubdivideLeaf()
{
  this->Entries[this->LastValidEntry].SubdivideLeaf(this->Grid, this->Tree, Level);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedGeometryCursor::IsRoot()
{
  return this->Entries[this->LastValidEntry].IsRoot();
}

//-----------------------------------------------------------------------------
unsigned int svtkHyperTreeGridNonOrientedGeometryCursor::GetLevel()
{
  return this->Level;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::ToChild(unsigned char ichild)
{
  unsigned int oldLastValidEntry = this->LastValidEntry;
  this->LastValidEntry++;
  //
  if (this->Entries.size() == static_cast<size_t>(this->LastValidEntry))
  {
    this->Entries.resize(this->LastValidEntry + 1);
  }
  //
  svtkHyperTreeGridGeometryEntry& entry = this->Entries[this->LastValidEntry];
  entry.Copy(&this->Entries[oldLastValidEntry]);
  entry.ToChild(
    this->Grid, this->Tree, this->Level, this->Scales->GetScale(this->Level + 1), ichild);
  this->Level++;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::ToRoot()
{
  assert("pre: hypertree_exist" && this->Entries.size() > 0);
  this->Level -= this->LastValidEntry;
  this->LastValidEntry = 0;
}

//---------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::ToParent()
{
  assert("pre: Non_root" && !this->IsRoot());
  this->LastValidEntry--;
  this->Level--;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "--svtkHyperTreeGridNonOrientedGeometryCursor--" << endl;
  os << indent << "Level: " << this->Level << endl;
  this->Tree->PrintSelf(os, indent);
  os << indent << "LastValidEntry: " << this->LastValidEntry << endl;
  this->Entries[this->LastValidEntry].PrintSelf(os, indent);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedGeometryCursor::Dump(ostream& os)
{
  os << "--svtkHyperTreeGridNonOrientedGeometryCursor--" << endl;
  os << "Grid: " << this->Grid << endl;
  os << "Tree: " << this->Tree << endl;
  os << "Scales: " << this->Scales << endl;
  os << "Level: " << this->Level << endl;
  os << "LastValidEntry: " << this->LastValidEntry << endl;
  int ientry = 0;
  for (; ientry <= this->LastValidEntry; ++ientry)
  {
    os << "Entries: #" << ientry << endl;
    this->Entries[ientry].Dump(os);
  }
  for (; static_cast<size_t>(ientry) < this->Entries.size(); ++ientry)
  {
    os << "Entries: #" << ientry << " Non USED" << endl;
    this->Entries[ientry].Dump(os);
  }
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedGeometryCursor::svtkHyperTreeGridNonOrientedGeometryCursor()
{
  this->Grid = nullptr;
  this->Tree = nullptr;
  this->Level = 0;
  this->LastValidEntry = -1;
  // Appel au constructeur par defaut this->Entries
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedGeometryCursor::~svtkHyperTreeGridNonOrientedGeometryCursor() {}

//-----------------------------------------------------------------------------
svtkSmartPointer<svtkHyperTreeGridOrientedGeometryCursor>
svtkHyperTreeGridNonOrientedGeometryCursor::GetHyperTreeGridOrientedGeometryCursor(
  svtkHyperTreeGrid* grid)
{
  svtkSmartPointer<svtkHyperTreeGridOrientedGeometryCursor> cursor =
    svtkSmartPointer<svtkHyperTreeGridOrientedGeometryCursor>::New();
  cursor->Initialize(grid, this->Tree, this->GetLevel(), this->GetVertexId(), this->GetOrigin());
  return cursor;
}
