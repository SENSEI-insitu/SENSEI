/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkHyperTreeGridNonOrientedCursor.cxx

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkHyperTreeGridNonOrientedCursor.h"

#include "svtkHyperTree.h"
#include "svtkHyperTreeGrid.h"
#include "svtkHyperTreeGridEntry.h"
#include "svtkHyperTreeGridTools.h"
#include "svtkObjectFactory.h"

#include <cassert>

svtkStandardNewMacro(svtkHyperTreeGridNonOrientedCursor);

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedCursor* svtkHyperTreeGridNonOrientedCursor::Clone()
{
  svtkHyperTreeGridNonOrientedCursor* clone = this->NewInstance();
  assert("post: clone_exists" && clone != nullptr);
  // Copy
  clone->Grid = this->Grid;
  clone->Tree = this->Tree;
  clone->Level = this->Level;
  clone->LastValidEntry = this->LastValidEntry;
  clone->Entries.resize(this->Entries.size());
  std::vector<svtkHyperTreeGridEntry>::iterator in = this->Entries.begin();
  std::vector<svtkHyperTreeGridEntry>::iterator out = clone->Entries.begin();
  for (; in != this->Entries.end(); ++in, ++out)
  {
    (*out).Copy(&(*in));
  }
  // Return clone
  return clone;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedCursor::Initialize(
  svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create)
{
  this->Grid = grid;
  this->Level = 0;
  this->LastValidEntry = 0;
  this->Entries.resize(1);
  this->Tree = this->Entries[0].Initialize(grid, treeIndex, create);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedCursor::Initialize(
  svtkHyperTreeGrid* grid, svtkHyperTree* tree, unsigned int level, svtkHyperTreeGridEntry& entry)
{
  this->Grid = grid;
  this->Tree = tree;
  this->Level = level;
  this->LastValidEntry = 0;
  this->Entries.resize(1);
  this->Entries[0].Copy(&entry);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedCursor::Initialize(
  svtkHyperTreeGrid* grid, svtkHyperTree* tree, unsigned int level, svtkIdType index)
{
  assert(this->Entries.size() && "this->Entries empty");
  this->Grid = grid;
  this->Tree = tree;
  this->Level = level;
  this->LastValidEntry = 0;
  this->Entries.resize(1);
  this->Entries[0].Initialize(index);
}

//---------------------------------------------------------------------------
svtkHyperTreeGrid* svtkHyperTreeGridNonOrientedCursor::GetGrid()
{
  return this->Grid;
}

//---------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedCursor::HasTree() const
{
  return svtk::hypertreegrid::HasTree(*this);
}

//---------------------------------------------------------------------------
svtkHyperTree* svtkHyperTreeGridNonOrientedCursor::GetTree() const
{
  return this->Tree;
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridNonOrientedCursor::GetVertexId()
{
  return this->Entries[this->LastValidEntry].GetVertexId();
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridNonOrientedCursor::GetGlobalNodeIndex()
{
  return this->Entries[this->LastValidEntry].GetGlobalNodeIndex(this->Tree);
}

//-----------------------------------------------------------------------------
unsigned char svtkHyperTreeGridNonOrientedCursor::GetDimension()
{
  return this->Grid->GetDimension();
}

//-----------------------------------------------------------------------------
unsigned char svtkHyperTreeGridNonOrientedCursor::GetNumberOfChildren()
{
  return this->Tree->GetNumberOfChildren();
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedCursor::SetGlobalIndexStart(svtkIdType index)
{
  this->Entries[this->LastValidEntry].SetGlobalIndexStart(this->Tree, index);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedCursor::SetGlobalIndexFromLocal(svtkIdType index)
{
  this->Entries[this->LastValidEntry].SetGlobalIndexFromLocal(this->Tree, index);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedCursor::SetMask(bool state)
{
  this->Entries[this->LastValidEntry].SetMask(this->Grid, this->Tree, state);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedCursor::IsMasked()
{
  return this->Entries[this->LastValidEntry].IsMasked(this->Grid, this->Tree);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedCursor::IsLeaf()
{
  return this->Entries[this->LastValidEntry].IsLeaf(this->Grid, this->Tree, this->Level);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedCursor::SubdivideLeaf()
{
  this->Entries[this->LastValidEntry].SubdivideLeaf(this->Grid, this->Tree, this->Level);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedCursor::IsRoot()
{
  return this->Entries[this->LastValidEntry].IsRoot();
}

//-----------------------------------------------------------------------------
unsigned int svtkHyperTreeGridNonOrientedCursor::GetLevel()
{
  return this->Level;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedCursor::ToChild(unsigned char ichild)
{
  unsigned int oldLastValidEntry = this->LastValidEntry;
  this->LastValidEntry++;
  //
  if (this->Entries.size() == static_cast<size_t>(this->LastValidEntry))
  {
    this->Entries.resize(this->LastValidEntry + 1);
  }
  //
  svtkHyperTreeGridEntry& entry = this->Entries[this->LastValidEntry];
  entry.Copy(&this->Entries[oldLastValidEntry]);
  entry.ToChild(this->Grid, this->Tree, this->Level, ichild);
  this->Level++;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedCursor::ToRoot()
{
  assert("pre: hypertree_exist" && this->Entries.size() > 0);
  this->LastValidEntry = 0;
  this->Level = 0;
}

//---------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedCursor::ToParent()
{
  assert("pre: not_root" && !this->IsRoot());
  this->LastValidEntry--;
  this->Level--;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedCursor::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "--svtkHyperTreeGridNonOrientedCursor--" << endl;
  os << indent << "Level: " << this->GetLevel() << endl;
  this->Tree->PrintSelf(os, indent);
  if (this->Entries.size())
  {
    os << indent << "LastValidEntry: " << this->LastValidEntry << endl;
    this->Entries[this->LastValidEntry].PrintSelf(os, indent);
  }
  else
  {
    os << indent << "No valid entry " << std::endl;
  }
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedCursor::svtkHyperTreeGridNonOrientedCursor()
{
  this->Grid = nullptr;
  this->Tree = nullptr;
  this->Level = 0;
  this->LastValidEntry = -1;
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedCursor::~svtkHyperTreeGridNonOrientedCursor() {}

//-----------------------------------------------------------------------------
