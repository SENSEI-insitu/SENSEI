/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkHyperTreeGridOrientedCursor.cxx

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkHyperTreeGridOrientedCursor.h"

#include "svtkHyperTree.h"
#include "svtkHyperTreeGrid.h"
#include "svtkHyperTreeGridEntry.h"
#include "svtkHyperTreeGridTools.h"
#include "svtkObjectFactory.h"

#include <cassert>

svtkStandardNewMacro(svtkHyperTreeGridOrientedCursor);

//-----------------------------------------------------------------------------
svtkHyperTreeGridOrientedCursor* svtkHyperTreeGridOrientedCursor::Clone()
{
  svtkHyperTreeGridOrientedCursor* clone = this->NewInstance();
  assert("post: clone_exists" && clone != nullptr);
  // Copy
  clone->Grid = this->Grid;
  clone->Tree = this->Tree;
  clone->Level = this->Level;
  clone->Entry.Copy(&(this->Entry));
  // Return clone
  return clone;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedCursor::Initialize(
  svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create)
{
  this->Grid = grid;
  this->Level = 0;
  this->Tree = this->Entry.Initialize(grid, treeIndex, create);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedCursor::Initialize(
  svtkHyperTreeGrid* grid, svtkHyperTree* tree, unsigned int level, svtkHyperTreeGridEntry& entry)
{
  this->Grid = grid;
  this->Tree = tree;
  this->Level = level;
  this->Entry.Copy(&entry);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedCursor::Initialize(
  svtkHyperTreeGrid* grid, svtkHyperTree* tree, unsigned int level, svtkIdType index)
{
  this->Grid = grid;
  this->Tree = tree;
  this->Level = level;
  this->Entry.Initialize(index);
}

//---------------------------------------------------------------------------
svtkHyperTreeGrid* svtkHyperTreeGridOrientedCursor::GetGrid()
{
  return this->Grid;
}

//---------------------------------------------------------------------------
bool svtkHyperTreeGridOrientedCursor::HasTree() const
{
  return svtk::hypertreegrid::HasTree(*this);
}

//---------------------------------------------------------------------------
svtkHyperTree* svtkHyperTreeGridOrientedCursor::GetTree() const
{
  return this->Tree;
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridOrientedCursor::GetVertexId()
{
  return this->Entry.GetVertexId();
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridOrientedCursor::GetGlobalNodeIndex()
{
  return this->Entry.GetGlobalNodeIndex(this->Tree);
}

//-----------------------------------------------------------------------------
unsigned char svtkHyperTreeGridOrientedCursor::GetDimension()
{
  return this->Grid->GetDimension();
}

//-----------------------------------------------------------------------------
unsigned char svtkHyperTreeGridOrientedCursor::GetNumberOfChildren()
{
  return this->Tree->GetNumberOfChildren();
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedCursor::SetGlobalIndexStart(svtkIdType index)
{
  this->Entry.SetGlobalIndexStart(this->Tree, index);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedCursor::SetGlobalIndexFromLocal(svtkIdType index)
{
  this->Entry.SetGlobalIndexFromLocal(this->Tree, index);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedCursor::SetMask(bool state)
{
  this->Entry.SetMask(this->Grid, this->Tree, state);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridOrientedCursor::IsMasked()
{
  return this->Entry.IsMasked(this->Grid, this->Tree);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridOrientedCursor::IsLeaf()
{
  return this->Entry.IsLeaf(this->Grid, this->Tree, this->Level);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedCursor::SubdivideLeaf()
{
  this->Entry.SubdivideLeaf(this->Grid, this->Tree, this->Level);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridOrientedCursor::IsRoot()
{
  return this->Entry.IsRoot();
}

//-----------------------------------------------------------------------------
unsigned int svtkHyperTreeGridOrientedCursor::GetLevel()
{
  return this->Level;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedCursor::ToChild(unsigned char ichild)
{
  this->Entry.ToChild(this->Grid, this->Tree, this->Level, ichild);
  this->Level++;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedCursor::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "--svtkHyperTreeGridOrientedCursor--" << endl;
  os << indent << "Level: " << this->GetLevel() << endl;
  this->Tree->PrintSelf(os, indent);
  this->Entry.PrintSelf(os, indent);
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridOrientedCursor::svtkHyperTreeGridOrientedCursor()
{
  this->Grid = nullptr;
  this->Level = 0;
  this->Tree = nullptr;
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridOrientedCursor::~svtkHyperTreeGridOrientedCursor() {}

//-----------------------------------------------------------------------------
