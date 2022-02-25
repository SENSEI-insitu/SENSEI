/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkHyperTreeGridOrientedGeometryCursor.cxx

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright Nonice for more information.

=========================================================================*/
#include "svtkHyperTreeGridOrientedGeometryCursor.h"

#include "svtkHyperTree.h"
#include "svtkHyperTreeGrid.h"
#include "svtkHyperTreeGridScales.h"

#include "svtkObjectFactory.h"

#include <cassert>

svtkStandardNewMacro(svtkHyperTreeGridOrientedGeometryCursor);

//-----------------------------------------------------------------------------
svtkHyperTreeGridOrientedGeometryCursor* svtkHyperTreeGridOrientedGeometryCursor::Clone()
{
  svtkHyperTreeGridOrientedGeometryCursor* clone = this->NewInstance();
  assert("post: clone_exists" && clone != nullptr);
  // Copy
  clone->Grid = this->Grid;
  clone->Tree = this->Tree;
  clone->Scales = this->Scales;
  clone->Level = this->Level;
  clone->Entry.Copy(&(this->Entry));
  // Return clone
  return clone;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedGeometryCursor::Initialize(
  svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create)
{
  this->Grid = grid;
  this->Level = 0;
  this->Tree = this->Entry.Initialize(grid, treeIndex, create);
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
void svtkHyperTreeGridOrientedGeometryCursor::Initialize(svtkHyperTreeGrid* grid, svtkHyperTree* tree,
  unsigned int level, svtkHyperTreeGridGeometryEntry& entry)
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
  this->Entry.Copy(&entry);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedGeometryCursor::Initialize(
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
  this->Entry.Initialize(index, origin);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedGeometryCursor::Initialize(
  svtkHyperTreeGridOrientedGeometryCursor* cursor)
{
  this->Grid = cursor->Grid;
  this->Tree = cursor->Tree;
  this->Scales = cursor->Scales;
  this->Level = cursor->Level;
  this->Entry.Copy(&(cursor->Entry));
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridOrientedGeometryCursor::GetVertexId()
{
  return this->Entry.GetVertexId();
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridOrientedGeometryCursor::GetGlobalNodeIndex()
{
  return this->Entry.GetGlobalNodeIndex(this->Tree);
}

//-----------------------------------------------------------------------------
unsigned char svtkHyperTreeGridOrientedGeometryCursor::GetDimension()
{
  return this->Grid->GetDimension();
}

//-----------------------------------------------------------------------------
unsigned char svtkHyperTreeGridOrientedGeometryCursor::GetNumberOfChildren()
{
  return this->Tree->GetNumberOfChildren();
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedGeometryCursor::SetGlobalIndexStart(svtkIdType index)
{
  this->Entry.SetGlobalIndexStart(this->Tree, index);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedGeometryCursor::SetGlobalIndexFromLocal(svtkIdType index)
{
  this->Entry.SetGlobalIndexFromLocal(this->Tree, index);
}

//-----------------------------------------------------------------------------
double* svtkHyperTreeGridOrientedGeometryCursor::GetOrigin()
{
  return this->Entry.GetOrigin();
}

//-----------------------------------------------------------------------------
double* svtkHyperTreeGridOrientedGeometryCursor::GetSize()
{
  return (double*)(this->Scales->GetScale(this->GetLevel()));
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedGeometryCursor::GetBounds(double bounds[6])
{
  this->Entry.GetBounds(this->GetSize(), bounds);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedGeometryCursor::GetPoint(double point[3])
{
  this->Entry.GetPoint(this->GetSize(), point);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedGeometryCursor::SetMask(bool state)
{
  this->Entry.SetMask(this->Grid, this->Tree, state);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridOrientedGeometryCursor::IsMasked()
{
  return this->Entry.IsMasked(this->Grid, this->Tree);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridOrientedGeometryCursor::IsLeaf()
{
  return this->Entry.IsLeaf(this->Grid, this->Tree, this->Level);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedGeometryCursor::SubdivideLeaf()
{
  this->Entry.SubdivideLeaf(this->Grid, this->Tree, this->Level);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridOrientedGeometryCursor::IsRoot()
{
  return this->Entry.IsRoot();
}

//-----------------------------------------------------------------------------
unsigned int svtkHyperTreeGridOrientedGeometryCursor::GetLevel()
{
  return this->Level;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedGeometryCursor::ToChild(unsigned char ichild)
{
  this->Entry.ToChild(
    this->Grid, this->Tree, this->Level, this->Scales->GetScale(this->Level + 1), ichild);
  this->Level++;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedGeometryCursor::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "--svtkHyperTreeGridOrientedGeometryCursor--" << endl;
  os << indent << "Level: " << this->GetLevel() << endl;
  this->Tree->PrintSelf(os, indent);
  this->Entry.PrintSelf(os, indent);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridOrientedGeometryCursor::Dump(ostream& os)
{
  os << "--svtkHyperTreeGridOrientedGeometryCursor--" << endl;
  os << "Grid: " << this->Grid << endl;
  os << "Tree: " << this->Tree << endl;
  os << "Scales: " << this->Scales << endl;
  os << "Level: " << this->Level << endl;
  os << "Entry: " << endl;
  this->Entry.Dump(os);
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridOrientedGeometryCursor::svtkHyperTreeGridOrientedGeometryCursor()
{
  this->Grid = nullptr;
  this->Tree = nullptr;
  this->Level = 0;
  // Appel au constructeur par defaut this->Entry
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridOrientedGeometryCursor::~svtkHyperTreeGridOrientedGeometryCursor() {}

//-----------------------------------------------------------------------------
