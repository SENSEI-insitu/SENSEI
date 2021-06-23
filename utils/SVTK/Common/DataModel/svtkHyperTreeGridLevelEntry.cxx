/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkHyperTreeGridLevelEntry.cxx

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkHyperTreeGridLevelEntry.h"

#include "svtkBitArray.h"

#include "svtkHyperTree.h"
#include "svtkHyperTreeGrid.h"
#include "svtkHyperTreeGridNonOrientedCursor.h"

#include <cassert>

//-----------------------------------------------------------------------------
svtkHyperTreeGridLevelEntry::svtkHyperTreeGridLevelEntry(
  svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create)
  : Tree(grid->GetTree(treeIndex, create))
  , Level(0)
  , Index(0)
{
}

//-----------------------------------------------------------------------------

svtkSmartPointer<svtkHyperTreeGridNonOrientedCursor>
svtkHyperTreeGridLevelEntry::GetHyperTreeGridNonOrientedCursor(svtkHyperTreeGrid* grid)
{
  // JB assert ( "pre: level==0" && this->Level == 0 );
  svtkSmartPointer<svtkHyperTreeGridNonOrientedCursor> cursor =
    svtkSmartPointer<svtkHyperTreeGridNonOrientedCursor>::New();
  cursor->Initialize(grid, this->GetTree(), this->Level, this->Index);
  return cursor;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridLevelEntry::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "--svtkHyperTreeGridLevelEntry--" << endl;
  this->Tree->PrintSelf(os, indent);
  os << indent << "Level:" << this->Level << endl;
  os << indent << "Index:" << this->Index << endl;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridLevelEntry::Dump(ostream& os)
{
  os << "Level:" << this->Level << endl;
  os << "Index:" << this->Index << endl;
}

//-----------------------------------------------------------------------------
svtkHyperTree* svtkHyperTreeGridLevelEntry::Initialize(
  svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create)
{
  this->Tree = grid->GetTree(treeIndex, create);
  this->Level = 0;
  this->Index = 0;
  return this->Tree;
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridLevelEntry::GetGlobalNodeIndex() const
{
  // JB BAD assert( "pre: not_tree" &&
  //     JB BAD     this->Tree );
  // Pourquoi ceci juste dans cette fonction entry ?
  if (this->Tree)
  {
    return this->Tree->GetGlobalIndexFromLocal(this->Index);
  }
  return -1;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridLevelEntry::SetGlobalIndexStart(svtkIdType index)
{
  assert("pre: not_tree" && this->Tree);
  this->Tree->SetGlobalIndexStart(index);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridLevelEntry::SetGlobalIndexFromLocal(svtkIdType index)
{
  assert("pre: not_tree" && this->Tree);
  this->Tree->SetGlobalIndexFromLocal(this->Index, index);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridLevelEntry::SetMask(const svtkHyperTreeGrid* grid, bool value)
{
  assert("pre: not_tree" && this->Tree);
  // JB Comment faire pour definir un accesseur a DepthLimiter qui est const
  const_cast<svtkHyperTreeGrid*>(grid)->GetMask()->InsertTuple1(this->GetGlobalNodeIndex(), value);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridLevelEntry::IsMasked(const svtkHyperTreeGrid* grid) const
{
  // JB Comment faire pour definir un accesseur a DepthLimiter qui est const
  if (this->Tree && const_cast<svtkHyperTreeGrid*>(grid)->HasMask())
  {
    return const_cast<svtkHyperTreeGrid*>(grid)->GetMask()->GetValue(this->GetGlobalNodeIndex()) !=
      0;
  }
  return false;
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridLevelEntry::IsLeaf(const svtkHyperTreeGrid* grid) const
{
  assert("pre: not_tree" && this->Tree);
  // How to set an accessor to DepthLimiter which is const?
  if (this->Level == const_cast<svtkHyperTreeGrid*>(grid)->GetDepthLimiter())
  {
    return true;
  }
  return this->Tree->IsLeaf(this->Index);
}

//---------------------------------------------------------------------------
void svtkHyperTreeGridLevelEntry::SubdivideLeaf(const svtkHyperTreeGrid* grid)
{
  assert("pre: not_tree" && this->Tree);
  // JB Comment faire pour definir un accesseur a DepthLimiter qui est const
  assert(
    "pre: depth_limiter" && this->Level <= const_cast<svtkHyperTreeGrid*>(grid)->GetDepthLimiter());
  assert("pre: is_masked" && !this->IsMasked(grid));
  if (this->IsLeaf(grid))
  {
    this->Tree->SubdivideLeaf(this->Index, this->Level);
  }
}

//---------------------------------------------------------------------------
bool svtkHyperTreeGridLevelEntry::IsTerminalNode(const svtkHyperTreeGrid* grid) const
{
  assert("pre: not_tree" && this->Tree);
  bool result = !this->IsLeaf(grid);
  if (result)
  {
    result = this->Tree->IsTerminalNode(this->Index);
  }
  assert("post: compatible" && (!result || !this->IsLeaf(grid)));
  return result;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridLevelEntry::ToChild(const svtkHyperTreeGrid* grid, unsigned char ichild)
{
  (void)grid; // Used in assert
  assert("pre: not_tree" && this->Tree);
  assert("pre: not_leaf" && !this->IsLeaf(grid));
  assert("pre: valid_child" && ichild < this->Tree->GetNumberOfChildren());
  // JB Comment faire pour definir un accesseur a DepthLimiter qui est const
  assert(
    "pre: depth_limiter" && this->Level <= const_cast<svtkHyperTreeGrid*>(grid)->GetDepthLimiter());
  assert("pre: is_masked" && !this->IsMasked(grid));
  this->Index = this->Tree->GetElderChildIndex(this->Index) + ichild;
  this->Level++;
}
