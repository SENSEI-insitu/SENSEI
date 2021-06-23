/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkHyperTreeGridEntry.cxx

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkHyperTreeGridEntry.h"

#include "svtkBitArray.h"

#include "svtkHyperTree.h"
#include "svtkHyperTreeGrid.h"

#include <cassert>

//-----------------------------------------------------------------------------
void svtkHyperTreeGridEntry::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "--svtkHyperTreeGridEntry--" << endl;
  os << indent << "Index:" << this->Index << endl;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridEntry::Dump(ostream& os)
{
  os << "Index:" << this->Index << endl;
}

//-----------------------------------------------------------------------------
svtkHyperTree* svtkHyperTreeGridEntry::Initialize(
  svtkHyperTreeGrid* grid, svtkIdType treeIndex, bool create)
{
  assert(grid != nullptr);
  this->Index = 0;
  return grid->GetTree(treeIndex, create);
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridEntry::GetGlobalNodeIndex(const svtkHyperTree* tree) const
{
  assert("pre: not_tree" && tree);
  return tree->GetGlobalIndexFromLocal(this->Index);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridEntry::SetGlobalIndexStart(svtkHyperTree* tree, svtkIdType index)
{
  assert("pre: not_tree" && tree);
  tree->SetGlobalIndexStart(index);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridEntry::SetGlobalIndexFromLocal(svtkHyperTree* tree, svtkIdType index)
{
  assert("pre: not_tree" && tree);
  tree->SetGlobalIndexFromLocal(this->Index, index);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridEntry::SetMask(
  const svtkHyperTreeGrid* grid, const svtkHyperTree* tree, bool value)
{
  assert("pre: not_tree" && tree);
  const_cast<svtkHyperTreeGrid*>(grid)->GetMask()->InsertTuple1(
    this->GetGlobalNodeIndex(tree), value);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridEntry::IsMasked(const svtkHyperTreeGrid* grid, const svtkHyperTree* tree) const
{
  if (tree && const_cast<svtkHyperTreeGrid*>(grid)->HasMask())
  {
    return const_cast<svtkHyperTreeGrid*>(grid)->GetMask()->GetValue(
             this->GetGlobalNodeIndex(tree)) != 0;
  }
  return false;
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridEntry::IsLeaf(
  const svtkHyperTreeGrid* grid, const svtkHyperTree* tree, unsigned int level) const
{
  assert("pre: not_tree" && tree);
  if (level == const_cast<svtkHyperTreeGrid*>(grid)->GetDepthLimiter())
  {
    return true;
  }
  return tree->IsLeaf(this->Index);
}

//---------------------------------------------------------------------------
void svtkHyperTreeGridEntry::SubdivideLeaf(
  const svtkHyperTreeGrid* grid, svtkHyperTree* tree, unsigned int level)
{
  assert("pre: not_tree" && tree);
  // JB Comment faire pour definir un accesseur a DepthLimiter qui est const
  assert("pre: depth_limiter" && level <= const_cast<svtkHyperTreeGrid*>(grid)->GetDepthLimiter());
  assert("pre: is_masked" && !this->IsMasked(grid, tree));
  if (this->IsLeaf(grid, tree, level))
  {
    tree->SubdivideLeaf(this->Index, level);
  }
}

//---------------------------------------------------------------------------
bool svtkHyperTreeGridEntry::IsTerminalNode(
  const svtkHyperTreeGrid* grid, const svtkHyperTree* tree, unsigned int level) const
{
  (void)level; // remove warning for release build
  assert("pre: not_tree" && tree);
  bool result = !this->IsLeaf(grid, tree, level);
  if (result)
  {
    result = tree->IsTerminalNode(this->Index);
  }
  assert("post: compatible" && (!result || !this->IsLeaf(grid, tree, level)));
  return result;
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridEntry::ToChild(
  const svtkHyperTreeGrid* grid, const svtkHyperTree* tree, unsigned int level, unsigned char ichild)
{
  (void)grid;  // only used in assert
  (void)level; // only used in assert
  assert("pre: not_tree" && tree);
  assert("pre: not_leaf" && !this->IsLeaf(grid, tree, level));
  assert("pre: not_valid_child" && ichild < tree->GetNumberOfChildren());
  // JB Comment faire pour definir un accesseur a DepthLimiter qui est const
  assert("pre: depth_limiter" && level <= const_cast<svtkHyperTreeGrid*>(grid)->GetDepthLimiter());
  assert("pre: is_masked" && !IsMasked(grid, tree));
  this->Index = tree->GetElderChildIndex(this->Index) + ichild;
}
