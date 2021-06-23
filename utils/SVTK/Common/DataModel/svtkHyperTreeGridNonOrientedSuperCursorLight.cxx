/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkHyperTreeGridNonOrientedSuperCursorLight.cxx

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright Nonice for more information.

=========================================================================*/
#include "svtkHyperTreeGridNonOrientedSuperCursorLight.h"
#include "svtkHyperTree.h"
#include "svtkHyperTreeGrid.h"
#include "svtkHyperTreeGridNonOrientedGeometryCursor.h"

#include "svtkBitArray.h"
#include "svtkObjectFactory.h"
#include "svtkSmartPointer.h"

#include "svtkHyperTreeGridTools.h"

#include <cassert>
#include <climits>

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedSuperCursorLight* svtkHyperTreeGridNonOrientedSuperCursorLight::Clone()
{
  svtkHyperTreeGridNonOrientedSuperCursorLight* clone = this->NewInstance();
  assert("post: clone_exists" && clone != nullptr);
  // Copy
  clone->Grid = this->Grid;
  clone->CentralCursor->Initialize(this->CentralCursor.Get());
  clone->CurrentFirstNonValidEntryByLevel = this->CurrentFirstNonValidEntryByLevel;
  {
    clone->FirstNonValidEntryByLevel.resize(this->FirstNonValidEntryByLevel.size());
    std::vector<unsigned int>::iterator in = this->FirstNonValidEntryByLevel.begin();
    std::vector<unsigned int>::iterator out = clone->FirstNonValidEntryByLevel.begin();
    for (; in != this->FirstNonValidEntryByLevel.end(); ++in, ++out)
    {
      (*out) = (*in);
    }
  }
  {
    clone->Entries.resize(this->Entries.size());
    std::vector<svtkHyperTreeGridLevelEntry>::iterator in = this->Entries.begin();
    std::vector<svtkHyperTreeGridLevelEntry>::iterator out = clone->Entries.begin();
    for (; in != this->Entries.end(); ++in, ++out)
    {
      (*out).Copy(&(*in));
    }
  }
  clone->FirstCurrentNeighboorReferenceEntry = this->FirstCurrentNeighboorReferenceEntry;
  {
    clone->ReferenceEntries.resize(this->ReferenceEntries.size());
    std::vector<unsigned int>::iterator in = this->ReferenceEntries.begin();
    std::vector<unsigned int>::iterator out = clone->ReferenceEntries.begin();
    for (; in != this->ReferenceEntries.end(); ++in, ++out)
    {
      (*out) = (*in);
    }
  }
  clone->IndiceCentralCursor = this->IndiceCentralCursor;
  clone->NumberOfCursors = this->NumberOfCursors;
  clone->ChildCursorToParentCursorTable = this->ChildCursorToParentCursorTable;
  clone->ChildCursorToChildTable = this->ChildCursorToChildTable;
  return clone;
}

//---------------------------------------------------------------------------
svtkHyperTreeGrid* svtkHyperTreeGridNonOrientedSuperCursorLight::GetGrid()
{
  return this->Grid;
}

//---------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedSuperCursorLight::HasTree()
{
  return this->CentralCursor->HasTree();
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedSuperCursorLight::HasTree(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->HasTree();
  }
  return svtk::hypertreegrid::HasTree(this->Entries[this->GetIndiceEntry(icursor)]);
}

//---------------------------------------------------------------------------
svtkHyperTree* svtkHyperTreeGridNonOrientedSuperCursorLight::GetTree()
{
  return this->CentralCursor->GetTree();
}

//---------------------------------------------------------------------------
svtkHyperTree* svtkHyperTreeGridNonOrientedSuperCursorLight::GetTree(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->GetTree();
  }
  return this->Entries[this->GetIndiceEntry(icursor)].GetTree();
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridNonOrientedSuperCursorLight::GetVertexId()
{
  return this->CentralCursor->GetVertexId();
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridNonOrientedSuperCursorLight::GetVertexId(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->GetVertexId();
  }
  return this->Entries[this->GetIndiceEntry(icursor)].GetVertexId();
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridNonOrientedSuperCursorLight::GetGlobalNodeIndex()
{
  return this->CentralCursor->GetGlobalNodeIndex();
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridNonOrientedSuperCursorLight::GetGlobalNodeIndex(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->GetGlobalNodeIndex();
  }
  return this->Entries[this->GetIndiceEntry(icursor)].GetGlobalNodeIndex();
}

//-----------------------------------------------------------------------------
svtkHyperTree* svtkHyperTreeGridNonOrientedSuperCursorLight::GetInformation(
  unsigned int icursor, unsigned int& level, bool& leaf, svtkIdType& id)
{
  if (icursor == this->IndiceCentralCursor)
  {
    level = this->CentralCursor->GetLevel();
    leaf = this->CentralCursor->IsLeaf();
    id = this->CentralCursor->GetGlobalNodeIndex();
    return this->CentralCursor->GetTree();
  }
  svtkHyperTreeGridLevelEntry& entry = this->Entries[this->GetIndiceEntry(icursor)];
  svtkHyperTree* tree = entry.GetTree();
  if (tree)
  {
    level = entry.GetLevel();
    leaf = entry.IsLeaf(this->Grid);
    id = entry.GetGlobalNodeIndex();
  }
  return tree;
}

//-----------------------------------------------------------------------------
unsigned char svtkHyperTreeGridNonOrientedSuperCursorLight::GetDimension()
{
  return this->Grid->GetDimension();
}

//-----------------------------------------------------------------------------
unsigned char svtkHyperTreeGridNonOrientedSuperCursorLight::GetNumberOfChildren()
{
  return this->CentralCursor->GetTree()->GetNumberOfChildren();
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursorLight::SetGlobalIndexStart(svtkIdType index)
{
  this->CentralCursor->SetGlobalIndexStart(index);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursorLight::SetGlobalIndexFromLocal(svtkIdType index)
{
  this->CentralCursor->SetGlobalIndexFromLocal(index);
}

//-----------------------------------------------------------------------------
double* svtkHyperTreeGridNonOrientedSuperCursorLight::GetOrigin()
{
  return this->CentralCursor->GetOrigin();
}

//-----------------------------------------------------------------------------
double* svtkHyperTreeGridNonOrientedSuperCursorLight::GetSize()
{
  return this->CentralCursor->GetSize();
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursorLight::GetBounds(double bounds[6])
{
  this->CentralCursor->GetBounds(bounds);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursorLight::SetMask(bool state)
{
  assert("pre: not_tree" && this->CentralCursor->GetTree());
  this->CentralCursor->SetMask(state);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursorLight::SetMask(unsigned int icursor, bool state)
{
  if (icursor == this->IndiceCentralCursor)
  {
    this->SetMask(state);
  }
  else
  {
    svtkHyperTreeGridLevelEntry& entry = this->Entries[this->GetIndiceEntry(icursor)];
    assert("pre: not_tree" && entry.GetTree());
    entry.SetMask(this->Grid, state);
  }
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedSuperCursorLight::IsMasked()
{
  return this->CentralCursor->IsMasked();
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedSuperCursorLight::IsMasked(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->IsMasked();
  }
  svtkHyperTreeGridLevelEntry& entry = this->Entries[this->GetIndiceEntry(icursor)];
  return entry.IsMasked(this->Grid);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursorLight::GetPoint(double point[3])
{
  this->CentralCursor->GetPoint(point);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedSuperCursorLight::IsLeaf()
{
  return this->CentralCursor->IsLeaf();
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedSuperCursorLight::IsLeaf(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->IsLeaf();
  }
  return this->Entries[this->GetIndiceEntry(icursor)].IsLeaf(this->Grid);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursorLight::SubdivideLeaf()
{
  this->CentralCursor->SubdivideLeaf();
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedSuperCursorLight::IsRoot()
{
  return this->CentralCursor->IsRoot();
}

//-----------------------------------------------------------------------------
unsigned int svtkHyperTreeGridNonOrientedSuperCursorLight::GetLevel()
{
  return this->CentralCursor->GetLevel();
}

//-----------------------------------------------------------------------------
unsigned int svtkHyperTreeGridNonOrientedSuperCursorLight::GetLevel(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->GetLevel();
  }
  return this->Entries[this->GetIndiceEntry(icursor)].GetLevel();
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursorLight::ToChild(unsigned char ichild)
{
  assert("pre: Non_leaf" && !this->IsLeaf());
  //
  ++this->CurrentFirstNonValidEntryByLevel;
  if (this->FirstNonValidEntryByLevel.size() == this->CurrentFirstNonValidEntryByLevel)
  {
    this->FirstNonValidEntryByLevel.resize(this->CurrentFirstNonValidEntryByLevel + 1);
  }
  this->FirstNonValidEntryByLevel[this->CurrentFirstNonValidEntryByLevel] =
    this->FirstNonValidEntryByLevel[this->CurrentFirstNonValidEntryByLevel - 1];
  //
  this->FirstCurrentNeighboorReferenceEntry += (this->NumberOfCursors - 1);
  if (this->ReferenceEntries.size() == this->FirstCurrentNeighboorReferenceEntry)
  {
    this->ReferenceEntries.resize(
      this->FirstCurrentNeighboorReferenceEntry + (this->NumberOfCursors - 1));
  }
  // Point into traversal tables at child location
  int offset = ichild * this->NumberOfCursors;
  const unsigned int* pTab = this->ChildCursorToParentCursorTable + offset;
  const unsigned int* cTab = this->ChildCursorToChildTable + offset;

  // Move each cursor in the supercursor down to a child
  for (unsigned int i = 0; i < this->NumberOfCursors; ++i)
  {
    if (i != this->IndiceCentralCursor)
    {
      // Make relevant cursor in parent cell point towards current child cursor
      unsigned int j = pTab[i];

      // If neighnoring cell is further subdivided, then descend into it
      unsigned int reference = UINT_MAX;
      if (j == this->IndiceCentralCursor)
      {
        reference = this->FirstNonValidEntryByLevel[this->CurrentFirstNonValidEntryByLevel];
        ++this->FirstNonValidEntryByLevel[this->CurrentFirstNonValidEntryByLevel];
        if (this->Entries.size() <= reference)
        {
          this->Entries.resize(reference + 1);
        }
        //
        if (i > this->IndiceCentralCursor)
        {
          this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry + i - 1] = reference;
        }
        else
        {
          this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry + i] = reference;
        }
        //
        svtkHyperTreeGridLevelEntry& current = this->Entries[reference];
        current.Initialize(this->CentralCursor->GetTree(), this->CentralCursor->GetLevel(),
          this->CentralCursor->GetVertexId());
        //
        // JB1901 ne pas descendre si masque
        if (!this->IsMasked()) // JB1901 new code
        {                      // JB1901 new code
          //
          if (current.GetTree() && !current.IsLeaf(this->Grid))
          {
            // Move to child
            current.ToChild(this->Grid, cTab[i]);
          }
          //
        }
      }
      else
      {
        unsigned int previous = this->GetIndicePreviousEntry(j);
        //
        if (this->Entries[previous].GetTree() && !this->Entries[previous].IsLeaf(this->Grid) &&
          !(this->GetGrid()->HasMask()
              ? this->GetGrid()->GetMask()->GetValue(this->Entries[previous].GetGlobalNodeIndex())
              : 0))
        {
          reference = this->FirstNonValidEntryByLevel[this->CurrentFirstNonValidEntryByLevel];
          ++this->FirstNonValidEntryByLevel[this->CurrentFirstNonValidEntryByLevel];
          if (this->Entries.size() <= reference)
          {
            this->Entries.resize(reference + 1);
          }
          if (i > this->IndiceCentralCursor)
          {
            this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry + i - 1] = reference;
          }
          else
          {
            this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry + i] = reference;
          }
          //
          svtkHyperTreeGridLevelEntry& current = this->Entries[reference];
          current.Copy(&(this->Entries[previous]));
          current.ToChild(this->Grid, cTab[i]);
        }
        else
        {
          if (j > this->IndiceCentralCursor)
          {
            reference = this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry -
              (this->NumberOfCursors - 1) + j - 1];
          }
          else
          {
            reference = this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry -
              (this->NumberOfCursors - 1) + j];
          }
          if (i > this->IndiceCentralCursor)
          {
            this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry + i - 1] = reference;
          }
          else
          {
            this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry + i] = reference;
          }
        }
      }
    }
  } // i
  this->CentralCursor->ToChild(cTab[this->IndiceCentralCursor]);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursorLight::ToRoot()
{
  assert("pre: hypertree_exist" && this->Entries.size() > 0);
  this->CentralCursor->ToRoot();
  this->CurrentFirstNonValidEntryByLevel = 0;
  this->FirstCurrentNeighboorReferenceEntry = 0;
}

//---------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursorLight::ToParent()
{
  assert("pre: Non_root" && !this->IsRoot());
  this->CentralCursor->ToParent();
  this->CurrentFirstNonValidEntryByLevel--;
  this->FirstCurrentNeighboorReferenceEntry -= (this->NumberOfCursors - 1);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursorLight::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "--svtkHyperTreeGridNonOrientedSuperCursorLight--" << endl;
  this->CentralCursor->PrintSelf(os, indent);
  os << indent << "IndiceCentralCursor: " << this->IndiceCentralCursor << endl;
  os << indent << "NumberOfCursors: " << this->NumberOfCursors << endl;
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedSuperCursorLight::svtkHyperTreeGridNonOrientedSuperCursorLight()
{
  this->Grid = nullptr;
  this->IndiceCentralCursor = 0;
  this->NumberOfCursors = 0;
  this->ChildCursorToParentCursorTable = nullptr;
  this->ChildCursorToChildTable = nullptr;
  this->CurrentFirstNonValidEntryByLevel = 0;
  this->FirstCurrentNeighboorReferenceEntry = 0;

  this->CentralCursor = svtkSmartPointer<svtkHyperTreeGridNonOrientedGeometryCursor>::New();
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedSuperCursorLight::~svtkHyperTreeGridNonOrientedSuperCursorLight() {}

//-----------------------------------------------------------------------------
