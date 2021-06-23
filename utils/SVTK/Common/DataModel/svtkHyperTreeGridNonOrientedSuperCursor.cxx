/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkHyperTreeGridNonOrientedSuperCursor.cxx

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright Nonice for more information.

=========================================================================*/
#include "svtkHyperTreeGridNonOrientedSuperCursor.h"

#include "svtkBitArray.h"

#include "svtkHyperTree.h"
#include "svtkHyperTreeGrid.h"
#include "svtkHyperTreeGridNonOrientedGeometryCursor.h"

#include "svtkObjectFactory.h"
#include "svtkSmartPointer.h"

#include "svtkHyperTreeGridTools.h"

#include <cassert>
#include <climits>
#include <ostream>
#include <vector>
//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedSuperCursor* svtkHyperTreeGridNonOrientedSuperCursor::Clone()
{
  svtkHyperTreeGridNonOrientedSuperCursor* clone = this->NewInstance();
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
    std::vector<svtkHyperTreeGridGeometryLevelEntry>::iterator in = this->Entries.begin();
    std::vector<svtkHyperTreeGridGeometryLevelEntry>::iterator out = clone->Entries.begin();
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
svtkHyperTreeGrid* svtkHyperTreeGridNonOrientedSuperCursor::GetGrid()
{
  return this->Grid;
}

//---------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedSuperCursor::HasTree()
{
  return this->CentralCursor->HasTree();
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedSuperCursor::HasTree(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->HasTree();
  }
  return svtk::hypertreegrid::HasTree(this->Entries[this->GetIndiceEntry(icursor)]);
}

//---------------------------------------------------------------------------
svtkHyperTree* svtkHyperTreeGridNonOrientedSuperCursor::GetTree()
{
  return this->CentralCursor->GetTree();
}

//---------------------------------------------------------------------------
svtkHyperTree* svtkHyperTreeGridNonOrientedSuperCursor::GetTree(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->GetTree();
  }
  return this->Entries[this->GetIndiceEntry(icursor)].GetTree();
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridNonOrientedSuperCursor::GetVertexId()
{
  return this->CentralCursor->GetVertexId();
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridNonOrientedSuperCursor::GetVertexId(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->GetVertexId();
  }
  return this->Entries[this->GetIndiceEntry(icursor)].GetVertexId();
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridNonOrientedSuperCursor::GetGlobalNodeIndex()
{
  return this->CentralCursor->GetGlobalNodeIndex();
}

//-----------------------------------------------------------------------------
svtkIdType svtkHyperTreeGridNonOrientedSuperCursor::GetGlobalNodeIndex(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->GetGlobalNodeIndex();
  }
  return this->Entries[this->GetIndiceEntry(icursor)].GetGlobalNodeIndex();
}

//-----------------------------------------------------------------------------
svtkHyperTree* svtkHyperTreeGridNonOrientedSuperCursor::GetInformation(
  unsigned int icursor, unsigned int& level, bool& leaf, svtkIdType& id)
{
  if (icursor == this->IndiceCentralCursor)
  {
    level = this->CentralCursor->GetLevel();
    leaf = this->CentralCursor->IsLeaf();
    id = this->CentralCursor->GetGlobalNodeIndex();
    return this->CentralCursor->GetTree();
  }
  svtkHyperTreeGridGeometryLevelEntry& entry = this->Entries[this->GetIndiceEntry(icursor)];
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
unsigned char svtkHyperTreeGridNonOrientedSuperCursor::GetDimension()
{
  return this->Grid->GetDimension();
}

//-----------------------------------------------------------------------------
unsigned char svtkHyperTreeGridNonOrientedSuperCursor::GetNumberOfChildren()
{
  return this->CentralCursor->GetTree()->GetNumberOfChildren();
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursor::SetGlobalIndexStart(svtkIdType index)
{
  this->CentralCursor->SetGlobalIndexStart(index);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursor::SetGlobalIndexFromLocal(svtkIdType index)
{
  this->CentralCursor->SetGlobalIndexFromLocal(index);
}

//-----------------------------------------------------------------------------
double* svtkHyperTreeGridNonOrientedSuperCursor::GetOrigin()
{
  return this->CentralCursor->GetOrigin();
}

//-----------------------------------------------------------------------------
double* svtkHyperTreeGridNonOrientedSuperCursor::GetSize()
{
  return this->CentralCursor->GetSize();
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursor::GetBounds(double bounds[6])
{
  this->CentralCursor->GetBounds(bounds);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursor::GetBounds(unsigned int icursor, double bounds[6])
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->GetBounds(bounds);
  }
  return this->Entries[this->GetIndiceEntry(icursor)].GetBounds(bounds);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursor::GetPoint(double point[3])
{
  this->CentralCursor->GetPoint(point);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursor::GetPoint(unsigned int icursor, double point[3])
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->GetPoint(point);
  }
  return this->Entries[this->GetIndiceEntry(icursor)].GetPoint(point);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursor::SetMask(bool state)
{
  assert("pre: not_tree" && this->CentralCursor->GetTree());
  this->CentralCursor->SetMask(state);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursor::SetMask(unsigned int icursor, bool state)
{
  if (icursor == this->IndiceCentralCursor)
  {
    this->SetMask(state);
  }
  else
  {
    svtkHyperTreeGridGeometryLevelEntry& entry = this->Entries[this->GetIndiceEntry(icursor)];
    assert("pre: not_tree" && entry.GetTree());
    entry.SetMask(this->Grid, state);
  }
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedSuperCursor::IsMasked()
{
  return this->CentralCursor->IsMasked();
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedSuperCursor::IsMasked(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->IsMasked();
  }
  svtkHyperTreeGridGeometryLevelEntry& entry = this->Entries[this->GetIndiceEntry(icursor)];
  return entry.IsMasked(this->Grid);
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedSuperCursor::IsLeaf()
{
  return this->CentralCursor->IsLeaf();
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedSuperCursor::IsLeaf(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->IsLeaf();
  }
  return this->Entries[this->GetIndiceEntry(icursor)].IsLeaf(this->Grid);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursor::SubdivideLeaf()
{
  this->CentralCursor->SubdivideLeaf();
}

//-----------------------------------------------------------------------------
bool svtkHyperTreeGridNonOrientedSuperCursor::IsRoot()
{
  return this->CentralCursor->IsRoot();
}

//-----------------------------------------------------------------------------
unsigned int svtkHyperTreeGridNonOrientedSuperCursor::GetLevel()
{
  return this->CentralCursor->GetLevel();
}

//-----------------------------------------------------------------------------
unsigned int svtkHyperTreeGridNonOrientedSuperCursor::GetLevel(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->GetLevel();
  }
  return this->Entries[this->GetIndiceEntry(icursor)].GetLevel();
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursor::ToChild(unsigned char ichild)
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
        svtkHyperTreeGridGeometryLevelEntry& current = this->Entries[reference];
        current.Initialize(this->CentralCursor->GetTree(), this->CentralCursor->GetLevel(),
          this->CentralCursor->GetVertexId(), this->CentralCursor->GetOrigin());
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
          svtkHyperTreeGridGeometryLevelEntry& current = this->Entries[reference];
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
void svtkHyperTreeGridNonOrientedSuperCursor::ToRoot()
{
  assert("pre: hypertree_exist" && this->Entries.size() > 0);
  this->CentralCursor->ToRoot();
  this->CurrentFirstNonValidEntryByLevel = 0;
  this->FirstCurrentNeighboorReferenceEntry = 0;
}

//---------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursor::ToParent()
{
  assert("pre: Non_root" && !this->IsRoot());
  this->CentralCursor->ToParent();
  this->CurrentFirstNonValidEntryByLevel--;
  this->FirstCurrentNeighboorReferenceEntry -= (this->NumberOfCursors - 1);
}

//-----------------------------------------------------------------------------
void svtkHyperTreeGridNonOrientedSuperCursor::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "--svtkHyperTreeGridNonOrientedSuperCursor--" << endl;
  this->CentralCursor->PrintSelf(os, indent);
  os << indent << "IndiceCentralCursor: " << this->IndiceCentralCursor << endl;
  os << indent << "NumberOfCursors: " << this->NumberOfCursors << endl;
}

//-----------------------------------------------------------------------------
svtkHyperTreeGridNonOrientedSuperCursor::svtkHyperTreeGridNonOrientedSuperCursor()
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

svtkHyperTreeGridNonOrientedSuperCursor::~svtkHyperTreeGridNonOrientedSuperCursor() {}

//-----------------------------------------------------------------------------
svtkSmartPointer<svtkHyperTreeGridOrientedGeometryCursor>
svtkHyperTreeGridNonOrientedSuperCursor::GetOrientedGeometryCursor(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor->GetHyperTreeGridOrientedGeometryCursor(this->Grid);
  }
  return this->Entries[this->GetIndiceEntry(icursor)].GetHyperTreeGridOrientedGeometryCursor(
    this->Grid);
}

//-----------------------------------------------------------------------------

svtkSmartPointer<svtkHyperTreeGridNonOrientedGeometryCursor>
svtkHyperTreeGridNonOrientedSuperCursor::GetNonOrientedGeometryCursor(unsigned int icursor)
{
  if (icursor == this->IndiceCentralCursor)
  {
    return this->CentralCursor; // JB deja du bon type
  }
  assert(false);
  // JB ou faire le boulot pour le construire
  // JB On ne peut pas construire ce cursor car this->Level n'est pas forcement 0
  // JB oeut etre le fournir telle quelle ? Tant que l'on ne remonte pas au dessus du niveau
  // d'origine de creation du cursor ? ou alors creer le cursor avec son heritage.. cout plus eleve
  // ?!?
  return this->Entries[this->GetIndiceEntry(icursor)].GetHyperTreeGridNonOrientedGeometryCursor(
    this->Grid);
}

//-----------------------------------------------------------------------------

unsigned int svtkHyperTreeGridNonOrientedSuperCursor::GetIndiceEntry(unsigned int icursor)
{
  assert("pre: icursor != IndiceCentralCursor" && icursor != this->IndiceCentralCursor);
  assert("pre: valid_icursor" && icursor < this->NumberOfCursors);
  if (icursor > this->IndiceCentralCursor)
  {
    assert("pre: valid_icursor" &&
      0 <= long(this->FirstCurrentNeighboorReferenceEntry + icursor) - 1 &&
      long(this->FirstCurrentNeighboorReferenceEntry + icursor) - 1 <
        long(this->ReferenceEntries.size()));
    assert("pre: valid_icursor" &&
      this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry + icursor - 1] <
        this->Entries.size());
    return this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry + icursor - 1];
  }
  else
  {
    assert("pre: valid_icursor" &&
      this->FirstCurrentNeighboorReferenceEntry + icursor < this->ReferenceEntries.size());
    assert("pre: valid_icursor" &&
      this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry + icursor] <
        this->Entries.size());
    return this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry + icursor];
  }
}

//-----------------------------------------------------------------------------

unsigned int svtkHyperTreeGridNonOrientedSuperCursor::GetIndicePreviousEntry(unsigned int icursor)
{
  assert("pre: icursor != IndiceCentralCursor" && icursor != IndiceCentralCursor);
  assert("pre: valid_icursor" && icursor < this->NumberOfCursors);
  if (icursor > this->IndiceCentralCursor)
  {
    assert("pre: valid_icursor" &&
      0 <= long(this->FirstCurrentNeighboorReferenceEntry - (this->NumberOfCursors - 1) + icursor) -
          1 &&
      long(this->FirstCurrentNeighboorReferenceEntry - (this->NumberOfCursors - 1) + icursor) - 1 <
        long(this->ReferenceEntries.size()));
    assert("pre: valid_icursor" &&
      this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry -
        (this->NumberOfCursors - 1) + icursor - 1] < this->Entries.size());
    return this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry -
      (this->NumberOfCursors - 1) + icursor - 1];
  }
  else
  {
    assert("pre: valid_icursor" &&
      this->FirstCurrentNeighboorReferenceEntry - (this->NumberOfCursors - 1) + icursor <
        this->ReferenceEntries.size());
    assert("pre: valid_icursor" &&
      this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry -
        (this->NumberOfCursors - 1) + icursor] < this->Entries.size());
    return this->ReferenceEntries[this->FirstCurrentNeighboorReferenceEntry -
      (this->NumberOfCursors - 1) + icursor];
  }
}
