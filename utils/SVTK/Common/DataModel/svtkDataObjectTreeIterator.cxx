/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataObjectTreeIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkDataObjectTreeIterator.h"

#include "svtkDataObjectTree.h"
#include "svtkDataObjectTreeInternals.h"
#include "svtkObjectFactory.h"

class svtkDataObjectTreeIterator::svtkInternals
{
public:
  // This implements a simple, no frills, depth-first iterator that iterates
  // over the composite dataset.
  class svtkIterator
  {
    svtkDataObject* DataObject;
    svtkDataObjectTree* CompositeDataSet;

    svtkDataObjectTreeInternals::Iterator Iter;
    svtkDataObjectTreeInternals::ReverseIterator ReverseIter;
    svtkIterator* ChildIterator;

    svtkInternals* Parent;
    bool Reverse;
    bool PassSelf;
    unsigned int ChildIndex;

    void InitChildIterator()
    {
      if (!this->ChildIterator)
      {
        this->ChildIterator = new svtkIterator(this->Parent);
      }
      this->ChildIterator->Initialize(this->Reverse, nullptr);

      if (this->Reverse &&
        this->ReverseIter != this->GetInternals(this->CompositeDataSet)->Children.rend())
      {
        this->ChildIterator->Initialize(this->Reverse, this->ReverseIter->DataObject);
      }
      else if (!this->Reverse &&
        this->Iter != this->GetInternals(this->CompositeDataSet)->Children.end())
      {
        this->ChildIterator->Initialize(this->Reverse, this->Iter->DataObject);
      }
    }

    svtkDataObjectTreeInternals* GetInternals(svtkDataObjectTree* cd)
    {
      return this->Parent->GetInternals(cd);
    }

  public:
    svtkIterator(svtkInternals* parent)
    {
      this->ChildIterator = nullptr;
      this->Parent = parent;
    }

    ~svtkIterator()
    {
      delete this->ChildIterator;
      this->ChildIterator = nullptr;
    }

    void Initialize(bool reverse, svtkDataObject* dataObj)
    {
      svtkDataObjectTree* compositeData = svtkDataObjectTree::SafeDownCast(dataObj);
      this->Reverse = reverse;
      this->DataObject = dataObj;
      this->CompositeDataSet = compositeData;
      this->ChildIndex = 0;
      this->PassSelf = true;

      delete this->ChildIterator;
      this->ChildIterator = nullptr;

      if (compositeData)
      {
        this->Iter = this->GetInternals(compositeData)->Children.begin();
        this->ReverseIter = this->GetInternals(compositeData)->Children.rbegin();
        this->InitChildIterator();
      }
    }

    bool InSubTree()
    {
      if (this->PassSelf || this->IsDoneWithTraversal())
      {
        return false;
      }

      if (!this->ChildIterator)
      {
        return false;
      }

      if (this->ChildIterator->PassSelf)
      {
        return false;
      }

      return true;
    }

    bool IsDoneWithTraversal()
    {
      if (!this->DataObject)
      {
        return true;
      }

      if (this->PassSelf)
      {
        return false;
      }

      if (!this->CompositeDataSet)
      {
        return true;
      }

      if (this->Reverse &&
        this->ReverseIter == this->GetInternals(this->CompositeDataSet)->Children.rend())
      {
        return true;
      }

      if (!this->Reverse &&
        this->Iter == this->GetInternals(this->CompositeDataSet)->Children.end())
      {
        return true;
      }
      return false;
    }

    // Should not be called is this->IsDoneWithTraversal() returns true.
    svtkDataObject* GetCurrentDataObject()
    {
      if (this->PassSelf)
      {
        return this->DataObject;
      }
      return this->ChildIterator ? this->ChildIterator->GetCurrentDataObject() : nullptr;
    }

    svtkInformation* GetCurrentMetaData()
    {
      if (this->PassSelf || !this->ChildIterator)
      {
        return nullptr;
      }

      if (this->ChildIterator->PassSelf)
      {
        if (this->Reverse)
        {
          if (!this->ReverseIter->MetaData)
          {
            this->ReverseIter->MetaData.TakeReference(svtkInformation::New());
          }
          return this->ReverseIter->MetaData;
        }
        else
        {
          if (!this->Iter->MetaData)
          {
            this->Iter->MetaData.TakeReference(svtkInformation::New());
          }
          return this->Iter->MetaData;
        }
      }
      return this->ChildIterator->GetCurrentMetaData();
    }

    int HasCurrentMetaData()
    {
      if (this->PassSelf || !this->ChildIterator)
      {
        return 0;
      }

      if (this->ChildIterator->PassSelf)
      {
        return this->Reverse ? (this->ReverseIter->MetaData != nullptr)
                             : (this->Iter->MetaData != nullptr);
      }

      return this->ChildIterator->HasCurrentMetaData();
    }

    // Go to the next element.
    void Next()
    {
      if (this->PassSelf)
      {
        this->PassSelf = false;
      }
      else if (this->ChildIterator)
      {
        this->ChildIterator->Next();
        if (this->ChildIterator->IsDoneWithTraversal())
        {
          this->ChildIndex++;
          if (this->Reverse)
          {
            ++this->ReverseIter;
          }
          else
          {
            ++this->Iter;
          }
          this->InitChildIterator();
        }
      }
    }

    // Returns the full-tree index for the current location.
    svtkDataObjectTreeIndex GetCurrentIndex()
    {
      svtkDataObjectTreeIndex index;
      if (this->PassSelf || this->IsDoneWithTraversal() || !this->ChildIterator)
      {
        return index;
      }
      index.push_back(this->ChildIndex);
      svtkDataObjectTreeIndex childIndex = this->ChildIterator->GetCurrentIndex();
      index.insert(index.end(), childIndex.begin(), childIndex.end());
      return index;
    }
  };

  // Description:
  // Helper method used by svtkInternals to get access to the internals of
  // svtkDataObjectTree.
  svtkDataObjectTreeInternals* GetInternals(svtkDataObjectTree* cd)
  {
    return this->CompositeDataIterator->GetInternals(cd);
  }

  svtkInternals() { this->Iterator = new svtkIterator(this); }
  ~svtkInternals()
  {
    delete this->Iterator;
    this->Iterator = nullptr;
  }

  svtkIterator* Iterator;
  svtkDataObjectTreeIterator* CompositeDataIterator;
};

svtkStandardNewMacro(svtkDataObjectTreeIterator);
//----------------------------------------------------------------------------
svtkDataObjectTreeIterator::svtkDataObjectTreeIterator()
{
  this->VisitOnlyLeaves = 1;
  this->TraverseSubTree = 1;
  this->CurrentFlatIndex = 0;
  this->Internals = new svtkInternals();
  this->Internals->CompositeDataIterator = this;
}

//----------------------------------------------------------------------------
svtkDataObjectTreeIterator::~svtkDataObjectTreeIterator()
{
  delete this->Internals;
}

//----------------------------------------------------------------------------
int svtkDataObjectTreeIterator::IsDoneWithTraversal()
{
  return this->Internals->Iterator->IsDoneWithTraversal();
}

//----------------------------------------------------------------------------
void svtkDataObjectTreeIterator::GoToFirstItem()
{
  this->SetCurrentFlatIndex(0);
  this->Internals->Iterator->Initialize(this->Reverse != 0, this->DataSet);
  this->NextInternal();

  while (!this->Internals->Iterator->IsDoneWithTraversal())
  {
    svtkDataObject* dObj = this->Internals->Iterator->GetCurrentDataObject();
    if ((!dObj && this->SkipEmptyNodes) ||
      (this->VisitOnlyLeaves && svtkDataObjectTree::SafeDownCast(dObj)))
    {
      this->NextInternal();
    }
    else
    {
      break;
    }
  }
}

//----------------------------------------------------------------------------
void svtkDataObjectTreeIterator::GoToNextItem()
{
  if (!this->Internals->Iterator->IsDoneWithTraversal())
  {
    this->NextInternal();

    while (!this->Internals->Iterator->IsDoneWithTraversal())
    {
      svtkDataObject* dObj = this->Internals->Iterator->GetCurrentDataObject();
      if ((!dObj && this->SkipEmptyNodes) ||
        (this->VisitOnlyLeaves && svtkDataObjectTree::SafeDownCast(dObj)))
      {
        this->NextInternal();
      }
      else
      {
        break;
      }
    }
  }
}

//----------------------------------------------------------------------------
void svtkDataObjectTreeIterator::NextInternal()
{
  do
  {
    this->CurrentFlatIndex++;
    this->Internals->Iterator->Next();
  } while (!this->TraverseSubTree && this->Internals->Iterator->InSubTree());

  this->Modified();
}

//----------------------------------------------------------------------------
svtkDataObject* svtkDataObjectTreeIterator::GetCurrentDataObject()
{
  if (!this->IsDoneWithTraversal())
  {
    return this->Internals->Iterator->GetCurrentDataObject();
  }

  return nullptr;
}

//----------------------------------------------------------------------------
svtkInformation* svtkDataObjectTreeIterator::GetCurrentMetaData()
{
  if (!this->IsDoneWithTraversal())
  {
    return this->Internals->Iterator->GetCurrentMetaData();
  }

  return nullptr;
}

//----------------------------------------------------------------------------
int svtkDataObjectTreeIterator::HasCurrentMetaData()
{
  if (!this->IsDoneWithTraversal())
  {
    return this->Internals->Iterator->HasCurrentMetaData();
  }

  return 0;
}

//----------------------------------------------------------------------------
svtkDataObjectTreeIndex svtkDataObjectTreeIterator::GetCurrentIndex()
{
  return this->Internals->Iterator->GetCurrentIndex();
}

//----------------------------------------------------------------------------
unsigned int svtkDataObjectTreeIterator::GetCurrentFlatIndex()
{
  if (this->Reverse)
  {
    svtkErrorMacro("FlatIndex cannot be obtained when iterating in reverse order.");
    return 0;
  }
  return this->CurrentFlatIndex;
}

//----------------------------------------------------------------------------
svtkDataObjectTreeInternals* svtkDataObjectTreeIterator::GetInternals(svtkDataObjectTree* cd)
{
  if (cd)
  {
    return cd->Internals;
  }

  return nullptr;
}

//----------------------------------------------------------------------------
void svtkDataObjectTreeIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "VisitOnlyLeaves: " << (this->VisitOnlyLeaves ? "On" : "Off") << endl;
  os << indent << "Reverse: " << (this->Reverse ? "On" : "Off") << endl;
  os << indent << "TraverseSubTree: " << (this->TraverseSubTree ? "On" : "Off") << endl;
  os << indent << "SkipEmptyNodes: " << (this->SkipEmptyNodes ? "On" : "Off") << endl;
  os << indent << "CurrentFlatIndex: " << this->CurrentFlatIndex << endl;
}
