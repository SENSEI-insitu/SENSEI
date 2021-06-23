/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataObjectTree.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkDataObjectTree.h"

#include "svtkDataObjectTreeInternals.h"
#include "svtkDataObjectTreeIterator.h"
#include "svtkDataSet.h"
#include "svtkInformation.h"
#include "svtkInformationIntegerKey.h"
#include "svtkInformationStringKey.h"
#include "svtkInformationVector.h"
#include "svtkMultiPieceDataSet.h"
#include "svtkObjectFactory.h"

//----------------------------------------------------------------------------
svtkDataObjectTree::svtkDataObjectTree()
{
  this->Internals = new svtkDataObjectTreeInternals;
}

//----------------------------------------------------------------------------
svtkDataObjectTree::~svtkDataObjectTree()
{
  delete this->Internals;
}

//----------------------------------------------------------------------------
svtkDataObjectTree* svtkDataObjectTree::GetData(svtkInformation* info)
{
  return info ? svtkDataObjectTree::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkDataObjectTree* svtkDataObjectTree::GetData(svtkInformationVector* v, int i)
{
  return svtkDataObjectTree::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
void svtkDataObjectTree::SetNumberOfChildren(unsigned int num)
{
  this->Internals->Children.resize(num);
  this->Modified();
}

//----------------------------------------------------------------------------
unsigned int svtkDataObjectTree::GetNumberOfChildren()
{
  return static_cast<unsigned int>(this->Internals->Children.size());
}

//----------------------------------------------------------------------------
void svtkDataObjectTree::SetChild(unsigned int index, svtkDataObject* dobj)
{
  if (this->Internals->Children.size() <= index)
  {
    this->SetNumberOfChildren(index + 1);
  }

  svtkDataObjectTreeItem& item = this->Internals->Children[index];
  if (item.DataObject != dobj)
  {
    item.DataObject = dobj;
    this->Modified();
  }
}

//----------------------------------------------------------------------------
void svtkDataObjectTree::RemoveChild(unsigned int index)
{
  if (this->Internals->Children.size() <= index)
  {
    svtkErrorMacro("The input index is out of range.");
    return;
  }

  svtkDataObjectTreeItem& item = this->Internals->Children[index];
  item.DataObject = nullptr;
  this->Internals->Children.erase(this->Internals->Children.begin() + index);
  this->Modified();
}

//----------------------------------------------------------------------------
svtkDataObject* svtkDataObjectTree::GetChild(unsigned int index)
{
  if (index < this->Internals->Children.size())
  {
    return this->Internals->Children[index].DataObject;
  }

  return nullptr;
}

//----------------------------------------------------------------------------
svtkInformation* svtkDataObjectTree::GetChildMetaData(unsigned int index)
{
  if (index < this->Internals->Children.size())
  {
    svtkDataObjectTreeItem& item = this->Internals->Children[index];
    if (!item.MetaData)
    {
      // New svtkInformation is allocated is none is already present.
      item.MetaData.TakeReference(svtkInformation::New());
    }
    return item.MetaData;
  }
  return nullptr;
}

//----------------------------------------------------------------------------
void svtkDataObjectTree::SetChildMetaData(unsigned int index, svtkInformation* info)
{
  if (this->Internals->Children.size() <= index)
  {
    this->SetNumberOfChildren(index + 1);
  }

  svtkDataObjectTreeItem& item = this->Internals->Children[index];
  item.MetaData = info;
}

//----------------------------------------------------------------------------
int svtkDataObjectTree::HasChildMetaData(unsigned int index)
{
  if (index < this->Internals->Children.size())
  {
    svtkDataObjectTreeItem& item = this->Internals->Children[index];
    return (item.MetaData != nullptr) ? 1 : 0;
  }

  return 0;
}

//----------------------------------------------------------------------------
void svtkDataObjectTree::CopyStructure(svtkCompositeDataSet* compositeSource)
{
  if (!compositeSource)
  {
    return;
  }
  svtkDataObjectTree* source = svtkDataObjectTree::SafeDownCast(compositeSource);
  if (source == this)
  {
    return;
  }

  this->Internals->Children.clear();
  if (!source)
  {
    // WARNING:
    // If we copy the structure of from a non-tree composite data set
    // we create a special structure of two levels, the first level
    // is just a single multipiece and the second level are all the data sets.
    // This is likely to change in the future!
    svtkMultiPieceDataSet* mds = svtkMultiPieceDataSet::New();
    this->SetChild(0, mds);
    mds->Delete();

    svtkInformation* info = svtkInformation::New();
    info->Set(svtkCompositeDataSet::NAME(), "All Blocks");
    this->SetChildMetaData(0, info);
    info->FastDelete();

    int totalNumBlocks = 0;
    svtkCompositeDataIterator* iter = compositeSource->NewIterator();
    iter->SkipEmptyNodesOff();
    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
    {
      totalNumBlocks++;
    }
    iter->Delete();

    mds->SetNumberOfChildren(totalNumBlocks);
    return;
  }

  this->Internals->Children.resize(source->Internals->Children.size());

  svtkDataObjectTreeInternals::Iterator srcIter = source->Internals->Children.begin();
  svtkDataObjectTreeInternals::Iterator myIter = this->Internals->Children.begin();
  for (; srcIter != source->Internals->Children.end(); ++srcIter, ++myIter)
  {
    svtkDataObjectTree* compositeSrc = svtkDataObjectTree::SafeDownCast(srcIter->DataObject);
    if (compositeSrc)
    {
      svtkDataObjectTree* copy = compositeSrc->NewInstance();
      myIter->DataObject.TakeReference(copy);
      copy->CopyStructure(compositeSrc);
    }

    // shallow copy meta data.
    if (srcIter->MetaData)
    {
      svtkInformation* info = svtkInformation::New();
      info->Copy(srcIter->MetaData, /*deep=*/0);
      myIter->MetaData = info;
      info->FastDelete();
    }
  }
  this->Modified();
}

//----------------------------------------------------------------------------
svtkDataObjectTreeIterator* svtkDataObjectTree::NewTreeIterator()
{
  svtkDataObjectTreeIterator* iter = svtkDataObjectTreeIterator::New();
  iter->SetDataSet(this);
  return iter;
}

//----------------------------------------------------------------------------
svtkCompositeDataIterator* svtkDataObjectTree::NewIterator()
{
  return this->NewTreeIterator();
}

//----------------------------------------------------------------------------
void svtkDataObjectTree::SetDataSet(svtkCompositeDataIterator* iter, svtkDataObject* dataObj)
{
  svtkDataObjectTreeIterator* treeIter = svtkDataObjectTreeIterator::SafeDownCast(iter);
  if (treeIter)
  {
    this->SetDataSetFrom(treeIter, dataObj);
    return;
  }

  if (!iter || iter->IsDoneWithTraversal())
  {
    svtkErrorMacro("Invalid iterator location.");
    return;
  }

  // WARNING: We are doing something special here. See comments
  // in CopyStructure()

  unsigned int index = iter->GetCurrentFlatIndex();
  if (this->GetNumberOfChildren() != 1)
  {
    svtkErrorMacro("Structure is not expected. Did you forget to use copy structure?");
    return;
  }
  svtkMultiPieceDataSet* parent = svtkMultiPieceDataSet::SafeDownCast(this->GetChild(0));
  if (!parent)
  {
    svtkErrorMacro("Structure is not expected. Did you forget to use copy structure?");
    return;
  }
  parent->SetChild(index, dataObj);
}

//----------------------------------------------------------------------------
void svtkDataObjectTree::SetDataSetFrom(svtkDataObjectTreeIterator* iter, svtkDataObject* dataObj)
{
  if (!iter || iter->IsDoneWithTraversal())
  {
    svtkErrorMacro("Invalid iterator location.");
    return;
  }

  svtkDataObjectTreeIndex index = iter->GetCurrentIndex();

  if (index.empty())
  {
    // Sanity check.
    svtkErrorMacro("Invalid index returned by iterator.");
    return;
  }

  svtkDataObjectTree* parent = this;
  int numIndices = static_cast<int>(index.size());
  for (int cc = 0; cc < numIndices - 1; cc++)
  {
    if (!parent || parent->GetNumberOfChildren() <= index[cc])
    {
      svtkErrorMacro("Structure does not match. "
                    "You must use CopyStructure before calling this method.");
      return;
    }
    parent = svtkDataObjectTree::SafeDownCast(parent->GetChild(index[cc]));
  }

  if (!parent || parent->GetNumberOfChildren() <= index.back())
  {
    svtkErrorMacro("Structure does not match. "
                  "You must use CopyStructure before calling this method.");
    return;
  }

  parent->SetChild(index.back(), dataObj);
}

//----------------------------------------------------------------------------
svtkDataObject* svtkDataObjectTree::GetDataSet(svtkCompositeDataIterator* compositeIter)
{
  if (!compositeIter || compositeIter->IsDoneWithTraversal())
  {
    svtkErrorMacro("Invalid iterator location.");
    return nullptr;
  }

  svtkDataObjectTreeIterator* iter = svtkDataObjectTreeIterator::SafeDownCast(compositeIter);
  if (!iter)
  {
    // WARNING: We are doing something special here. See comments
    // in CopyStructure()
    // To do: More clear check of structures here. At least something like this->Depth()==1
    unsigned int currentFlatIndex = compositeIter->GetCurrentFlatIndex();

    if (this->GetNumberOfChildren() != 1)
    {
      svtkErrorMacro("Structure is not expected. Did you forget to use copy structure?");
      return nullptr;
    }
    svtkMultiPieceDataSet* parent = svtkMultiPieceDataSet::SafeDownCast(this->GetChild(0));
    if (!parent)
    {
      svtkErrorMacro("Structure is not expected. Did you forget to use copy structure?");
      return nullptr;
    }

    if (currentFlatIndex < parent->GetNumberOfChildren())
    {
      return parent->GetChild(currentFlatIndex);
    }
    else
    {
      return nullptr;
    }
  }

  svtkDataObjectTreeIndex index = iter->GetCurrentIndex();

  if (index.empty())
  {
    // Sanity check.
    svtkErrorMacro("Invalid index returned by iterator.");
    return nullptr;
  }

  svtkDataObjectTree* parent = this;
  int numIndices = static_cast<int>(index.size());
  for (int cc = 0; cc < numIndices - 1; cc++)
  {
    if (!parent || parent->GetNumberOfChildren() <= index[cc])
    {
      svtkErrorMacro("Structure does not match. "
                    "You must use CopyStructure before calling this method.");
      return nullptr;
    }
    parent = svtkDataObjectTree::SafeDownCast(parent->GetChild(index[cc]));
  }

  if (!parent || parent->GetNumberOfChildren() <= index.back())
  {
    svtkErrorMacro("Structure does not match. "
                  "You must use CopyStructure before calling this method.");
    return nullptr;
  }

  return parent->GetChild(index.back());
}

//----------------------------------------------------------------------------
svtkInformation* svtkDataObjectTree::GetMetaData(svtkCompositeDataIterator* compositeIter)
{
  svtkDataObjectTreeIterator* iter = svtkDataObjectTreeIterator::SafeDownCast(compositeIter);
  if (!iter || iter->IsDoneWithTraversal())
  {
    svtkErrorMacro("Invalid iterator location.");
    return nullptr;
  }

  svtkDataObjectTreeIndex index = iter->GetCurrentIndex();

  if (index.empty())
  {
    // Sanity check.
    svtkErrorMacro("Invalid index returned by iterator.");
    return nullptr;
  }

  svtkDataObjectTree* parent = this;
  int numIndices = static_cast<int>(index.size());
  for (int cc = 0; cc < numIndices - 1; cc++)
  {
    if (!parent || parent->GetNumberOfChildren() <= index[cc])
    {
      svtkErrorMacro("Structure does not match. "
                    "You must use CopyStructure before calling this method.");
      return nullptr;
    }
    parent = svtkDataObjectTree::SafeDownCast(parent->GetChild(index[cc]));
  }

  if (!parent || parent->GetNumberOfChildren() <= index.back())
  {
    svtkErrorMacro("Structure does not match. "
                  "You must use CopyStructure before calling this method.");
    return nullptr;
  }

  return parent->GetChildMetaData(index.back());
}

//----------------------------------------------------------------------------
int svtkDataObjectTree::HasMetaData(svtkCompositeDataIterator* compositeIter)
{
  svtkDataObjectTreeIterator* iter = svtkDataObjectTreeIterator::SafeDownCast(compositeIter);
  if (!iter || iter->IsDoneWithTraversal())
  {
    svtkErrorMacro("Invalid iterator location.");
    return 0;
  }

  svtkDataObjectTreeIndex index = iter->GetCurrentIndex();

  if (index.empty())
  {
    // Sanity check.
    svtkErrorMacro("Invalid index returned by iterator.");
    return 0;
  }

  svtkDataObjectTree* parent = this;
  int numIndices = static_cast<int>(index.size());
  for (int cc = 0; cc < numIndices - 1; cc++)
  {
    if (!parent || parent->GetNumberOfChildren() <= index[cc])
    {
      svtkErrorMacro("Structure does not match. "
                    "You must use CopyStructure before calling this method.");
      return 0;
    }
    parent = svtkDataObjectTree::SafeDownCast(parent->GetChild(index[cc]));
  }

  if (!parent || parent->GetNumberOfChildren() <= index.back())
  {
    svtkErrorMacro("Structure does not match. "
                  "You must use CopyStructure before calling this method.");
    return 0;
  }

  return parent->HasChildMetaData(index.back());
}

//----------------------------------------------------------------------------
void svtkDataObjectTree::ShallowCopy(svtkDataObject* src)
{
  if (src == this)
  {
    return;
  }

  this->Internals->Children.clear();
  this->Superclass::ShallowCopy(src);

  svtkDataObjectTree* from = svtkDataObjectTree::SafeDownCast(src);
  if (from)
  {
    unsigned int numChildren = from->GetNumberOfChildren();
    this->SetNumberOfChildren(numChildren);
    for (unsigned int cc = 0; cc < numChildren; cc++)
    {
      svtkDataObject* child = from->GetChild(cc);
      if (child)
      {
        if (child->IsA("svtkDataObjectTree"))
        {
          svtkDataObject* clone = child->NewInstance();
          clone->ShallowCopy(child);
          this->SetChild(cc, clone);
          clone->FastDelete();
        }
        else
        {
          this->SetChild(cc, child);
        }
      }
      if (from->HasChildMetaData(cc))
      {
        svtkInformation* toInfo = this->GetChildMetaData(cc);
        toInfo->Copy(from->GetChildMetaData(cc), /*deep=*/0);
      }
    }
  }
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkDataObjectTree::DeepCopy(svtkDataObject* src)
{
  if (src == this)
  {
    return;
  }

  this->Internals->Children.clear();
  this->Superclass::DeepCopy(src);

  svtkDataObjectTree* from = svtkDataObjectTree::SafeDownCast(src);
  if (from)
  {
    unsigned int numChildren = from->GetNumberOfChildren();
    this->SetNumberOfChildren(numChildren);
    for (unsigned int cc = 0; cc < numChildren; cc++)
    {
      svtkDataObject* fromChild = from->GetChild(cc);
      if (fromChild)
      {
        svtkDataObject* toChild = fromChild->NewInstance();
        toChild->DeepCopy(fromChild);
        this->SetChild(cc, toChild);
        toChild->FastDelete();
        if (from->HasChildMetaData(cc))
        {
          svtkInformation* toInfo = this->GetChildMetaData(cc);
          toInfo->Copy(from->GetChildMetaData(cc), /*deep=*/1);
        }
      }
    }
  }
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkDataObjectTree::Initialize()
{
  this->Internals->Children.clear();
  this->Superclass::Initialize();
}

//----------------------------------------------------------------------------
svtkIdType svtkDataObjectTree::GetNumberOfPoints()
{
  svtkIdType numPts = 0;
  svtkDataObjectTreeIterator* iter = svtkDataObjectTreeIterator::SafeDownCast(this->NewIterator());
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
  {
    svtkDataSet* ds = svtkDataSet::SafeDownCast(iter->GetCurrentDataObject());
    if (ds)
    {
      numPts += ds->GetNumberOfPoints();
    }
  }
  iter->Delete();
  return numPts;
}

//----------------------------------------------------------------------------
svtkIdType svtkDataObjectTree::GetNumberOfCells()
{
  svtkIdType numCells = 0;
  svtkDataObjectTreeIterator* iter = svtkDataObjectTreeIterator::SafeDownCast(this->NewIterator());
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
  {
    svtkDataSet* ds = svtkDataSet::SafeDownCast(iter->GetCurrentDataObject());
    if (ds)
    {
      numCells += ds->GetNumberOfCells();
    }
  }
  iter->Delete();
  return numCells;
}

//----------------------------------------------------------------------------
unsigned long svtkDataObjectTree::GetActualMemorySize()
{
  unsigned long memSize = 0;
  svtkDataObjectTreeIterator* iter = svtkDataObjectTreeIterator::SafeDownCast(this->NewIterator());
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
  {
    svtkDataObject* dobj = iter->GetCurrentDataObject();
    memSize += dobj->GetActualMemorySize();
  }
  iter->Delete();
  return memSize;
}

//----------------------------------------------------------------------------
void svtkDataObjectTree::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Number Of Children: " << this->GetNumberOfChildren() << endl;
  for (unsigned int cc = 0; cc < this->GetNumberOfChildren(); cc++)
  {
    const char* name = (this->HasChildMetaData(cc) && this->GetChildMetaData(cc)->Has(NAME()))
      ? this->GetChildMetaData(cc)->Get(NAME())
      : nullptr;

    svtkDataObject* child = this->GetChild(cc);
    if (child)
    {
      os << indent << "Child " << cc << ": " << child->GetClassName() << endl;
      os << indent << "Name: " << (name ? name : "(nullptr)") << endl;
      child->PrintSelf(os, indent.GetNextIndent());
    }
    else
    {
      os << indent << "Child " << cc << ": nullptr" << endl;
      os << indent << "Name: " << (name ? name : "(nullptr)") << endl;
    }
  }
}
