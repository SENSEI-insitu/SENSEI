/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUniformGridAMRDataIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkUniformGridAMRDataIterator.h"
#include "svtkAMRDataInternals.h"
#include "svtkAMRInformation.h"
#include "svtkDataObject.h"
#include "svtkInformation.h"
#include "svtkObjectFactory.h"
#include "svtkUniformGrid.h"
#include "svtkUniformGridAMR.h"
#include <cassert>

//----------------------------------------------------------------
class AMRIndexIterator : public svtkObject
{
public:
  static AMRIndexIterator* New();
  svtkTypeMacro(AMRIndexIterator, svtkObject);

  void Initialize(const std::vector<int>* numBlocks)
  {
    assert(numBlocks && !numBlocks->empty());
    this->Level = 0;
    this->Index = -1;
    this->NumBlocks = numBlocks;
    this->NumLevels = this->GetNumberOfLevels();
    this->Next();
  }
  void Next()
  {
    this->AdvanceIndex();
    // advanc the level either when we are at the right level of out of levels
    while (this->Level < this->NumLevels &&
      static_cast<unsigned int>(this->Index) >= this->GetNumberOfBlocks(this->Level + 1))
    {
      this->Level++;
    }
  }
  virtual bool IsDone() { return this->Level >= this->NumLevels; }
  unsigned int GetLevel() { return this->Level; }
  unsigned int GetId() { return this->Index - this->GetNumberOfBlocks(this->Level); }
  virtual unsigned int GetFlatIndex() { return this->Index; }

protected:
  AMRIndexIterator()
    : Level(0)
    , Index(0)
  {
  }
  ~AMRIndexIterator() override = default;
  unsigned int Level;
  int Index;
  unsigned int NumLevels;
  const std::vector<int>* NumBlocks;
  virtual void AdvanceIndex() { this->Index++; }
  virtual unsigned int GetNumberOfLevels()
  {
    return static_cast<unsigned int>(this->NumBlocks->size() - 1);
  }
  virtual unsigned int GetNumberOfBlocks(int i)
  {
    assert(i < static_cast<int>(this->NumBlocks->size()));
    return (*this->NumBlocks)[i];
  }
};
svtkStandardNewMacro(AMRIndexIterator);

//----------------------------------------------------------------

class AMRLoadedDataIndexIterator : public AMRIndexIterator
{
public:
  static AMRLoadedDataIndexIterator* New();
  svtkTypeMacro(AMRLoadedDataIndexIterator, AMRIndexIterator);
  AMRLoadedDataIndexIterator() = default;
  void Initialize(
    const std::vector<int>* numBlocks, const svtkAMRDataInternals::BlockList* dataBlocks)
  {
    assert(numBlocks && !numBlocks->empty());
    this->Level = 0;
    this->InternalIdx = -1;
    this->NumBlocks = numBlocks;
    this->DataBlocks = dataBlocks;
    this->NumLevels = this->GetNumberOfLevels();
    this->Next();
  }

protected:
  void AdvanceIndex() override
  {
    this->InternalIdx++;
    Superclass::Index = static_cast<size_t>(this->InternalIdx) < this->DataBlocks->size()
      ? (*this->DataBlocks)[this->InternalIdx].Index
      : 0;
  }
  bool IsDone() override
  {
    return static_cast<size_t>(this->InternalIdx) >= this->DataBlocks->size();
  }
  const svtkAMRDataInternals::BlockList* DataBlocks;
  int InternalIdx;

private:
  AMRLoadedDataIndexIterator(const AMRLoadedDataIndexIterator&) = delete;
  void operator=(const AMRLoadedDataIndexIterator&) = delete;
};
svtkStandardNewMacro(AMRLoadedDataIndexIterator);

//----------------------------------------------------------------

svtkStandardNewMacro(svtkUniformGridAMRDataIterator);

svtkUniformGridAMRDataIterator::svtkUniformGridAMRDataIterator()
{
  this->Information = svtkSmartPointer<svtkInformation>::New();
  this->AMR = nullptr;
  this->AMRData = nullptr;
  this->AMRInfo = nullptr;
}

svtkUniformGridAMRDataIterator::~svtkUniformGridAMRDataIterator() = default;

svtkDataObject* svtkUniformGridAMRDataIterator::GetCurrentDataObject()
{
  unsigned int level, id;
  this->GetCurrentIndexPair(level, id);
  svtkDataObject* obj = this->AMR->GetDataSet(level, id);
  return obj;
}

svtkInformation* svtkUniformGridAMRDataIterator::GetCurrentMetaData()
{
  double bounds[6];
  this->AMRInfo->GetBounds(this->GetCurrentLevel(), this->GetCurrentIndex(), bounds);
  this->Information->Set(svtkDataObject::BOUNDING_BOX(), bounds, 6);
  return this->Information;
}

unsigned int svtkUniformGridAMRDataIterator::GetCurrentFlatIndex()
{
  return this->Iter->GetFlatIndex();
}

void svtkUniformGridAMRDataIterator::GetCurrentIndexPair(unsigned int& level, unsigned int& id)
{
  level = this->Iter->GetLevel();
  id = this->Iter->GetId();
}

unsigned int svtkUniformGridAMRDataIterator::GetCurrentLevel()
{
  unsigned int level, id;
  this->GetCurrentIndexPair(level, id);
  return level;
}

unsigned int svtkUniformGridAMRDataIterator::GetCurrentIndex()
{
  unsigned int level, id;
  this->GetCurrentIndexPair(level, id);
  return id;
}

void svtkUniformGridAMRDataIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

void svtkUniformGridAMRDataIterator::GoToFirstItem()
{
  if (!this->DataSet)
  {
    return;
  }
  this->AMR = svtkUniformGridAMR::SafeDownCast(this->DataSet);
  this->AMRInfo = this->AMR->GetAMRInfo();
  this->AMRData = this->AMR->GetAMRData();

  if (this->AMRInfo)
  {
    if (this->GetSkipEmptyNodes())
    {
      svtkSmartPointer<AMRLoadedDataIndexIterator> itr =
        svtkSmartPointer<AMRLoadedDataIndexIterator>::New();
      itr->Initialize(&this->AMRInfo->GetNumBlocks(), &this->AMR->GetAMRData()->GetAllBlocks());
      this->Iter = itr;
    }
    else
    {
      this->Iter = svtkSmartPointer<AMRIndexIterator>::New();
      this->Iter->Initialize(&this->AMRInfo->GetNumBlocks());
    }
  }
}

void svtkUniformGridAMRDataIterator::GoToNextItem()
{
  this->Iter->Next();
}

//----------------------------------------------------------------------------
int svtkUniformGridAMRDataIterator::IsDoneWithTraversal()
{
  return (!this->Iter) || this->Iter->IsDone();
}
