/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHeap.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkHeap.h"
#include "svtkCommonMiscModule.h" // For export macro
#include "svtkObjectFactory.h"
#include <cstddef>

svtkStandardNewMacro(svtkHeap);

static size_t svtkGetLongAlignment()
{
  struct svtkTestAlignLong
  {
    char pad;
    long x;
  };

  return offsetof(svtkTestAlignLong, x);
}

class SVTKCOMMONMISC_EXPORT svtkHeapBlock
{
public:
  char* Data;
  svtkHeapBlock* Next;
  size_t Size; // Variable size guards against block size changing from SetBlockSize()
               // or large requests greater than the standard block size.

  svtkHeapBlock(size_t size)
    : Next(nullptr)
    , Size(size)
  {
    this->Data = new char[size];
  }
  ~svtkHeapBlock() { delete[] this->Data; }
};

svtkHeap::svtkHeap()
{
  this->BlockSize = 256000;
  this->NumberOfBlocks = 0;
  this->NumberOfAllocations = 0;
  this->Alignment = svtkGetLongAlignment();
  this->First = nullptr;
  this->Last = nullptr;
  this->Current = nullptr;
  this->Position = 0;
}

svtkHeap::~svtkHeap()
{
  this->CleanAll();
}

void svtkHeap::SetBlockSize(size_t _arg)
{
  svtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting BlockSize to "
                << static_cast<int>(_arg));
  if (this->BlockSize != _arg)
  {
    this->BlockSize = _arg;
    this->Modified();
  }
}

void* svtkHeap::AllocateMemory(size_t n)
{
  if (n % this->Alignment) // 4-byte word alignment
  {
    n += this->Alignment - (n % this->Alignment);
  }

  size_t blockSize = (n > this->BlockSize ? n : this->BlockSize);
  this->NumberOfAllocations++;

  if (!this->Current || (this->Position + n) >= this->Current->Size)
  {
    this->Add(blockSize);
  }

  char* ptr = this->Current->Data + this->Position;
  this->Position += n;

  return ptr;
}

// If a Reset() was invoked, then we reuse memory (i.e., the list of blocks)
// or allocate it as necessary. Otherwise a block is allocated and placed into
// the list of blocks.
void svtkHeap::Add(size_t blockSize)
{
  this->Position = 0; // reset to the beginning of the block

  if (this->Current && this->Current != this->Last &&
    this->Current->Next->Size >= blockSize) // reuse
  {
    this->Current = this->Current->Next;
  }

  else // allocate a new block
  {
    this->NumberOfBlocks++;
    svtkHeapBlock* block = new svtkHeapBlock(blockSize);

    if (!this->Last)
    {
      this->First = block;
      this->Current = block;
      this->Last = block;
      return;
    }

    this->Last->Next = block;
    this->Last = block;
    this->Current = block;
  }
}

void svtkHeap::CleanAll()
{
  this->Current = this->First;
  if (!this->Current)
  {
    return;
  }
  while (this->DeleteAndNext())
  {
    ;
  }
  this->First = this->Current = this->Last = nullptr;
  this->Position = 0;
}

svtkHeapBlock* svtkHeap::DeleteAndNext()
{
  if (this->Current)
  {
    svtkHeapBlock* tmp = this->Current;
    this->Current = this->Current->Next;
    delete tmp;
    return this->Current;
  }
  else
  {
    return nullptr;
  }
}

void svtkHeap::Reset()
{
  this->Current = this->First;
  this->Position = 0;
}

char* svtkHeap::StringDup(const char* str)
{
  char* newStr = static_cast<char*>(this->AllocateMemory(strlen(str) + 1));
  strcpy(newStr, str);
  return newStr;
}

void svtkHeap::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Block Size: " << static_cast<int>(this->BlockSize) << "\n";
  os << indent << "Number of Blocks: " << this->NumberOfBlocks << "\n";
  os << indent << "Number of Allocations: " << this->NumberOfAllocations << "\n";
  os << indent << "Current bytes allocated: "
     << ((this->NumberOfBlocks - 1) * static_cast<int>(this->BlockSize) +
          static_cast<int>(this->Position))
     << "\n";
}
