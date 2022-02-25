/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkRandomPool.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
=========================================================================*/
#include "svtkRandomPool.h"

#include "svtkArrayDispatch.h"
#include "svtkDataArray.h"
#include "svtkDataArrayRange.h"
#include "svtkMath.h"
#include "svtkMersenneTwister.h"
#include "svtkMinimalStandardRandomSequence.h"
#include "svtkMultiThreader.h"
#include "svtkNew.h"
#include "svtkObjectFactory.h"
#include "svtkSMPTools.h"

#include <algorithm>
#include <cassert>

svtkStandardNewMacro(svtkRandomPool);
svtkCxxSetObjectMacro(svtkRandomPool, Sequence, svtkRandomSequence);

//----------------------------------------------------------------------------
// Static methods to populate a data array.
namespace
{

// This method scales all components between (min,max)
template <typename ArrayT>
struct PopulateDA
{
  using T = svtk::GetAPIType<ArrayT>;
  const double* Pool;
  ArrayT* Array;
  T Min;
  T Max;

  PopulateDA(const double* pool, ArrayT* array, double min, double max)
    : Pool(pool)
    , Array(array)
  {
    this->Min = static_cast<T>(min);
    this->Max = static_cast<T>(max);
  }

  void Initialize() {}

  void operator()(svtkIdType dataId, svtkIdType endDataId)
  {
    const double* pool = this->Pool + dataId;
    const double* poolEnd = this->Pool + endDataId;
    const double range = static_cast<double>(this->Max - this->Min);

    auto output = svtk::DataArrayValueRange(this->Array, dataId, endDataId);

    std::transform(pool, poolEnd, output.begin(),
      [&](const double p) -> T { return this->Min + static_cast<T>(p * range); });
  }

  void Reduce() {}
};

struct PopulateLauncher
{
  template <typename ArrayT>
  void operator()(ArrayT* array, const double* pool, double min, double max) const
  {
    PopulateDA<ArrayT> popDA{ pool, array, min, max };
    svtkSMPTools::For(0, array->GetNumberOfValues(), popDA);
  }
};

// This method scales a selected component between (min,max)
template <typename ArrayT>
struct PopulateDAComponent
{
  using T = svtk::GetAPIType<ArrayT>;

  const double* Pool;
  ArrayT* Array;
  int CompNum;
  T Min;
  T Max;

  PopulateDAComponent(const double* pool, ArrayT* array, double min, double max, int compNum)
    : Pool(pool)
    , Array(array)
    , CompNum(compNum)
  {
    this->Min = static_cast<T>(min);
    this->Max = static_cast<T>(max);
  }

  void Initialize() {}

  void operator()(svtkIdType tupleId, svtkIdType endTupleId)
  {
    const int numComp = this->Array->GetNumberOfComponents();
    const double range = static_cast<double>(this->Max - this->Min);

    const svtkIdType valueId = tupleId * numComp + this->CompNum;
    const svtkIdType endValueId = endTupleId * numComp;

    const double* poolIter = this->Pool + valueId;
    const double* poolEnd = this->Pool + endValueId;

    auto data = svtk::DataArrayValueRange(this->Array, valueId, endValueId);
    auto dataIter = data.begin();

    for (; poolIter < poolEnd; dataIter += numComp, poolIter += numComp)
    {
      *dataIter = this->Min + static_cast<T>(*poolIter * range);
    }
  }

  void Reduce() {}
};

struct PopulateDAComponentLauncher
{
  template <typename ArrayT>
  void operator()(ArrayT* array, const double* pool, double min, double max, int compNum)
  {
    PopulateDAComponent<ArrayT> popDAC{ pool, array, min, max, compNum };
    svtkSMPTools::For(0, array->GetNumberOfTuples(), popDAC);
  }
};

} // anonymous namespace

// ----------------------------------------------------------------------------
svtkRandomPool::svtkRandomPool()
{
  this->Sequence = svtkMinimalStandardRandomSequence::New();
  this->Size = 0;
  this->NumberOfComponents = 1;
  this->ChunkSize = 10000;

  this->TotalSize = 0;
  this->Pool = nullptr;

  // Ensure that the modified time > generate time
  this->GenerateTime.Modified();
  this->Modified();
}

// ----------------------------------------------------------------------------
svtkRandomPool::~svtkRandomPool()
{
  this->SetSequence(nullptr);
  delete[] this->Pool;
}

//----------------------------------------------------------------------------
void svtkRandomPool::PopulateDataArray(svtkDataArray* da, double minRange, double maxRange)
{
  if (da == nullptr)
  {
    svtkWarningMacro(<< "Bad request");
    return;
  }

  svtkIdType size = da->GetNumberOfTuples();
  int numComp = da->GetNumberOfComponents();

  this->SetSize(size);
  this->SetNumberOfComponents(numComp);
  const double* pool = this->GeneratePool();
  if (pool == nullptr)
  {
    return;
  }

  // Now perform the scaling of all components
  using Dispatcher = svtkArrayDispatch::Dispatch;
  PopulateLauncher worker;
  if (!Dispatcher::Execute(da, worker, pool, minRange, maxRange))
  { // Fallback for unknown array types:
    worker(da, pool, minRange, maxRange);
  }

  // Make sure that the data array is marked modified
  da->Modified();
}

//----------------------------------------------------------------------------
void svtkRandomPool::PopulateDataArray(
  svtkDataArray* da, int compNum, double minRange, double maxRange)
{
  if (da == nullptr)
  {
    svtkWarningMacro(<< "Bad request");
    return;
  }

  svtkIdType size = da->GetNumberOfTuples();
  int numComp = da->GetNumberOfComponents();
  compNum = (compNum < 0 ? 0 : (compNum >= numComp ? numComp - 1 : compNum));

  this->SetSize(size);
  this->SetNumberOfComponents(numComp);
  const double* pool = this->GeneratePool();
  if (pool == nullptr)
  {
    return;
  }

  // Now perform the scaling for one of the components
  using Dispatcher = svtkArrayDispatch::Dispatch;
  PopulateDAComponentLauncher worker;
  if (!Dispatcher::Execute(da, worker, pool, minRange, maxRange, compNum))
  { // fallback
    worker(da, pool, minRange, maxRange, compNum);
  }

  // Make sure that the data array is marked modified
  da->Modified();
}

//----------------------------------------------------------------------------
// Support multithreading of sequence generation
struct svtkRandomPoolInfo
{
  svtkIdType NumThreads;
  svtkRandomSequence** Sequencer;
  double* Pool;
  svtkIdType SeqSize;
  svtkIdType SeqChunk;
  svtkRandomSequence* Sequence;

  svtkRandomPoolInfo(double* pool, svtkIdType seqSize, svtkIdType seqChunk, svtkIdType numThreads,
    svtkRandomSequence* ranSeq)
    : NumThreads(numThreads)
    , Pool(pool)
    , SeqSize(seqSize)
    , SeqChunk(seqChunk)
    , Sequence(ranSeq)
  {
    this->Sequencer = new svtkRandomSequence*[numThreads];
    for (svtkIdType i = 0; i < numThreads; ++i)
    {
      this->Sequencer[i] = ranSeq->NewInstance();
      assert(this->Sequencer[i] != nullptr);
      this->Sequencer[i]->Initialize(static_cast<svtkTypeUInt32>(i));
    }
  }

  ~svtkRandomPoolInfo()
  {
    for (svtkIdType i = 0; i < this->NumThreads; ++i)
    {
      this->Sequencer[i]->Delete();
    }
    delete[] this->Sequencer;
  }
};

//----------------------------------------------------------------------------
// This is the multithreaded piece of random sequence generation.
static SVTK_THREAD_RETURN_TYPE svtkRandomPool_ThreadedMethod(void* arg)
{
  // Grab input
  svtkRandomPoolInfo* info;
  int threadId;

  threadId = ((svtkMultiThreader::ThreadInfo*)(arg))->ThreadID;
  info = (svtkRandomPoolInfo*)(((svtkMultiThreader::ThreadInfo*)(arg))->UserData);

  // Generate subsequence and place into global sequence in correct spot
  svtkRandomSequence* sequencer = info->Sequencer[threadId];
  double* pool = info->Pool;
  svtkIdType i, start = threadId * info->SeqChunk;
  svtkIdType end = start + info->SeqChunk;
  end = (end < info->SeqSize ? end : info->SeqSize);
  for (i = start; i < end; ++i, sequencer->Next())
  {
    pool[i] = sequencer->GetValue();
  }

  return SVTK_THREAD_RETURN_VALUE;
}

// ----------------------------------------------------------------------------
// May use threaded sequence generation if the length of the sequence is
// greater than a pre-defined work size.
const double* svtkRandomPool::GeneratePool()
{
  // Return if generation has already occurred
  if (this->GenerateTime > this->MTime)
  {
    return this->Pool;
  }

  // Check for valid input and correct if necessary
  this->TotalSize = this->Size * this->NumberOfComponents;
  if (this->TotalSize <= 0 || this->Sequence == nullptr)
  {
    svtkWarningMacro(<< "Bad pool size");
    this->Size = this->TotalSize = 1000;
    this->NumberOfComponents = 1;
  }
  this->ChunkSize = (this->ChunkSize < 1000 ? 1000 : this->ChunkSize);
  this->Pool = new double[this->TotalSize];

  // Control the number of threads spawned.
  svtkIdType seqSize = this->TotalSize;
  svtkIdType seqChunk = this->ChunkSize;
  svtkIdType numThreads = (seqSize / seqChunk) + 1;
  svtkRandomSequence* sequencer = this->Sequence;

  // Fast path don't spin up threads
  if (numThreads == 1)
  {
    sequencer->Initialize(31415);
    double* p = this->Pool;
    for (svtkIdType i = 0; i < seqSize; ++i, sequencer->Next())
    {
      *p++ = sequencer->GetValue();
    }
  }

  // Otherwise spawn threads to fill in chunks of the sequence.
  else
  {
    svtkNew<svtkMultiThreader> threader;
    threader->SetNumberOfThreads(numThreads);
    svtkIdType actualThreads = threader->GetNumberOfThreads();
    if (actualThreads < numThreads) // readjust work load
    {
      numThreads = actualThreads;
    }

    // Now distribute work
    svtkRandomPoolInfo info(this->Pool, seqSize, seqChunk, numThreads, this->Sequence);
    threader->SetSingleMethod(svtkRandomPool_ThreadedMethod, (void*)&info);
    threader->SingleMethodExecute();
  } // spawning threads

  // Update generation time
  this->GenerateTime.Modified();
  return this->Pool;
}

// ----------------------------------------------------------------------------
void svtkRandomPool::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Sequence: " << this->Sequence << "\n";
  os << indent << "Size: " << this->Size << "\n";
  os << indent << "Number Of Components: " << this->NumberOfComponents << "\n";
  os << indent << "Chunk Size: " << this->ChunkSize << "\n";
}
