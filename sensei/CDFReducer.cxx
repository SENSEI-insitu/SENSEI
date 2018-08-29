#include <limits>

#include <vtkCommunicator.h>
#include <vtkMultiProcessController.h>

#include "CDFReducer.h"

// ----------------------------------------------------------------------------

struct Entry
{
  Entry()
    : GlobalID(0)
    , Value(0)
  {
  }
  Entry(vtkIdType id, double value)
    : GlobalID(id)
    , Value(value)
  {
  }

  bool operator<(Entry const& other) const
  {
    if (this->Value == other.Value)
    {
      return this->GlobalID > other.GlobalID;
    }
    return this->Value > other.Value;
  }

  vtkIdType GlobalID;
  double Value;
};

// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------

CDFReducer::~CDFReducer()
{
  this->LocalValues = nullptr; // We are not the owner of that array
  if (this->ReducedCDF != nullptr)
  {
    delete[] this->ReducedCDF;
    this->ReducedCDF = nullptr;
  }
  if (this->CDFOffsets != nullptr)
  {
    delete[] this->CDFOffsets;
    this->CDFOffsets = nullptr;
  }
  if (this->LocalCDF != nullptr)
  {
    delete[] this->LocalCDF;
    this->LocalCDF = nullptr;
  }
  if (this->RemoteCDFs != nullptr)
  {
    delete[] this->RemoteCDFs;
    this->RemoteCDFs = nullptr;
  }
}

// ----------------------------------------------------------------------------

double CDFReducer::GetValueAtIndex(vtkIdType targetIdx, vtkIdType pid, vtkIdType mpiSize, int depth)
{
  double lastValue = 0;
  this->ExecutionCount++;

  this->Handler.Reset(targetIdx);
  this->CDFStep = this->Handler.Step;

  // Fill local CDF
  vtkIdType localOffset = this->CDFOffsets[pid];
  for (vtkIdType i = 0; i < this->CDFSize; i++)
  {
    vtkIdType gid = localOffset + this->CDFStep + (i * this->CDFStep);
    this->LocalCDF[i] =
      (gid < this->ArraySize) ? this->LocalValues[gid] : std::numeric_limits<double>::max();
  }

  // Gather CDFs to pid(0)
  this->Controller->Gather(this->LocalCDF, this->RemoteCDFs, this->CDFSize, 0);

  // Only pid(0)
  if (pid == 0)
  {
    size_t size = (size_t)(this->CDFSize * mpiSize);
    std::vector<Entry> sortedCDFs(size);
    for (size_t i = 0; i < size; i++)
    {
      sortedCDFs[i].GlobalID = (vtkIdType)i;
      sortedCDFs[i].Value = this->RemoteCDFs[i];
    }
    std::sort(sortedCDFs.begin(), sortedCDFs.end()); // In reverse

    // Move forward only if no skip possible
    while (this->Handler.Move(sortedCDFs.back().GlobalID / this->CDFSize))
    {
      // std::cout << sortedCDFs.back().GlobalID << " => " << sortedCDFs.back().GlobalID / this->CDFSize << std::endl;
      lastValue = sortedCDFs.back().Value;
      sortedCDFs.pop_back();
    }

    // Synch up offsets
    this->Controller->Broadcast(this->CDFOffsets, mpiSize, 0);
  }
  else
  {
    // Just update offset from 0
    this->Controller->Broadcast(this->CDFOffsets, mpiSize, 0);
    this->Handler.UpdateCurrentIndex();
  }

  // if (pid == 0)
  // {
  //   for (int i = 0; i < depth; i++)
  //   {
  //     std::cout << "  ";
  //   }
  //   std::cout << " - index: " << this->Handler.CurrentIndex << " - target: " << targetIdx << std::endl;
  // }


  if (this->Handler.CurrentIndex == targetIdx)
  {
    return lastValue;
  }
  else
  {
    // Need to refine
    return this->GetValueAtIndex(targetIdx, pid, mpiSize, depth + 1);
  }
}

// ----------------------------------------------------------------------------

double* CDFReducer::Compute(double* localSortedValues,
                            vtkIdType localArraySize,
                            vtkIdType outputCDFSize)
{
  // Initialization
  vtkIdType MPI_ID = this->Controller->GetLocalProcessId();
  vtkIdType MPI_SIZE = this->Controller->GetNumberOfProcesses();

  if (this->LocalCDF == nullptr)
  {
    this->LocalCDF = new double[this->CDFSize];
  }
  if (this->RemoteCDFs == nullptr)
  {
    this->RemoteCDFs = new double[MPI_SIZE * this->CDFSize];
  }

  if (this->CDFOffsets == nullptr)
  {
    this->CDFOffsets = new vtkIdType[MPI_SIZE];
  }
  for (vtkIdType idx = 0; idx < MPI_SIZE; idx++)
  {
    this->CDFOffsets[idx] = 0;
  }

  this->ArraySize = localArraySize;

  this->LocalValues = localSortedValues;

  if (this->ReducedCDF == nullptr || this->ReducedCDFSize != outputCDFSize)
  {
    delete[] this->ReducedCDF;
    this->ReducedCDF = new double[outputCDFSize];
  }
  this->ReducedCDFSize = outputCDFSize;

  // Share basic information (min, max, counts)
  double globalMin, globalMax;
  this->Controller->AllReduce(&this->LocalValues[0], &globalMin, 1, vtkCommunicator::MIN_OP);
  this->Controller->AllReduce(
    &this->LocalValues[localArraySize - 1], &globalMax, 1, vtkCommunicator::MAX_OP);
  this->Controller->AllReduce(&localArraySize, &this->TotalCount, 1, vtkCommunicator::SUM_OP);

  this->ReducedCDF[0] = globalMin;
  this->ReducedCDF[outputCDFSize - 1] = globalMax;

  // Look for indexes we will search value for
  vtkIdType splitSize = outputCDFSize - 1;

  // Proper init
  this->Handler.Init(this->CDFOffsets, MPI_SIZE, this->CDFSize);

  // Fill resulting CDF
  double avgCallstack = 0;
  for (vtkIdType i = 1; i < splitSize; i++)
  {
    this->ExecutionCount = 0;
    this->ReducedCDF[i] = this->GetValueAtIndex(i * this->TotalCount / splitSize, MPI_ID, MPI_SIZE, 1);
    avgCallstack += double(this->ExecutionCount);
  }
  avgCallstack /= (splitSize - 1);
  if (MPI_ID == 0)
  {
    std::cout << "Avg stack size: " << avgCallstack << std::endl;
  }

  return this->ReducedCDF;
}