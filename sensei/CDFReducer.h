//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2014 Sandia Corporation.
//  Copyright 2014 UT-Battelle, LLC.
//  Copyright 2014 Los Alamos National Security.
//
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================

#ifndef CDFReducer_h
#define CDFReducer_h

class vtkMultiProcessController;

struct StepHandler
{
  StepHandler()
    : Offsets(nullptr)
    , MPISize(0)
    , CDFSize(0)
    , ProcessStepCount(nullptr)
  {}

  ~StepHandler()
  {
    delete[] this->ProcessStepCount;
    this->ProcessStepCount = nullptr;
  }

  void Init(vtkIdType* &offsets, vtkIdType mpiSize, vtkIdType cdfSize)
  {
    this->Offsets = offsets;
    this->MPISize = mpiSize;
    this->CDFSize = cdfSize;
    this->ProcessStepCount = new vtkIdType[mpiSize];
  }

  void Reset(vtkIdType targetIdx)
  {
    this->TargetIdx = targetIdx;
    this->CurrentIndex = 0;
    this->Step = 1;
    this->KeepMoving = true;

    // Init counts
    for (int i = 0; i < this->MPISize; i++)
    {
      this->ProcessStepCount[i] = 0;
    }

    // Figure out the current index based on offsets
    this->UpdateCurrentIndex();

    // Compute best Step size
    this->Step = (this->TargetIdx - this->CurrentIndex) / this->CDFSize;
    if (this->Step < 2)
    {
      this->Step = 1;
      this->ErrorStep = 0;
    }
    else
    {
      this->ErrorStep = this->MPISize * (this->Step - 1);
    }
  }

  bool CanMoveForward()
  {
    return this->KeepMoving;
  }

  bool Move(vtkIdType processId)
  {
    if (!this->KeepMoving)
    {
      return this->KeepMoving;
    }
    if (this->Step + this->CurrentIndex > this->TargetIdx - this->ErrorStep) {
      return (this->KeepMoving = false);
    }

    if (++this->ProcessStepCount[processId] == this->CDFSize)
    {
      return (this->KeepMoving = false);
    }

    this->Offsets[processId] += this->Step;
    this->CurrentIndex += this->Step;

    return this->KeepMoving;
  }

  void UpdateCurrentIndex()
  {
    this->CurrentIndex = 0;
    for (vtkIdType idx = 0; idx < this->MPISize; idx++)
    {
      this->CurrentIndex += this->Offsets[idx];
    }
  }

  vtkIdType* Offsets;
  vtkIdType TargetIdx;
  vtkIdType MPISize;
  vtkIdType CDFSize;
  vtkIdType CurrentIndex;
  vtkIdType Step;
  bool KeepMoving;
  vtkIdType* ProcessStepCount;
  vtkIdType ErrorStep;
};

class CDFReducer
{
public:
  CDFReducer(vtkMultiProcessController* controller)
    : Controller(controller)
    , ArraySize(0)
    , LocalValues(nullptr)
    , ReducedCDF(nullptr)
    , ReducedCDFSize(0)
    , TotalCount(0)
    , CDFOffsets(nullptr)
    , CDFStep(1)
    , CDFSize(512)
    , LocalCDF(nullptr)
    , RemoteCDFs(nullptr)
    {};
  ~CDFReducer();

  double* Compute(double* localSortedValues, vtkIdType localArraySize, vtkIdType outputCDFSize);
  vtkIdType GetBufferSize() { return this->CDFSize; };
  void SetBufferSize(vtkIdType exchangeCDFSize) { this->CDFSize = exchangeCDFSize; };
  vtkIdType GetTotalCount() { return this->TotalCount; };

protected:
  double GetValueAtIndex(vtkIdType targetIdx, vtkIdType pid, vtkIdType mpiSize, int depth);

private:
  vtkMultiProcessController* Controller;
  vtkIdType ArraySize;
  double* LocalValues;
  double* ReducedCDF;
  vtkIdType ReducedCDFSize;
  vtkIdType TotalCount;
  //
  vtkIdType* CDFOffsets;
  vtkIdType CDFStep;
  vtkIdType CDFSize;
  double* LocalCDF;
  double* RemoteCDFs;
  //
  int ExecutionCount;
  //
  StepHandler Handler;
};
#endif
