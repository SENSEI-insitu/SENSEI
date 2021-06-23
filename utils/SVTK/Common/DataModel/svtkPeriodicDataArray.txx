/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPeriodicDataArray.txx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

    This software is distributed WITHOUT ANY WARRANTY; without even
    the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
    PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkArrayIteratorTemplate.h"
#include "svtkIdList.h"
#include "svtkVariant.h"

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::PrintSelf(ostream& os, svtkIndent indent)
{
  this->svtkPeriodicDataArray<Scalar>::Superclass::PrintSelf(os, indent);

  os << indent << "TempScalarArray: " << this->TempScalarArray << "\n";
  os << indent << "TempDoubleArray: " << this->TempDoubleArray << "\n";
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::Initialize()
{
  delete[] this->TempScalarArray;
  this->TempScalarArray = nullptr;
  delete[] this->TempDoubleArray;
  this->TempDoubleArray = nullptr;
  this->TempTupleIdx = -1;

  if (this->Data)
  {
    this->Data->Delete();
    this->Data = nullptr;
  }

  this->MaxId = -1;
  this->Size = 0;
  this->InvalidRange = true;
  this->Normalize = false;
  this->Modified();
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::InitializeArray(svtkAOSDataArrayTemplate<Scalar>* data)
{
  this->Initialize();
  if (!data)
  {
    svtkErrorMacro(<< "No original data provided.");
    return;
  }

  this->NumberOfComponents = data->GetNumberOfComponents();
  this->Size = data->GetSize();
  this->MaxId = data->GetMaxId();
  this->Data = data;
  this->Data->Register(nullptr);
  this->TempScalarArray = new Scalar[this->NumberOfComponents];
  this->TempDoubleArray = new double[this->NumberOfComponents];
  this->SetName(data->GetName());
  this->InvalidRange = true;
  this->Modified();
}

//------------------------------------------------------------------------------
template <class Scalar>
bool svtkPeriodicDataArray<Scalar>::ComputeScalarRange(double* range)
{
  if (this->NumberOfComponents == 3)
  {
    if (this->InvalidRange)
    {
      this->ComputePeriodicRange();
    }
    for (int i = 0; i < 3; i++)
    {
      range[i * 2] = this->PeriodicRange[i * 2 + 0];
      range[i * 2 + 1] = this->PeriodicRange[i * 2 + 1];
    }
  }
  else
  {
    // Not implemented for tensor
    for (int i = 0; i < this->NumberOfComponents; i++)
    {
      range[i * 2] = 0;
      range[i * 2 + 1] = 1;
    }
  }
  return true;
}

//------------------------------------------------------------------------------
template <class Scalar>
bool svtkPeriodicDataArray<Scalar>::ComputeVectorRange(double range[2])
{
  if (this->NumberOfComponents == 3 && this->Data)
  {
    this->Data->GetRange(range, -1);
  }
  else
  {
    // Not implemented for tensor
    range[0] = 0;
    range[1] = 1;
  }
  return true;
}
//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::ComputePeriodicRange()
{
  if (this->Data)
  {
    this->Data->GetRange(this->PeriodicRange, 0);
    this->Data->GetRange(this->PeriodicRange + 2, 1);
    this->Data->GetRange(this->PeriodicRange + 4, 2);
    Scalar boxPoints[8][3];

    boxPoints[0][0] = this->PeriodicRange[0];
    boxPoints[0][1] = this->PeriodicRange[2];
    boxPoints[0][2] = this->PeriodicRange[4];

    boxPoints[1][0] = this->PeriodicRange[0];
    boxPoints[1][1] = this->PeriodicRange[3];
    boxPoints[1][2] = this->PeriodicRange[4];

    boxPoints[2][0] = this->PeriodicRange[1];
    boxPoints[2][1] = this->PeriodicRange[3];
    boxPoints[2][2] = this->PeriodicRange[4];

    boxPoints[3][0] = this->PeriodicRange[1];
    boxPoints[3][1] = this->PeriodicRange[2];
    boxPoints[3][2] = this->PeriodicRange[4];

    boxPoints[4][0] = this->PeriodicRange[0];
    boxPoints[4][1] = this->PeriodicRange[2];
    boxPoints[4][2] = this->PeriodicRange[5];

    boxPoints[5][0] = this->PeriodicRange[0];
    boxPoints[5][1] = this->PeriodicRange[3];
    boxPoints[5][2] = this->PeriodicRange[5];

    boxPoints[6][0] = this->PeriodicRange[1];
    boxPoints[6][1] = this->PeriodicRange[3];
    boxPoints[6][2] = this->PeriodicRange[5];

    boxPoints[7][0] = this->PeriodicRange[1];
    boxPoints[7][1] = this->PeriodicRange[2];
    boxPoints[7][2] = this->PeriodicRange[5];

    for (int i = 0; i < 8; i++)
    {
      this->Transform(boxPoints[i]);
    }

    this->PeriodicRange[0] = this->PeriodicRange[2] = this->PeriodicRange[4] = SVTK_DOUBLE_MAX;
    this->PeriodicRange[1] = this->PeriodicRange[3] = this->PeriodicRange[5] = -SVTK_DOUBLE_MAX;

    for (int i = 0; i < 8; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        if (boxPoints[i][j] < this->PeriodicRange[2 * j])
        {
          this->PeriodicRange[2 * j] = boxPoints[i][j];
        }
        if (boxPoints[i][j] > this->PeriodicRange[2 * j + 1])
        {
          this->PeriodicRange[2 * j + 1] = boxPoints[i][j];
        }
      }
    }
    this->InvalidRange = false;
  }
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::GetTuples(svtkIdList* ptIds, svtkAbstractArray* output)
{
  svtkDataArray* da = svtkDataArray::FastDownCast(output);
  if (!da)
  {
    svtkWarningMacro(<< "Input is not a svtkDataArray");
    return;
  }

  if (da->GetNumberOfComponents() != this->GetNumberOfComponents())
  {
    svtkWarningMacro(<< "Incorrect number of components in input array.");
    return;
  }

  const svtkIdType numPoints = ptIds->GetNumberOfIds();
  double* tempData = new double[this->NumberOfComponents];
  for (svtkIdType i = 0; i < numPoints; ++i)
  {
    this->GetTuple(ptIds->GetId(i), tempData);
    da->SetTuple(i, tempData);
  }
  delete[] tempData;
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::GetTuples(svtkIdType p1, svtkIdType p2, svtkAbstractArray* output)
{
  svtkDataArray* da = svtkDataArray::FastDownCast(output);
  if (!da)
  {
    svtkErrorMacro(<< "Input is not a svtkDataArray");
    return;
  }

  if (da->GetNumberOfComponents() != this->GetNumberOfComponents())
  {
    svtkErrorMacro(<< "Incorrect number of components in input array.");
    return;
  }

  double* tempData = new double[this->NumberOfComponents];
  for (svtkIdType daTupleId = 0; p1 <= p2; ++p1)
  {
    this->GetTuple(p1, tempData);
    da->SetTuple(daTupleId++, tempData);
  }
  delete[] tempData;
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::Squeeze()
{
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkArrayIterator* svtkPeriodicDataArray<Scalar>::NewIterator()
{
  svtkArrayIteratorTemplate<Scalar>* iter = svtkArrayIteratorTemplate<Scalar>::New();
  iter->Initialize(this);
  return iter;
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkIdType svtkPeriodicDataArray<Scalar>::LookupValue(svtkVariant)
{
  svtkErrorMacro("Lookup not implemented in this container.");
  return -1;
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::LookupValue(svtkVariant, svtkIdList*)
{
  svtkErrorMacro("Lookup not implemented in this container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkVariant svtkPeriodicDataArray<Scalar>::GetVariantValue(svtkIdType idx)
{
  return svtkVariant(this->GetValueReference(idx));
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::ClearLookup()
{
  svtkErrorMacro("Lookup not implemented in this container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
double* svtkPeriodicDataArray<Scalar>::GetTuple(svtkIdType i)
{
  if (this->TempTupleIdx != i)
  {
    this->GetTypedTuple(i, this->TempScalarArray);
    this->TempTupleIdx = i;
  }
  for (int j = 0; j < this->NumberOfComponents; j++)
  {
    this->TempDoubleArray[j] = static_cast<double>(this->TempScalarArray[j]);
  }
  return this->TempDoubleArray;
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::GetTuple(svtkIdType i, double* tuple)
{
  if (this->TempTupleIdx != i)
  {
    this->GetTypedTuple(i, this->TempScalarArray);
    this->TempTupleIdx = i;
  }
  for (int j = 0; j < this->NumberOfComponents; j++)
  {
    tuple[j] = static_cast<double>(this->TempScalarArray[j]);
  }
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkIdType svtkPeriodicDataArray<Scalar>::LookupTypedValue(Scalar)
{
  svtkErrorMacro("Lookup not implemented in this container.");
  return 0;
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::LookupTypedValue(Scalar, svtkIdList*)
{
  svtkErrorMacro("Lookup not implemented in this container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
typename svtkPeriodicDataArray<Scalar>::ValueType svtkPeriodicDataArray<Scalar>::GetValue(
  svtkIdType idx) const
{
  return const_cast<svtkPeriodicDataArray<Scalar>*>(this)->GetValueReference(idx);
}

//------------------------------------------------------------------------------
template <class Scalar>
typename svtkPeriodicDataArray<Scalar>::ValueType& svtkPeriodicDataArray<Scalar>::GetValueReference(
  svtkIdType idx)
{
  svtkIdType tupleIdx = idx / this->NumberOfComponents;
  if (tupleIdx != this->TempTupleIdx)
  {
    this->GetTypedTuple(tupleIdx, this->TempScalarArray);
    this->TempTupleIdx = tupleIdx;
  }
  return this->TempScalarArray[idx % this->NumberOfComponents];
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::GetTypedTuple(svtkIdType tupleId, Scalar* tuple) const
{
  this->Data->GetTypedTuple(tupleId, tuple);
  this->Transform(tuple);
}

//------------------------------------------------------------------------------
template <class Scalar>
typename svtkPeriodicDataArray<Scalar>::ValueType svtkPeriodicDataArray<Scalar>::GetTypedComponent(
  svtkIdType tupleId, int compId) const
{
  if (tupleId != this->TempTupleIdx)
  {
    this->Data->GetTypedTuple(tupleId, this->TempScalarArray);
    this->Transform(const_cast<Scalar*>(this->TempScalarArray));
    *const_cast<svtkIdType*>(&this->TempTupleIdx) = tupleId;
  }

  return this->TempScalarArray[compId];
}

//------------------------------------------------------------------------------
template <class Scalar>
unsigned long int svtkPeriodicDataArray<Scalar>::GetActualMemorySize() const
{
  return static_cast<unsigned long int>(
    (this->NumberOfComponents * (sizeof(Scalar) + sizeof(double)) + sizeof(*this)) / 1024);
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkTypeBool svtkPeriodicDataArray<Scalar>::Allocate(svtkIdType, svtkIdType)
{
  svtkErrorMacro("Read only container.");
  return 0;
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkTypeBool svtkPeriodicDataArray<Scalar>::Resize(svtkIdType)
{
  svtkErrorMacro("Read only container.");
  return 0;
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::SetNumberOfTuples(svtkIdType)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::SetTuple(svtkIdType, svtkIdType, svtkAbstractArray*)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::SetTuple(svtkIdType, const float*)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::SetTuple(svtkIdType, const double*)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::InsertTuple(svtkIdType, svtkIdType, svtkAbstractArray*)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::InsertTuple(svtkIdType, const float*)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::InsertTuple(svtkIdType, const double*)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::InsertTuples(svtkIdList*, svtkIdList*, svtkAbstractArray*)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::InsertTuples(svtkIdType, svtkIdType, svtkIdType, svtkAbstractArray*)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkIdType svtkPeriodicDataArray<Scalar>::InsertNextTuple(svtkIdType, svtkAbstractArray*)
{
  svtkErrorMacro("Read only container.");
  return -1;
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkIdType svtkPeriodicDataArray<Scalar>::InsertNextTuple(const float*)
{
  svtkErrorMacro("Read only container.");
  return -1;
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkIdType svtkPeriodicDataArray<Scalar>::InsertNextTuple(const double*)
{
  svtkErrorMacro("Read only container.");
  return -1;
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::DeepCopy(svtkAbstractArray*)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::DeepCopy(svtkDataArray*)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::InterpolateTuple(
  svtkIdType, svtkIdList*, svtkAbstractArray*, double*)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::InterpolateTuple(
  svtkIdType, svtkIdType, svtkAbstractArray*, svtkIdType, svtkAbstractArray*, double)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::SetVariantValue(svtkIdType, svtkVariant)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::InsertVariantValue(svtkIdType, svtkVariant)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::RemoveTuple(svtkIdType)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::RemoveFirstTuple()
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::RemoveLastTuple()
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::SetTypedTuple(svtkIdType, const Scalar*)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::SetTypedComponent(svtkIdType, int, Scalar)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::InsertTypedTuple(svtkIdType, const Scalar*)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkIdType svtkPeriodicDataArray<Scalar>::InsertNextTypedTuple(const Scalar*)
{
  svtkErrorMacro("Read only container.");
  return -1;
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::SetValue(svtkIdType, Scalar)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkIdType svtkPeriodicDataArray<Scalar>::InsertNextValue(Scalar)
{
  svtkErrorMacro("Read only container.");
  return -1;
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::InsertValue(svtkIdType, Scalar)
{
  svtkErrorMacro("Read only container.");
}

//------------------------------------------------------------------------------
template <class Scalar>
bool svtkPeriodicDataArray<Scalar>::AllocateTuples(svtkIdType)
{
  svtkErrorMacro("Read only container.");
  return false;
}

//------------------------------------------------------------------------------
template <class Scalar>
bool svtkPeriodicDataArray<Scalar>::ReallocateTuples(svtkIdType)
{
  svtkErrorMacro("Read only container.");
  return false;
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkPeriodicDataArray<Scalar>::InvalidateRange()
{
  this->InvalidRange = true;
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkPeriodicDataArray<Scalar>::svtkPeriodicDataArray()
{
  this->NumberOfComponents = 0;
  this->TempScalarArray = nullptr;
  this->TempDoubleArray = nullptr;
  this->TempTupleIdx = -1;
  this->Data = nullptr;
  this->MaxId = -1;
  this->Size = 0;

  this->InvalidRange = true;
  this->Normalize = false;
  this->PeriodicRange[0] = this->PeriodicRange[2] = this->PeriodicRange[4] = SVTK_DOUBLE_MAX;
  this->PeriodicRange[1] = this->PeriodicRange[3] = this->PeriodicRange[5] = -SVTK_DOUBLE_MAX;
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkPeriodicDataArray<Scalar>::~svtkPeriodicDataArray()
{
  this->Initialize();
}
