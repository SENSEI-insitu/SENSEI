/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArray.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkDataArray.h"
#include "svtkAOSDataArrayTemplate.h" // For fast paths
#include "svtkArrayDispatch.h"
#include "svtkBitArray.h"
#include "svtkCharArray.h"
#include "svtkDataArrayPrivate.txx"
#include "svtkDataArrayRange.h"
#include "svtkDoubleArray.h"
#include "svtkFloatArray.h"
#include "svtkGenericDataArray.h"
#include "svtkIdList.h"
#include "svtkIdTypeArray.h"
#include "svtkInformation.h"
#include "svtkInformationDoubleVectorKey.h"
#include "svtkInformationInformationVectorKey.h"
#include "svtkInformationStringKey.h"
#include "svtkInformationVector.h"
#include "svtkIntArray.h"
#include "svtkLongArray.h"
#include "svtkLookupTable.h"
#include "svtkMath.h"
#include "svtkSOADataArrayTemplate.h" // For fast paths
#ifdef SVTK_USE_SCALED_SOA_ARRAYS
#include "svtkScaledSOADataArrayTemplate.h" // For fast paths
#endif
#include "svtkShortArray.h"
#include "svtkSignedCharArray.h"
#include "svtkTypeTraits.h"
#include "svtkUnsignedCharArray.h"
#include "svtkUnsignedIntArray.h"
#include "svtkUnsignedLongArray.h"
#include "svtkUnsignedShortArray.h"

#include <algorithm> // for min(), max()

namespace
{

//--------Copy tuples from src to dest------------------------------------------
struct DeepCopyWorker
{
  // AoS --> AoS same-type specialization:
  template <typename ValueType>
  void operator()(
    svtkAOSDataArrayTemplate<ValueType>* src, svtkAOSDataArrayTemplate<ValueType>* dst) const
  {
    std::copy(src->Begin(), src->End(), dst->Begin());
  }

#if defined(__clang__) && defined(__has_warning)
#if __has_warning("-Wunused-template")
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-template"
#endif
#endif

  // SoA --> SoA same-type specialization:
  template <typename ValueType>
  void operator()(
    svtkSOADataArrayTemplate<ValueType>* src, svtkSOADataArrayTemplate<ValueType>* dst) const
  {
    svtkIdType numTuples = src->GetNumberOfTuples();
    for (int comp = 0; comp < src->GetNumberOfComponents(); ++comp)
    {
      ValueType* srcBegin = src->GetComponentArrayPointer(comp);
      ValueType* srcEnd = srcBegin + numTuples;
      ValueType* dstBegin = dst->GetComponentArrayPointer(comp);

      std::copy(srcBegin, srcEnd, dstBegin);
    }
  }

#ifdef SVTK_USE_SCALED_SOA_ARRAYS
  // ScaleSoA --> ScaleSoA same-type specialization:
  template <typename ValueType>
  void operator()(
    svtkScaledSOADataArrayTemplate<ValueType>* src, svtkScaledSOADataArrayTemplate<ValueType>* dst)
  {
    svtkIdType numTuples = src->GetNumberOfTuples();
    for (int comp = 0; comp < src->GetNumberOfComponents(); ++comp)
    {
      ValueType* srcBegin = src->GetComponentArrayPointer(comp);
      ValueType* srcEnd = srcBegin + numTuples;
      ValueType* dstBegin = dst->GetComponentArrayPointer(comp);

      std::copy(srcBegin, srcEnd, dstBegin);
    }
    dst->SetScale(src->GetScale());
  }
#endif
// Undo warning suppression.
#if defined(__clang__) && defined(__has_warning)
#if __has_warning("-Wunused-template")
#pragma clang diagnostic pop
#endif
#endif

  // Generic implementation:
  template <typename SrcArrayT, typename DstArrayT>
  void DoGenericCopy(SrcArrayT* src, DstArrayT* dst) const
  {
    const auto srcRange = svtk::DataArrayValueRange(src);
    auto dstRange = svtk::DataArrayValueRange(dst);

    using DstT = typename decltype(dstRange)::ValueType;
    auto destIter = dstRange.begin();
    // use for loop instead of copy to avoid -Wconversion warnings
    for (auto v = srcRange.cbegin(); v != srcRange.cend(); ++v, ++destIter)
    {
      *destIter = static_cast<DstT>(*v);
    }
  }

  // These overloads are split so that the above specializations will be
  // used properly.
  template <typename Array1DerivedT, typename Array1ValueT, typename Array2DerivedT,
    typename Array2ValueT>
  void operator()(svtkGenericDataArray<Array1DerivedT, Array1ValueT>* src,
    svtkGenericDataArray<Array2DerivedT, Array2ValueT>* dst) const
  {
    this->DoGenericCopy(src, dst);
  }

  void operator()(svtkDataArray* src, svtkDataArray* dst) const { this->DoGenericCopy(src, dst); }
};

//------------InterpolateTuple workers------------------------------------------
struct InterpolateMultiTupleWorker
{
  svtkIdType DestTuple;
  svtkIdType* TupleIds;
  svtkIdType NumTuples;
  double* Weights;

  InterpolateMultiTupleWorker(
    svtkIdType destTuple, svtkIdType* tupleIds, svtkIdType numTuples, double* weights)
    : DestTuple(destTuple)
    , TupleIds(tupleIds)
    , NumTuples(numTuples)
    , Weights(weights)
  {
  }

  template <typename Array1T, typename Array2T>
  void operator()(Array1T* src, Array2T* dst) const
  {
    // Use svtkDataArrayAccessor here instead of a range, since we need to use
    // Insert for legacy compat
    svtkDataArrayAccessor<Array1T> s(src);
    svtkDataArrayAccessor<Array2T> d(dst);

    typedef typename svtkDataArrayAccessor<Array2T>::APIType DestType;

    int numComp = src->GetNumberOfComponents();

    for (int c = 0; c < numComp; ++c)
    {
      double val = 0.;
      for (svtkIdType tupleId = 0; tupleId < this->NumTuples; ++tupleId)
      {
        svtkIdType t = this->TupleIds[tupleId];
        double weight = this->Weights[tupleId];
        val += weight * static_cast<double>(s.Get(t, c));
      }
      DestType valT;
      svtkMath::RoundDoubleToIntegralIfNecessary(val, &valT);
      d.Insert(this->DestTuple, c, valT);
    }
  }
};

struct InterpolateTupleWorker
{
  svtkIdType SrcTuple1;
  svtkIdType SrcTuple2;
  svtkIdType DstTuple;
  double Weight;

  InterpolateTupleWorker(
    svtkIdType srcTuple1, svtkIdType srcTuple2, svtkIdType dstTuple, double weight)
    : SrcTuple1(srcTuple1)
    , SrcTuple2(srcTuple2)
    , DstTuple(dstTuple)
    , Weight(weight)
  {
  }

  template <typename Array1T, typename Array2T, typename Array3T>
  void operator()(Array1T* src1, Array2T* src2, Array3T* dst) const
  {
    // Use accessor here instead of ranges since we need to use Insert for
    // legacy compat
    svtkDataArrayAccessor<Array1T> s1(src1);
    svtkDataArrayAccessor<Array2T> s2(src2);
    svtkDataArrayAccessor<Array3T> d(dst);

    typedef typename svtkDataArrayAccessor<Array3T>::APIType DestType;

    const int numComps = dst->GetNumberOfComponents();
    const double oneMinusT = 1. - this->Weight;
    double val;
    DestType valT;

    for (int c = 0; c < numComps; ++c)
    {
      val = s1.Get(this->SrcTuple1, c) * oneMinusT + s2.Get(this->SrcTuple2, c) * this->Weight;
      svtkMath::RoundDoubleToIntegralIfNecessary(val, &valT);
      d.Insert(this->DstTuple, c, valT);
    }
  }
};

//-----------------GetTuples (id list)------------------------------------------
struct GetTuplesFromListWorker
{
  svtkIdList* Ids;

  GetTuplesFromListWorker(svtkIdList* ids)
    : Ids(ids)
  {
  }

  template <typename Array1T, typename Array2T>
  void operator()(Array1T* src, Array2T* dst) const
  {
    const auto srcTuples = svtk::DataArrayTupleRange(src);
    auto dstTuples = svtk::DataArrayTupleRange(dst);

    svtkIdType* srcTupleId = this->Ids->GetPointer(0);
    svtkIdType* srcTupleIdEnd = this->Ids->GetPointer(Ids->GetNumberOfIds());

    auto dstTupleIter = dstTuples.begin();
    while (srcTupleId != srcTupleIdEnd)
    {
      *dstTupleIter++ = srcTuples[*srcTupleId++];
    }
  }
};

//-----------------GetTuples (tuple range)--------------------------------------
struct GetTuplesRangeWorker
{
  svtkIdType Start;
  svtkIdType End; // Note that End is inclusive.

  GetTuplesRangeWorker(svtkIdType start, svtkIdType end)
    : Start(start)
    , End(end)
  {
  }

  template <typename Array1T, typename Array2T>
  void operator()(Array1T* src, Array2T* dst) const
  {
    const auto srcTuples = svtk::DataArrayTupleRange(src);
    auto dstTuples = svtk::DataArrayTupleRange(dst);

    for (svtkIdType srcT = this->Start, dstT = 0; srcT <= this->End; ++srcT, ++dstT)
    {
      dstTuples[dstT] = srcTuples[srcT];
    }
  }
};

//----------------SetTuple (from array)-----------------------------------------
struct SetTupleArrayWorker
{
  svtkIdType SrcTuple;
  svtkIdType DstTuple;

  SetTupleArrayWorker(svtkIdType srcTuple, svtkIdType dstTuple)
    : SrcTuple(srcTuple)
    , DstTuple(dstTuple)
  {
  }

  template <typename SrcArrayT, typename DstArrayT>
  void operator()(SrcArrayT* src, DstArrayT* dst) const
  {
    const auto srcTuples = svtk::DataArrayTupleRange(src);
    auto dstTuples = svtk::DataArrayTupleRange(dst);

    dstTuples[this->DstTuple] = srcTuples[this->SrcTuple];
  }
};

//----------------SetTuples (from array+svtkIdList)------------------------------
struct SetTuplesIdListWorker
{
  svtkIdList* SrcTuples;
  svtkIdList* DstTuples;

  SetTuplesIdListWorker(svtkIdList* srcTuples, svtkIdList* dstTuples)
    : SrcTuples(srcTuples)
    , DstTuples(dstTuples)
  {
  }

  template <typename SrcArrayT, typename DstArrayT>
  void operator()(SrcArrayT* src, DstArrayT* dst) const
  {
    const auto srcTuples = svtk::DataArrayTupleRange(src);
    auto dstTuples = svtk::DataArrayTupleRange(dst);

    svtkIdType numTuples = this->SrcTuples->GetNumberOfIds();
    for (svtkIdType t = 0; t < numTuples; ++t)
    {
      svtkIdType srcT = this->SrcTuples->GetId(t);
      svtkIdType dstT = this->DstTuples->GetId(t);

      dstTuples[dstT] = srcTuples[srcT];
    }
  }
};

//----------------SetTuples (from array+range)----------------------------------
struct SetTuplesRangeWorker
{
  svtkIdType SrcStartTuple;
  svtkIdType DstStartTuple;
  svtkIdType NumTuples;

  SetTuplesRangeWorker(svtkIdType srcStartTuple, svtkIdType dstStartTuple, svtkIdType numTuples)
    : SrcStartTuple(srcStartTuple)
    , DstStartTuple(dstStartTuple)
    , NumTuples(numTuples)
  {
  }

  // Generic implementation. We perform the obvious optimizations for AOS/SOA
  // in the derived class implementations.
  template <typename SrcArrayT, typename DstArrayT>
  void operator()(SrcArrayT* src, DstArrayT* dst) const
  {
    const auto srcTuples = svtk::DataArrayTupleRange(src);
    auto dstTuples = svtk::DataArrayTupleRange(dst);

    svtkIdType srcT = this->SrcStartTuple;
    svtkIdType srcTEnd = srcT + this->NumTuples;
    svtkIdType dstT = this->DstStartTuple;

    while (srcT < srcTEnd)
    {
      dstTuples[dstT++] = srcTuples[srcT++];
    }
  }
};

template <typename InfoType, typename KeyType>
bool hasValidKey(InfoType info, KeyType key, double range[2])
{
  if (info->Has(key))
  {
    info->Get(key, range);
    return true;
  }
  return false;
}

template <typename InfoType, typename KeyType, typename ComponentKeyType>
bool hasValidKey(InfoType info, KeyType key, ComponentKeyType ckey, double range[2], int comp)
{
  if (info->Has(key))
  {
    info->Get(key)->GetInformationObject(comp)->Get(ckey, range);
    return true;
  }
  return false;
}

} // end anon namespace

svtkInformationKeyRestrictedMacro(svtkDataArray, COMPONENT_RANGE, DoubleVector, 2);
svtkInformationKeyRestrictedMacro(svtkDataArray, L2_NORM_RANGE, DoubleVector, 2);
svtkInformationKeyRestrictedMacro(svtkDataArray, L2_NORM_FINITE_RANGE, DoubleVector, 2);
svtkInformationKeyMacro(svtkDataArray, UNITS_LABEL, String);

//----------------------------------------------------------------------------
// Construct object with default tuple dimension (number of components) of 1.
svtkDataArray::svtkDataArray()
{
  this->LookupTable = nullptr;
  this->Range[0] = 0;
  this->Range[1] = 0;
  this->FiniteRange[0] = 0;
  this->FiniteRange[1] = 0;
}

//----------------------------------------------------------------------------
svtkDataArray::~svtkDataArray()
{
  if (this->LookupTable)
  {
    this->LookupTable->Delete();
  }
  this->SetName(nullptr);
}

//----------------------------------------------------------------------------
void svtkDataArray::DeepCopy(svtkAbstractArray* aa)
{
  if (aa == nullptr)
  {
    return;
  }

  svtkDataArray* da = svtkDataArray::FastDownCast(aa);
  if (da == nullptr)
  {
    svtkErrorMacro(<< "Input array is not a svtkDataArray (" << aa->GetClassName() << ")");
    return;
  }

  this->DeepCopy(da);
}

//----------------------------------------------------------------------------
// Normally subclasses will do this when the input and output type of the
// DeepCopy are the same. When they are not the same, then we use the
// templated code below.
void svtkDataArray::DeepCopy(svtkDataArray* da)
{
  // Match the behavior of the old AttributeData
  if (da == nullptr)
  {
    return;
  }

  if (this != da)
  {
    this->Superclass::DeepCopy(da); // copy Information object

    svtkIdType numTuples = da->GetNumberOfTuples();
    int numComps = da->NumberOfComponents;

    this->SetNumberOfComponents(numComps);
    this->SetNumberOfTuples(numTuples);

    if (numTuples != 0)
    {
      DeepCopyWorker worker;
      if (!svtkArrayDispatch::Dispatch2::Execute(da, this, worker))
      {
        // If dispatch fails, use fallback:
        worker(da, this);
      }
    }

    this->SetLookupTable(nullptr);
    if (da->LookupTable)
    {
      this->LookupTable = da->LookupTable->NewInstance();
      this->LookupTable->DeepCopy(da->LookupTable);
    }
  }

  this->Squeeze();
}

//------------------------------------------------------------------------------
void svtkDataArray::ShallowCopy(svtkDataArray* other)
{
  // Deep copy by default. Subclasses may override this behavior.
  this->DeepCopy(other);
}

//------------------------------------------------------------------------------
void svtkDataArray::SetTuple(svtkIdType dstTupleIdx, svtkIdType srcTupleIdx, svtkAbstractArray* source)
{
  svtkDataArray* srcDA = svtkDataArray::FastDownCast(source);
  if (!srcDA)
  {
    svtkErrorMacro(
      "Source array must be a svtkDataArray subclass (got " << source->GetClassName() << ").");
    return;
  }

  if (!svtkDataTypesCompare(source->GetDataType(), this->GetDataType()))
  {
    svtkErrorMacro("Type mismatch: Source: " << source->GetDataTypeAsString()
                                            << " Dest: " << this->GetDataTypeAsString());
    return;
  }

  if (source->GetNumberOfComponents() != this->GetNumberOfComponents())
  {
    svtkErrorMacro("Number of components do not match: Source: "
      << source->GetNumberOfComponents() << " Dest: " << this->GetNumberOfComponents());
    return;
  }

  SetTupleArrayWorker worker(srcTupleIdx, dstTupleIdx);
  if (!svtkArrayDispatch::Dispatch2SameValueType::Execute(srcDA, this, worker))
  {
    worker(srcDA, this);
  }
}

//----------------------------------------------------------------------------
void svtkDataArray::SetTuple(svtkIdType i, const float* source)
{
  for (int c = 0; c < this->NumberOfComponents; ++c)
  {
    this->SetComponent(i, c, static_cast<double>(source[c]));
  }
}

//----------------------------------------------------------------------------
void svtkDataArray::SetTuple(svtkIdType i, const double* source)
{
  for (int c = 0; c < this->NumberOfComponents; ++c)
  {
    this->SetComponent(i, c, source[c]);
  }
}

//----------------------------------------------------------------------------
void svtkDataArray::InsertTuple(
  svtkIdType dstTupleIdx, svtkIdType srcTupleIdx, svtkAbstractArray* source)
{
  svtkIdType newSize = (dstTupleIdx + 1) * this->NumberOfComponents;
  if (this->Size < newSize)
  {
    if (!this->Resize(dstTupleIdx + 1))
    {
      svtkErrorMacro("Resize failed.");
      return;
    }
  }

  this->MaxId = std::max(this->MaxId, newSize - 1);

  this->SetTuple(dstTupleIdx, srcTupleIdx, source);
}

//----------------------------------------------------------------------------
svtkIdType svtkDataArray::InsertNextTuple(svtkIdType srcTupleIdx, svtkAbstractArray* source)
{
  svtkIdType tupleIdx = this->GetNumberOfTuples();
  this->InsertTuple(tupleIdx, srcTupleIdx, source);
  return tupleIdx;
}

//----------------------------------------------------------------------------
void svtkDataArray::InsertTuples(svtkIdList* dstIds, svtkIdList* srcIds, svtkAbstractArray* src)
{
  if (dstIds->GetNumberOfIds() == 0)
  {
    return;
  }
  if (dstIds->GetNumberOfIds() != srcIds->GetNumberOfIds())
  {
    svtkErrorMacro("Mismatched number of tuples ids. Source: "
      << srcIds->GetNumberOfIds() << " Dest: " << dstIds->GetNumberOfIds());
    return;
  }
  if (!svtkDataTypesCompare(src->GetDataType(), this->GetDataType()))
  {
    svtkErrorMacro("Data type mismatch: Source: " << src->GetDataTypeAsString()
                                                 << " Dest: " << this->GetDataTypeAsString());
    return;
  }
  if (src->GetNumberOfComponents() != this->GetNumberOfComponents())
  {
    svtkErrorMacro("Number of components do not match: Source: "
      << src->GetNumberOfComponents() << " Dest: " << this->GetNumberOfComponents());
    return;
  }
  svtkDataArray* srcDA = svtkDataArray::FastDownCast(src);
  if (!srcDA)
  {
    svtkErrorMacro("Source array must be a subclass of svtkDataArray. Got: " << src->GetClassName());
    return;
  }

  svtkIdType maxSrcTupleId = srcIds->GetId(0);
  svtkIdType maxDstTupleId = dstIds->GetId(0);
  for (int i = 1; i < dstIds->GetNumberOfIds(); ++i)
  {
    maxSrcTupleId = std::max(maxSrcTupleId, srcIds->GetId(i));
    maxDstTupleId = std::max(maxDstTupleId, dstIds->GetId(i));
  }

  if (maxSrcTupleId >= src->GetNumberOfTuples())
  {
    svtkErrorMacro("Source array too small, requested tuple at index "
      << maxSrcTupleId << ", but there are only " << src->GetNumberOfTuples()
      << " tuples in the array.");
    return;
  }

  svtkIdType newSize = (maxDstTupleId + 1) * this->NumberOfComponents;
  if (this->Size < newSize)
  {
    if (!this->Resize(maxDstTupleId + 1))
    {
      svtkErrorMacro("Resize failed.");
      return;
    }
  }

  this->MaxId = std::max(this->MaxId, newSize - 1);

  SetTuplesIdListWorker worker(srcIds, dstIds);
  if (!svtkArrayDispatch::Dispatch2SameValueType::Execute(srcDA, this, worker))
  {
    worker(srcDA, this);
  }
}

//----------------------------------------------------------------------------
void svtkDataArray::InsertTuples(
  svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkAbstractArray* src)
{
  if (n == 0)
  {
    return;
  }
  if (!svtkDataTypesCompare(src->GetDataType(), this->GetDataType()))
  {
    svtkErrorMacro("Data type mismatch: Source: " << src->GetDataTypeAsString()
                                                 << " Dest: " << this->GetDataTypeAsString());
    return;
  }
  if (src->GetNumberOfComponents() != this->GetNumberOfComponents())
  {
    svtkErrorMacro("Number of components do not match: Source: "
      << src->GetNumberOfComponents() << " Dest: " << this->GetNumberOfComponents());
    return;
  }
  svtkDataArray* srcDA = svtkDataArray::FastDownCast(src);
  if (!srcDA)
  {
    svtkErrorMacro("Source array must be a subclass of svtkDataArray. Got: " << src->GetClassName());
    return;
  }

  svtkIdType maxSrcTupleId = srcStart + n - 1;
  svtkIdType maxDstTupleId = dstStart + n - 1;

  if (maxSrcTupleId >= src->GetNumberOfTuples())
  {
    svtkErrorMacro("Source array too small, requested tuple at index "
      << maxSrcTupleId << ", but there are only " << src->GetNumberOfTuples()
      << " tuples in the array.");
    return;
  }

  svtkIdType newSize = (maxDstTupleId + 1) * this->NumberOfComponents;
  if (this->Size < newSize)
  {
    if (!this->Resize(maxDstTupleId + 1))
    {
      svtkErrorMacro("Resize failed.");
      return;
    }
  }

  this->MaxId = std::max(this->MaxId, newSize - 1);

  SetTuplesRangeWorker worker(srcStart, dstStart, n);
  if (!svtkArrayDispatch::Dispatch2SameValueType::Execute(srcDA, this, worker))
  {
    worker(srcDA, this);
  }
}

//----------------------------------------------------------------------------
// These can be overridden for more efficiency
double svtkDataArray::GetComponent(svtkIdType tupleIdx, int compIdx)
{
  double *tuple = new double[this->NumberOfComponents], c;

  this->GetTuple(tupleIdx, tuple);
  c = tuple[compIdx];
  delete[] tuple;

  return c;
}

//----------------------------------------------------------------------------
void svtkDataArray::SetComponent(svtkIdType tupleIdx, int compIdx, double value)
{
  double* tuple = new double[this->NumberOfComponents];

  if (tupleIdx < this->GetNumberOfTuples())
  {
    this->GetTuple(tupleIdx, tuple);
  }
  else
  {
    for (int k = 0; k < this->NumberOfComponents; k++)
    {
      tuple[k] = 0.0;
    }
  }

  tuple[compIdx] = value;
  this->SetTuple(tupleIdx, tuple);

  delete[] tuple;
}

//----------------------------------------------------------------------------
void svtkDataArray::InsertComponent(svtkIdType tupleIdx, int compIdx, double value)
{
  double* tuple = new double[this->NumberOfComponents];

  if (tupleIdx < this->GetNumberOfTuples())
  {
    this->GetTuple(tupleIdx, tuple);
  }
  else
  {
    for (int k = 0; k < this->NumberOfComponents; k++)
    {
      tuple[k] = 0.0;
    }
  }

  tuple[compIdx] = value;
  this->InsertTuple(tupleIdx, tuple);

  delete[] tuple;
}

//----------------------------------------------------------------------------
void svtkDataArray::GetData(
  svtkIdType tupleMin, svtkIdType tupleMax, int compMin, int compMax, svtkDoubleArray* data)
{
  int i;
  svtkIdType j;
  int numComp = this->GetNumberOfComponents();
  double* tuple = new double[numComp];
  double* ptr = data->WritePointer(0, (tupleMax - tupleMin + 1) * (compMax - compMin + 1));

  for (j = tupleMin; j <= tupleMax; j++)
  {
    this->GetTuple(j, tuple);
    for (i = compMin; i <= compMax; i++)
    {
      *ptr++ = tuple[i];
    }
  }
  delete[] tuple;
}

//----------------------------------------------------------------------------
// Interpolate array value from other array value given the
// indices and associated interpolation weights.
// This method assumes that the two arrays are of the same time.
void svtkDataArray::InterpolateTuple(
  svtkIdType dstTupleIdx, svtkIdList* tupleIds, svtkAbstractArray* source, double* weights)
{
  if (!svtkDataTypesCompare(this->GetDataType(), source->GetDataType()))
  {
    svtkErrorMacro("Cannot interpolate arrays of different type.");
    return;
  }

  svtkDataArray* da = svtkDataArray::FastDownCast(source);
  if (!da)
  {
    svtkErrorMacro(<< "Source array is not a svtkDataArray.");
    return;
  }

  int numComps = this->GetNumberOfComponents();
  if (da->GetNumberOfComponents() != numComps)
  {
    svtkErrorMacro("Number of components do not match: Source: "
      << source->GetNumberOfComponents() << " Dest: " << this->GetNumberOfComponents());
    return;
  }

  svtkIdType numIds = tupleIds->GetNumberOfIds();
  svtkIdType* ids = tupleIds->GetPointer(0);

  bool fallback = da->GetDataType() == SVTK_BIT || this->GetDataType() == SVTK_BIT;

  if (!fallback)
  {
    InterpolateMultiTupleWorker worker(dstTupleIdx, ids, numIds, weights);
    // Use fallback if dispatch fails.
    fallback = !svtkArrayDispatch::Dispatch2SameValueType::Execute(da, this, worker);
  }

  // Fallback to a separate implementation that checks svtkDataArray::GetDataType
  // rather than relying on API types, since we'll need to round differently
  // depending on type, and the API type for svtkDataArray is always double.
  if (fallback)
  {
    bool doRound = !(this->GetDataType() == SVTK_FLOAT || this->GetDataType() == SVTK_DOUBLE);
    double typeMin = this->GetDataTypeMin();
    double typeMax = this->GetDataTypeMax();

    for (int c = 0; c < numComps; ++c)
    {
      double val = 0.;
      for (svtkIdType j = 0; j < numIds; ++j)
      {
        val += weights[j] * da->GetComponent(ids[j], c);
      }

      // Clamp to data type range:
      val = std::max(val, typeMin);
      val = std::min(val, typeMax);

      // Round for floating point types:
      if (doRound)
      {
        val = std::floor((val >= 0.) ? (val + 0.5) : (val - 0.5));
      }

      this->InsertComponent(dstTupleIdx, c, val);
    }
  }
}

//----------------------------------------------------------------------------
// Interpolate value from the two values, p1 and p2, and an
// interpolation factor, t. The interpolation factor ranges from (0,1),
// with t=0 located at p1. This method assumes that the three arrays are of
// the same type. p1 is value at index id1 in fromArray1, while, p2 is
// value at index id2 in fromArray2.
void svtkDataArray::InterpolateTuple(svtkIdType dstTuple, svtkIdType srcTuple1,
  svtkAbstractArray* source1, svtkIdType srcTuple2, svtkAbstractArray* source2, double t)
{
  int type = this->GetDataType();

  if (!svtkDataTypesCompare(type, source1->GetDataType()) ||
    !svtkDataTypesCompare(type, source2->GetDataType()))
  {
    svtkErrorMacro("All arrays to InterpolateValue must be of same type.");
    return;
  }

  if (srcTuple1 >= source1->GetNumberOfTuples())
  {
    svtkErrorMacro("Tuple 1 out of range for provided array. "
                  "Requested tuple: "
      << srcTuple1
      << " "
         "Tuples: "
      << source1->GetNumberOfTuples());
    return;
  }

  if (srcTuple2 >= source2->GetNumberOfTuples())
  {
    svtkErrorMacro("Tuple 2 out of range for provided array. "
                  "Requested tuple: "
      << srcTuple2
      << " "
         "Tuples: "
      << source2->GetNumberOfTuples());
    return;
  }

  svtkDataArray* src1DA = svtkDataArray::FastDownCast(source1);
  svtkDataArray* src2DA = svtkDataArray::FastDownCast(source2);
  if (!src1DA || !src2DA)
  {
    svtkErrorMacro("Both arrays must be svtkDataArray subclasses.");
    return;
  }

  bool fallback = type == SVTK_BIT;

  if (!fallback)
  {
    InterpolateTupleWorker worker(srcTuple1, srcTuple2, dstTuple, t);
    // Use fallback if dispatch fails:
    fallback = !svtkArrayDispatch::Dispatch3SameValueType::Execute(src1DA, src2DA, this, worker);
  }

  // Fallback to a separate implementation that checks svtkDataArray::GetDataType
  // rather than relying on API types, since we'll need to round differently
  // depending on type, and the API type for svtkDataArray is always double.
  if (fallback)
  {
    bool doRound = !(this->GetDataType() == SVTK_FLOAT || this->GetDataType() == SVTK_DOUBLE);
    double typeMin = this->GetDataTypeMin();
    double typeMax = this->GetDataTypeMax();
    int numComp = source1->GetNumberOfComponents();
    double in1;
    double in2;
    double out;
    for (int c = 0; c < numComp; c++)
    {
      in1 = src1DA->GetComponent(srcTuple1, c);
      in2 = src2DA->GetComponent(srcTuple2, c);
      out = in1 + t * (in2 - in1);
      // Clamp to datatype range:
      out = std::max(out, typeMin);
      out = std::min(out, typeMax);
      // Round if needed:
      if (doRound)
      {
        out = std::floor((out >= 0.) ? (out + 0.5) : (out - 0.5));
      }
      this->InsertComponent(dstTuple, c, out);
    }
  }
}

//----------------------------------------------------------------------------
void svtkDataArray::CreateDefaultLookupTable()
{
  if (this->LookupTable)
  {
    this->LookupTable->UnRegister(this);
  }
  this->LookupTable = svtkLookupTable::New();
  // make sure it is built
  // otherwise problems with InsertScalar trying to map through
  // non built lut
  this->LookupTable->Build();
}

//----------------------------------------------------------------------------
void svtkDataArray::SetLookupTable(svtkLookupTable* lut)
{
  if (this->LookupTable != lut)
  {
    if (this->LookupTable)
    {
      this->LookupTable->UnRegister(this);
    }
    this->LookupTable = lut;
    if (this->LookupTable)
    {
      this->LookupTable->Register(this);
    }
    this->Modified();
  }
}

//----------------------------------------------------------------------------
double* svtkDataArray::GetTupleN(svtkIdType i, int n)
{
  int numComp = this->GetNumberOfComponents();
  if (numComp != n)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != " << n);
  }
  return this->GetTuple(i);
}

//----------------------------------------------------------------------------
double svtkDataArray::GetTuple1(svtkIdType i)
{
  int numComp = this->GetNumberOfComponents();
  if (numComp != 1)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 1");
  }
  return *(this->GetTuple(i));
}

//----------------------------------------------------------------------------
double* svtkDataArray::GetTuple2(svtkIdType i)
{
  return this->GetTupleN(i, 2);
}
//----------------------------------------------------------------------------
double* svtkDataArray::GetTuple3(svtkIdType i)
{
  return this->GetTupleN(i, 3);
}
//----------------------------------------------------------------------------
double* svtkDataArray::GetTuple4(svtkIdType i)
{
  return this->GetTupleN(i, 4);
}
//----------------------------------------------------------------------------
double* svtkDataArray::GetTuple6(svtkIdType i)
{
  return this->GetTupleN(i, 6);
}
//----------------------------------------------------------------------------
double* svtkDataArray::GetTuple9(svtkIdType i)
{
  return this->GetTupleN(i, 9);
}

//----------------------------------------------------------------------------
void svtkDataArray::SetTuple1(svtkIdType i, double value)
{
  int numComp = this->GetNumberOfComponents();
  if (numComp != 1)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 1");
  }
  this->SetTuple(i, &value);
}
//----------------------------------------------------------------------------
void svtkDataArray::SetTuple2(svtkIdType i, double val0, double val1)
{
  double tuple[2];
  int numComp = this->GetNumberOfComponents();
  if (numComp != 2)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 2");
  }
  tuple[0] = val0;
  tuple[1] = val1;
  this->SetTuple(i, tuple);
}
//----------------------------------------------------------------------------
void svtkDataArray::SetTuple3(svtkIdType i, double val0, double val1, double val2)
{
  double tuple[3];
  int numComp = this->GetNumberOfComponents();
  if (numComp != 3)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 3");
  }
  tuple[0] = val0;
  tuple[1] = val1;
  tuple[2] = val2;
  this->SetTuple(i, tuple);
}
//----------------------------------------------------------------------------
void svtkDataArray::SetTuple4(svtkIdType i, double val0, double val1, double val2, double val3)
{
  double tuple[4];
  int numComp = this->GetNumberOfComponents();
  if (numComp != 4)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 4");
  }
  tuple[0] = val0;
  tuple[1] = val1;
  tuple[2] = val2;
  tuple[3] = val3;
  this->SetTuple(i, tuple);
}
//----------------------------------------------------------------------------
void svtkDataArray::SetTuple6(
  svtkIdType i, double val0, double val1, double val2, double val3, double val4, double val5)
{
  double tuple[6];
  int numComp = this->GetNumberOfComponents();
  if (numComp != 6)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 6");
  }
  tuple[0] = val0;
  tuple[1] = val1;
  tuple[2] = val2;
  tuple[3] = val3;
  tuple[4] = val4;
  tuple[5] = val5;
  this->SetTuple(i, tuple);
}
//----------------------------------------------------------------------------
void svtkDataArray::SetTuple9(svtkIdType i, double val0, double val1, double val2, double val3,
  double val4, double val5, double val6, double val7, double val8)
{
  double tuple[9];
  int numComp = this->GetNumberOfComponents();
  if (numComp != 9)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 9");
  }
  tuple[0] = val0;
  tuple[1] = val1;
  tuple[2] = val2;
  tuple[3] = val3;
  tuple[4] = val4;
  tuple[5] = val5;
  tuple[6] = val6;
  tuple[7] = val7;
  tuple[8] = val8;
  this->SetTuple(i, tuple);
}

//----------------------------------------------------------------------------
void svtkDataArray::InsertTuple1(svtkIdType i, double value)
{
  int numComp = this->GetNumberOfComponents();
  if (numComp != 1)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 1");
  }
  this->InsertTuple(i, &value);
}
//----------------------------------------------------------------------------
void svtkDataArray::InsertTuple2(svtkIdType i, double val0, double val1)
{
  double tuple[2];
  int numComp = this->GetNumberOfComponents();
  if (numComp != 2)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 2");
  }
  tuple[0] = val0;
  tuple[1] = val1;
  this->InsertTuple(i, tuple);
}
//----------------------------------------------------------------------------
void svtkDataArray::InsertTuple3(svtkIdType i, double val0, double val1, double val2)
{
  double tuple[3];
  int numComp = this->GetNumberOfComponents();
  if (numComp != 3)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 3");
  }
  tuple[0] = val0;
  tuple[1] = val1;
  tuple[2] = val2;
  this->InsertTuple(i, tuple);
}
//----------------------------------------------------------------------------
void svtkDataArray::InsertTuple4(svtkIdType i, double val0, double val1, double val2, double val3)
{
  double tuple[4];
  int numComp = this->GetNumberOfComponents();
  if (numComp != 4)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 4");
  }
  tuple[0] = val0;
  tuple[1] = val1;
  tuple[2] = val2;
  tuple[3] = val3;
  this->InsertTuple(i, tuple);
}

//----------------------------------------------------------------------------
void svtkDataArray::InsertTuple6(
  svtkIdType i, double val0, double val1, double val2, double val3, double val4, double val5)
{
  if (this->NumberOfComponents != 6)
  {
    svtkErrorMacro("The number of components do not match the number requested: "
      << this->NumberOfComponents << " != 6");
  }
  double tuple[6] = { val0, val1, val2, val3, val4, val5 };
  this->InsertTuple(i, tuple);
}

//----------------------------------------------------------------------------
void svtkDataArray::InsertTuple9(svtkIdType i, double val0, double val1, double val2, double val3,
  double val4, double val5, double val6, double val7, double val8)
{
  double tuple[9];
  int numComp = this->GetNumberOfComponents();
  if (numComp != 9)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 9");
  }
  tuple[0] = val0;
  tuple[1] = val1;
  tuple[2] = val2;
  tuple[3] = val3;
  tuple[4] = val4;
  tuple[5] = val5;
  tuple[6] = val6;
  tuple[7] = val7;
  tuple[8] = val8;
  this->InsertTuple(i, tuple);
}

//----------------------------------------------------------------------------
void svtkDataArray::InsertNextTuple1(double value)
{
  int numComp = this->GetNumberOfComponents();
  if (numComp != 1)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 1");
  }
  this->InsertNextTuple(&value);
}
//----------------------------------------------------------------------------
void svtkDataArray::InsertNextTuple2(double val0, double val1)
{
  double tuple[2];
  int numComp = this->GetNumberOfComponents();
  if (numComp != 2)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 2");
  }
  tuple[0] = val0;
  tuple[1] = val1;
  this->InsertNextTuple(tuple);
}
//----------------------------------------------------------------------------
void svtkDataArray::InsertNextTuple3(double val0, double val1, double val2)
{
  double tuple[3];
  int numComp = this->GetNumberOfComponents();
  if (numComp != 3)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 3");
  }
  tuple[0] = val0;
  tuple[1] = val1;
  tuple[2] = val2;
  this->InsertNextTuple(tuple);
}
//----------------------------------------------------------------------------
void svtkDataArray::InsertNextTuple4(double val0, double val1, double val2, double val3)
{
  double tuple[4];
  int numComp = this->GetNumberOfComponents();
  if (numComp != 4)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 4");
  }
  tuple[0] = val0;
  tuple[1] = val1;
  tuple[2] = val2;
  tuple[3] = val3;
  this->InsertNextTuple(tuple);
}

//----------------------------------------------------------------------------
void svtkDataArray::InsertNextTuple6(
  double val0, double val1, double val2, double val3, double val4, double val5)
{
  if (this->NumberOfComponents != 6)
  {
    svtkErrorMacro("The number of components do not match the number requested: "
      << this->NumberOfComponents << " != 6");
  }

  double tuple[6] = { val0, val1, val2, val3, val4, val5 };
  this->InsertNextTuple(tuple);
}

//----------------------------------------------------------------------------
void svtkDataArray::InsertNextTuple9(double val0, double val1, double val2, double val3, double val4,
  double val5, double val6, double val7, double val8)
{
  double tuple[9];
  int numComp = this->GetNumberOfComponents();
  if (numComp != 9)
  {
    svtkErrorMacro(
      "The number of components do not match the number requested: " << numComp << " != 9");
  }
  tuple[0] = val0;
  tuple[1] = val1;
  tuple[2] = val2;
  tuple[3] = val3;
  tuple[4] = val4;
  tuple[5] = val5;
  tuple[6] = val6;
  tuple[7] = val7;
  tuple[8] = val8;
  this->InsertNextTuple(tuple);
}

//----------------------------------------------------------------------------
unsigned long svtkDataArray::GetActualMemorySize() const
{
  svtkIdType numPrims;
  double size;
  // The allocated array may be larger than the number of primitives used.
  // numPrims = this->GetNumberOfTuples() * this->GetNumberOfComponents();
  numPrims = this->GetSize();

  size = svtkDataArray::GetDataTypeSize(this->GetDataType());

  // kibibytes
  return static_cast<unsigned long>(ceil((size * static_cast<double>(numPrims)) / 1024.0));
}

//----------------------------------------------------------------------------
svtkDataArray* svtkDataArray::CreateDataArray(int dataType)
{
  svtkAbstractArray* aa = svtkAbstractArray::CreateArray(dataType);
  svtkDataArray* da = svtkDataArray::FastDownCast(aa);
  if (!da && aa)
  {
    // Requested array is not a svtkDataArray. Delete the allocated array.
    aa->Delete();
  }
  return da;
}

//----------------------------------------------------------------------------
void svtkDataArray::GetTuples(svtkIdList* tupleIds, svtkAbstractArray* aa)
{
  svtkDataArray* da = svtkDataArray::FastDownCast(aa);
  if (!da)
  {
    svtkErrorMacro("Input is not a svtkDataArray, but " << aa->GetClassName());
    return;
  }

  if ((da->GetNumberOfComponents() != this->GetNumberOfComponents()))
  {
    svtkErrorMacro("Number of components for input and output do not match.\n"
                  "Source: "
      << this->GetNumberOfComponents()
      << "\n"
         "Destination: "
      << da->GetNumberOfComponents());
    return;
  }

  GetTuplesFromListWorker worker(tupleIds);
  if (!svtkArrayDispatch::Dispatch2::Execute(this, da, worker))
  {
    // Use fallback if dispatch fails.
    worker(this, da);
  }
}

//----------------------------------------------------------------------------
void svtkDataArray::GetTuples(svtkIdType p1, svtkIdType p2, svtkAbstractArray* aa)
{
  svtkDataArray* da = svtkDataArray::FastDownCast(aa);
  if (!da)
  {
    svtkWarningMacro("Input is not a svtkDataArray.");
    return;
  }

  if ((da->GetNumberOfComponents() != this->GetNumberOfComponents()))
  {
    svtkErrorMacro("Number of components for input and output do not match.\n"
                  "Source: "
      << this->GetNumberOfComponents()
      << "\n"
         "Destination: "
      << da->GetNumberOfComponents());
    return;
  }

  GetTuplesRangeWorker worker(p1, p2);
  if (!svtkArrayDispatch::Dispatch2::Execute(this, da, worker))
  {
    // Use fallback if dispatch fails.
    worker(this, da);
  }
}

//----------------------------------------------------------------------------
void svtkDataArray::FillComponent(int compIdx, double value)
{
  if (compIdx < 0 || compIdx >= this->GetNumberOfComponents())
  {
    svtkErrorMacro(<< "Specified component " << compIdx << " is not in [0, "
                  << this->GetNumberOfComponents() << ")");
    return;
  }

  // Xcode 8.2 calls GetNumberOfTuples() after each iteration.
  // Prevent this by storing the result in a local variable.
  svtkIdType numberOfTuples = this->GetNumberOfTuples();
  for (svtkIdType i = 0; i < numberOfTuples; i++)
  {
    this->SetComponent(i, compIdx, value);
  }
}

//----------------------------------------------------------------------------
void svtkDataArray::Fill(double value)
{
  for (int i = 0; i < this->GetNumberOfComponents(); ++i)
  {
    this->FillComponent(i, value);
  }
}

//----------------------------------------------------------------------------
void svtkDataArray::CopyComponent(int dstComponent, svtkDataArray* src, int srcComponent)
{
  if (this->GetNumberOfTuples() != src->GetNumberOfTuples())
  {
    svtkErrorMacro(<< "Number of tuples in 'from' (" << src->GetNumberOfTuples() << ") and 'to' ("
                  << this->GetNumberOfTuples() << ") do not match.");
    return;
  }

  if (dstComponent < 0 || dstComponent >= this->GetNumberOfComponents())
  {
    svtkErrorMacro(<< "Specified component " << dstComponent << " in 'to' array is not in [0, "
                  << this->GetNumberOfComponents() << ")");
    return;
  }

  if (srcComponent < 0 || srcComponent >= src->GetNumberOfComponents())
  {
    svtkErrorMacro(<< "Specified component " << srcComponent << " in 'from' array is not in [0, "
                  << src->GetNumberOfComponents() << ")");
    return;
  }

  svtkIdType i;
  for (i = 0; i < this->GetNumberOfTuples(); i++)
  {
    this->SetComponent(i, dstComponent, src->GetComponent(i, srcComponent));
  }
}

//----------------------------------------------------------------------------
double svtkDataArray::GetMaxNorm()
{
  svtkIdType i;
  double norm, maxNorm;
  int nComponents = this->GetNumberOfComponents();

  maxNorm = 0.0;
  for (i = 0; i < this->GetNumberOfTuples(); i++)
  {
    norm = svtkMath::Norm(this->GetTuple(i), nComponents);
    if (norm > maxNorm)
    {
      maxNorm = norm;
    }
  }

  return maxNorm;
}

//----------------------------------------------------------------------------
int svtkDataArray::CopyInformation(svtkInformation* infoFrom, int deep)
{
  // Copy everything + give base classes a chance to
  // Exclude keys which they don't want copied.
  this->Superclass::CopyInformation(infoFrom, deep);

  // Remove any keys we own that are not to be copied here.
  svtkInformation* myInfo = this->GetInformation();
  // Range:
  if (myInfo->Has(L2_NORM_RANGE()))
  {
    myInfo->Remove(L2_NORM_RANGE());
  }

  return 1;
}

//----------------------------------------------------------------------------
void svtkDataArray::ComputeFiniteRange(double range[2], int comp)
{
  // this method needs a large refactoring to be way easier to read

  if (comp >= this->NumberOfComponents)
  { // Ignore requests for nonexistent components.
    return;
  }
  // If we got component -1 on a vector array, compute vector magnitude.
  if (comp < 0 && this->NumberOfComponents == 1)
  {
    comp = 0;
  }

  range[0] = svtkTypeTraits<double>::Max();
  range[1] = svtkTypeTraits<double>::Min();

  svtkInformation* info = this->GetInformation();
  svtkInformationDoubleVectorKey* rkey;
  if (comp < 0)
  {
    rkey = L2_NORM_FINITE_RANGE();
    // hasValidKey will update range to the cached value if it exists.
    if (!hasValidKey(info, rkey, range))
    {

      this->ComputeFiniteVectorRange(range);
      info->Set(rkey, range, 2);
    }
    return;
  }
  else
  {
    rkey = COMPONENT_RANGE();

    // hasValidKey will update range to the cached value if it exists.
    if (!hasValidKey(info, PER_FINITE_COMPONENT(), rkey, range, comp))
    {
      double* allCompRanges = new double[this->NumberOfComponents * 2];
      const bool computed = this->ComputeFiniteScalarRange(allCompRanges);
      if (computed)
      {
        // construct the keys and add them to the info object
        svtkInformationVector* infoVec = svtkInformationVector::New();
        info->Set(PER_FINITE_COMPONENT(), infoVec);

        infoVec->SetNumberOfInformationObjects(this->NumberOfComponents);
        for (int i = 0; i < this->NumberOfComponents; ++i)
        {
          infoVec->GetInformationObject(i)->Set(rkey, allCompRanges + (i * 2), 2);
        }
        infoVec->FastDelete();

        // update the range passed in since we have a valid range.
        range[0] = allCompRanges[comp * 2];
        range[1] = allCompRanges[(comp * 2) + 1];
      }
      delete[] allCompRanges;
    }
  }
}

//----------------------------------------------------------------------------
void svtkDataArray::ComputeRange(double range[2], int comp)
{
  // this method needs a large refactoring to be way easier to read

  if (comp >= this->NumberOfComponents)
  { // Ignore requests for nonexistent components.
    return;
  }
  // If we got component -1 on a vector array, compute vector magnitude.
  if (comp < 0 && this->NumberOfComponents == 1)
  {
    comp = 0;
  }

  range[0] = svtkTypeTraits<double>::Max();
  range[1] = svtkTypeTraits<double>::Min();

  svtkInformation* info = this->GetInformation();
  svtkInformationDoubleVectorKey* rkey;
  if (comp < 0)
  {
    rkey = L2_NORM_RANGE();
    // hasValidKey will update range to the cached value if it exists.
    if (!hasValidKey(info, rkey, range))
    {
      this->ComputeVectorRange(range);
      info->Set(rkey, range, 2);
    }
    return;
  }
  else
  {
    rkey = COMPONENT_RANGE();

    // hasValidKey will update range to the cached value if it exists.
    if (!hasValidKey(info, PER_COMPONENT(), rkey, range, comp))
    {
      double* allCompRanges = new double[this->NumberOfComponents * 2];
      const bool computed = this->ComputeScalarRange(allCompRanges);
      if (computed)
      {
        // construct the keys and add them to the info object
        svtkInformationVector* infoVec = svtkInformationVector::New();
        info->Set(PER_COMPONENT(), infoVec);

        infoVec->SetNumberOfInformationObjects(this->NumberOfComponents);
        for (int i = 0; i < this->NumberOfComponents; ++i)
        {
          infoVec->GetInformationObject(i)->Set(rkey, allCompRanges + (i * 2), 2);
        }
        infoVec->FastDelete();

        // update the range passed in since we have a valid range.
        range[0] = allCompRanges[comp * 2];
        range[1] = allCompRanges[(comp * 2) + 1];
      }
      delete[] allCompRanges;
    }
  }
}

//----------------------------------------------------------------------------
// call modified on superclass
void svtkDataArray::Modified()
{
  if (this->HasInformation())
  {
    // Clear key-value pairs that are now out of date.
    svtkInformation* info = this->GetInformation();
    info->Remove(L2_NORM_RANGE());
    info->Remove(L2_NORM_FINITE_RANGE());
  }
  this->Superclass::Modified();
}

namespace
{

// Wrap the DoCompute[Scalar|Vector]Range calls for svtkArrayDispatch:
struct ScalarRangeDispatchWrapper
{
  bool Success;
  double* Range;

  ScalarRangeDispatchWrapper(double* range)
    : Success(false)
    , Range(range)
  {
  }

  template <typename ArrayT>
  void operator()(ArrayT* array)
  {
    this->Success = svtkDataArrayPrivate::DoComputeScalarRange(
      array, this->Range, svtkDataArrayPrivate::AllValues());
  }
};

struct VectorRangeDispatchWrapper
{
  bool Success;
  double* Range;

  VectorRangeDispatchWrapper(double* range)
    : Success(false)
    , Range(range)
  {
  }

  template <typename ArrayT>
  void operator()(ArrayT* array)
  {
    this->Success = svtkDataArrayPrivate::DoComputeVectorRange(
      array, this->Range, svtkDataArrayPrivate::AllValues());
  }
};

// Wrap the DoCompute[Scalar|Vector]Range calls for svtkArrayDispatch:
struct FiniteScalarRangeDispatchWrapper
{
  bool Success;
  double* Range;

  FiniteScalarRangeDispatchWrapper(double* range)
    : Success(false)
    , Range(range)
  {
  }

  template <typename ArrayT>
  void operator()(ArrayT* array)
  {
    this->Success = svtkDataArrayPrivate::DoComputeScalarRange(
      array, this->Range, svtkDataArrayPrivate::FiniteValues());
  }
};

struct FiniteVectorRangeDispatchWrapper
{
  bool Success;
  double* Range;

  FiniteVectorRangeDispatchWrapper(double* range)
    : Success(false)
    , Range(range)
  {
  }

  template <typename ArrayT>
  void operator()(ArrayT* array)
  {
    this->Success = svtkDataArrayPrivate::DoComputeVectorRange(
      array, this->Range, svtkDataArrayPrivate::FiniteValues());
  }
};

} // end anon namespace

//----------------------------------------------------------------------------
bool svtkDataArray::ComputeScalarRange(double* ranges)
{
  ScalarRangeDispatchWrapper worker(ranges);
  if (!svtkArrayDispatch::Dispatch::Execute(this, worker))
  {
    worker(this);
  }
  return worker.Success;
}

//-----------------------------------------------------------------------------
bool svtkDataArray::ComputeVectorRange(double range[2])
{
  VectorRangeDispatchWrapper worker(range);
  if (!svtkArrayDispatch::Dispatch::Execute(this, worker))
  {
    worker(this);
  }
  return worker.Success;
}

//----------------------------------------------------------------------------
bool svtkDataArray::ComputeFiniteScalarRange(double* ranges)
{
  FiniteScalarRangeDispatchWrapper worker(ranges);
  if (!svtkArrayDispatch::Dispatch::Execute(this, worker))
  {
    worker(this);
  }
  return worker.Success;
}

//-----------------------------------------------------------------------------
bool svtkDataArray::ComputeFiniteVectorRange(double range[2])
{
  FiniteVectorRangeDispatchWrapper worker(range);
  if (!svtkArrayDispatch::Dispatch::Execute(this, worker))
  {
    worker(this);
  }
  return worker.Success;
}

//----------------------------------------------------------------------------
void svtkDataArray::GetDataTypeRange(double range[2])
{
  svtkDataArray::GetDataTypeRange(this->GetDataType(), range);
}

//----------------------------------------------------------------------------
double svtkDataArray::GetDataTypeMin()
{
  return svtkDataArray::GetDataTypeMin(this->GetDataType());
}

//----------------------------------------------------------------------------
double svtkDataArray::GetDataTypeMax()
{
  return svtkDataArray::GetDataTypeMax(this->GetDataType());
}

//----------------------------------------------------------------------------
void svtkDataArray::GetDataTypeRange(int type, double range[2])
{
  range[0] = svtkDataArray::GetDataTypeMin(type);
  range[1] = svtkDataArray::GetDataTypeMax(type);
}

//----------------------------------------------------------------------------
double svtkDataArray::GetDataTypeMin(int type)
{
  switch (type)
  {
    case SVTK_BIT:
      return static_cast<double>(SVTK_BIT_MIN);
    case SVTK_SIGNED_CHAR:
      return static_cast<double>(SVTK_SIGNED_CHAR_MIN);
    case SVTK_UNSIGNED_CHAR:
      return static_cast<double>(SVTK_UNSIGNED_CHAR_MIN);
    case SVTK_CHAR:
      return static_cast<double>(SVTK_CHAR_MIN);
    case SVTK_UNSIGNED_SHORT:
      return static_cast<double>(SVTK_UNSIGNED_SHORT_MIN);
    case SVTK_SHORT:
      return static_cast<double>(SVTK_SHORT_MIN);
    case SVTK_UNSIGNED_INT:
      return static_cast<double>(SVTK_UNSIGNED_INT_MIN);
    case SVTK_INT:
      return static_cast<double>(SVTK_INT_MIN);
    case SVTK_UNSIGNED_LONG:
      return static_cast<double>(SVTK_UNSIGNED_LONG_MIN);
    case SVTK_LONG:
      return static_cast<double>(SVTK_LONG_MIN);
    case SVTK_UNSIGNED_LONG_LONG:
      return static_cast<double>(SVTK_UNSIGNED_LONG_LONG_MIN);
    case SVTK_LONG_LONG:
      return static_cast<double>(SVTK_LONG_LONG_MIN);
    case SVTK_FLOAT:
      return static_cast<double>(SVTK_FLOAT_MIN);
    case SVTK_DOUBLE:
      return static_cast<double>(SVTK_DOUBLE_MIN);
    case SVTK_ID_TYPE:
      return static_cast<double>(SVTK_ID_MIN);
    default:
      return 0;
  }
}

//----------------------------------------------------------------------------
double svtkDataArray::GetDataTypeMax(int type)
{
  switch (type)
  {
    case SVTK_BIT:
      return static_cast<double>(SVTK_BIT_MAX);
    case SVTK_SIGNED_CHAR:
      return static_cast<double>(SVTK_SIGNED_CHAR_MAX);
    case SVTK_UNSIGNED_CHAR:
      return static_cast<double>(SVTK_UNSIGNED_CHAR_MAX);
    case SVTK_CHAR:
      return static_cast<double>(SVTK_CHAR_MAX);
    case SVTK_UNSIGNED_SHORT:
      return static_cast<double>(SVTK_UNSIGNED_SHORT_MAX);
    case SVTK_SHORT:
      return static_cast<double>(SVTK_SHORT_MAX);
    case SVTK_UNSIGNED_INT:
      return static_cast<double>(SVTK_UNSIGNED_INT_MAX);
    case SVTK_INT:
      return static_cast<double>(SVTK_INT_MAX);
    case SVTK_UNSIGNED_LONG:
      return static_cast<double>(SVTK_UNSIGNED_LONG_MAX);
    case SVTK_LONG:
      return static_cast<double>(SVTK_LONG_MAX);
    case SVTK_UNSIGNED_LONG_LONG:
      return static_cast<double>(SVTK_UNSIGNED_LONG_LONG_MAX);
    case SVTK_LONG_LONG:
      return static_cast<double>(SVTK_LONG_LONG_MAX);
    case SVTK_FLOAT:
      return static_cast<double>(SVTK_FLOAT_MAX);
    case SVTK_DOUBLE:
      return static_cast<double>(SVTK_DOUBLE_MAX);
    case SVTK_ID_TYPE:
      return static_cast<double>(SVTK_ID_MAX);
    default:
      return 1;
  }
}

//----------------------------------------------------------------------------
void svtkDataArray::RemoveLastTuple()
{
  if (this->GetNumberOfTuples() > 0)
  {
    this->Resize(this->GetNumberOfTuples() - 1);
  }
}

//----------------------------------------------------------------------------
void svtkDataArray::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  const char* name = this->GetName();
  if (name)
  {
    os << indent << "Name: " << name << "\n";
  }
  else
  {
    os << indent << "Name: (none)\n";
  }
  os << indent << "Number Of Components: " << this->NumberOfComponents << "\n";
  os << indent << "Number Of Tuples: " << this->GetNumberOfTuples() << "\n";
  os << indent << "Size: " << this->Size << "\n";
  os << indent << "MaxId: " << this->MaxId << "\n";
  if (this->LookupTable)
  {
    os << indent << "Lookup Table:\n";
    this->LookupTable->PrintSelf(os, indent.GetNextIndent());
  }
  else
  {
    os << indent << "LookupTable: (none)\n";
  }
}
