/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCellArray.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkCellArray.h"

#include "svtkArrayDispatch.h"
#include "svtkCellArrayIterator.h"
#include "svtkDataArrayRange.h"
#include "svtkIdTypeArray.h"
#include "svtkIntArray.h"
#include "svtkLongArray.h"
#include "svtkLongLongArray.h"
#include "svtkObjectFactory.h"
#include "svtkSMPThreadLocal.h"
#include "svtkSMPTools.h"

#include <algorithm>
#include <array>
#include <iterator>

namespace
{

// These implementations are for methods that will be deprecated in the future:
namespace deprec
{

struct GetSizeImpl
{
  template <typename CellStateT>
  svtkIdType operator()(CellStateT& cells) const
  {
    return (cells.GetOffsets()->GetSize() + cells.GetConnectivity()->GetSize());
  }
};

// Given a legacy Location, find the corresponding cellId. The location
// *must* refer to a [numPts] entry in the old connectivity array, or the
// returned CellId will be -1.
struct LocationToCellIdFunctor
{
  template <typename CellStateT>
  svtkIdType operator()(CellStateT& cells, svtkIdType location) const
  {
    using ValueType = typename CellStateT::ValueType;

    const auto offsets = svtk::DataArrayValueRange<1>(cells.GetOffsets());

    // Use a binary-search to find the location:
    auto it = this->BinarySearchOffset(
      offsets.begin(), offsets.end() - 1, static_cast<ValueType>(location));

    const svtkIdType cellId = std::distance(offsets.begin(), it);

    if (it == offsets.end() - 1 /* no match found */ ||
      (*it + cellId) != location /* `location` not at cell head */)
    { // Location invalid.
      return -1;
    }

    // return the cell id:
    return cellId;
  }

  template <typename IterT>
  IterT BinarySearchOffset(const IterT& beginIter, const IterT& endIter,
    const typename std::iterator_traits<IterT>::value_type& targetLocation) const
  {
    using ValueType = typename std::iterator_traits<IterT>::value_type;
    using DifferenceType = typename std::iterator_traits<IterT>::difference_type;

    DifferenceType roiSize = std::distance(beginIter, endIter);

    IterT roiBegin = beginIter;
    while (roiSize > 0)
    {
      IterT it = roiBegin;
      const DifferenceType step = roiSize / 2;
      std::advance(it, step);
      // This differs from a generic binary search in the following line:
      // Adding the distance from the start of the array to the current
      // iterator will account for the cellSize entries in the old cell array
      // format, such that curLocation would be the offset in the old style
      // connectivity array.
      const ValueType curLocation = *it + std::distance(beginIter, it);
      if (curLocation < targetLocation)
      {
        roiBegin = ++it;
        roiSize -= step + 1;
      }
      else
      {
        roiSize = step;
      }
    }

    return roiBegin;
  }
};

struct CellIdToLocationFunctor
{
  template <typename CellStateT>
  svtkIdType operator()(CellStateT& cells, svtkIdType cellId) const
  {
    // Adding the cellId to the offset of that cell id gives us the cell
    // location in the old-style svtkCellArray connectivity array.
    return static_cast<svtkIdType>(cells.GetOffsets()->GetValue(cellId)) + cellId;
  }
};

struct GetInsertLocationImpl
{
  template <typename CellStateT>
  svtkIdType operator()(CellStateT& cells) const
  {
    // The insert location used to just be the tail of the connectivity array.
    // Compute the equivalent value:
    return (
      (cells.GetOffsets()->GetNumberOfValues() - 1) + cells.GetConnectivity()->GetNumberOfValues());
  }
};

} // end namespace deprec

struct PrintDebugImpl
{
  template <typename CellStateT>
  void operator()(CellStateT& state, std::ostream& os)
  {
    using ValueType = typename CellStateT::ValueType;

    const svtkIdType numCells = state.GetNumberOfCells();
    for (svtkIdType cellId = 0; cellId < numCells; ++cellId)
    {
      os << "cell " << cellId << ": ";

      const auto cellRange = state.GetCellRange(cellId);
      for (ValueType ptId : cellRange)
      {
        os << ptId << " ";
      }

      os << "\n";
    }
  }
};

struct InitializeImpl
{
  template <typename CellStateT>
  void operator()(CellStateT& cells) const
  {
    cells.GetConnectivity()->Initialize();
    cells.GetOffsets()->Initialize();
    cells.GetOffsets()->InsertNextValue(0);
  }
};

struct SqueezeImpl
{
  template <typename CellStateT>
  void operator()(CellStateT& cells) const
  {
    cells.GetConnectivity()->Squeeze();
    cells.GetOffsets()->Squeeze();
  }
};

struct IsValidImpl
{
  template <typename CellStateT>
  bool operator()(CellStateT& state) const
  {
    using ValueType = typename CellStateT::ValueType;
    auto* offsetArray = state.GetOffsets();
    auto* connArray = state.GetConnectivity();

    // Both arrays must be single component
    if (offsetArray->GetNumberOfComponents() != 1 || connArray->GetNumberOfComponents() != 1)
    {
      return false;
    }

    auto offsets = svtk::DataArrayValueRange<1>(offsetArray);

    // Offsets must have at least one value, and the first value must be zero
    if (offsets.size() == 0 || *offsets.cbegin() != 0)
    {
      return false;
    }

    // Values in offsets must not decrease
    auto it = std::adjacent_find(offsets.cbegin(), offsets.cend(),
      [](const ValueType a, const ValueType b) -> bool { return a > b; });
    if (it != offsets.cend())
    {
      return false;
    }

    // The last value in offsets must be the size of the connectivity array.
    if (connArray->GetNumberOfValues() != *(offsets.cend() - 1))
    {
      return false;
    }

    return true;
  }
};

template <typename T>
struct CanConvert
{
  template <typename CellStateT>
  bool operator()(CellStateT& state) const
  {
    using ArrayType = typename CellStateT::ArrayType;
    using ValueType = typename CellStateT::ValueType;

    // offsets are sorted, so just check the last value, but we have to compute
    // the full range of the connectivity array.
    auto* off = state.GetOffsets();
    if (off->GetNumberOfValues() > 0 && !this->CheckValue(off->GetValue(off->GetMaxId())))
    {
      return false;
    }

    std::array<ValueType, 2> connRange;
    auto* mutConn = const_cast<ArrayType*>(state.GetConnectivity());
    if (mutConn->GetNumberOfValues() > 0)
    {
      mutConn->GetValueRange(connRange.data(), 0);
      if (!this->CheckValue(connRange[0]) || !this->CheckValue(connRange[1]))
      {
        return false;
      }
    }

    return true;
  }

  template <typename U>
  bool CheckValue(const U& val) const
  {
    return val == static_cast<U>(static_cast<T>(val));
  }
};

struct ExtractAndInitialize
{
  template <typename CellStateT, typename TargetArrayT>
  bool operator()(CellStateT& state, TargetArrayT* offsets, TargetArrayT* conn) const
  {
    return (
      this->Process(state.GetOffsets(), offsets) && this->Process(state.GetConnectivity(), conn));
  }

  template <typename SourceArrayT, typename TargetArrayT>
  bool Process(SourceArrayT* src, TargetArrayT* dst) const
  {
    // Check that allocation suceeds:
    if (!dst->Resize(src->GetNumberOfTuples()))
    {
      return false;
    }

    // Copy data:
    dst->DeepCopy(src);

    // Free old memory:
    src->Resize(0);

    return true;
  }
};

struct IsHomogeneousImpl
{
  template <typename CellArraysT>
  svtkIdType operator()(CellArraysT& state) const
  {
    using ValueType = typename CellArraysT::ValueType;
    auto* offsets = state.GetOffsets();

    const svtkIdType numCells = state.GetNumberOfCells();
    if (numCells == 0)
    {
      return 0;
    }

    // Initialize using the first cell:
    const svtkIdType firstCellSize = state.GetCellSize(0);

    // Verify the rest:
    auto offsetRange = svtk::DataArrayValueRange<1>(offsets);
    auto it = std::adjacent_find(offsetRange.begin() + 1, offsetRange.end(),
      [&](const ValueType a, const ValueType b) -> bool { return (b - a != firstCellSize); });

    if (it != offsetRange.end())
    { // Found a cell that doesn't match the size of the first cell:
      return -1;
    }

    return firstCellSize;
  }
};

struct AllocateExactImpl
{
  template <typename CellStateT>
  bool operator()(CellStateT& cells, svtkIdType numCells, svtkIdType connectivitySize) const
  {
    const bool result = (cells.GetOffsets()->Allocate(numCells + 1) &&
      cells.GetConnectivity()->Allocate(connectivitySize));
    if (result)
    {
      cells.GetOffsets()->InsertNextValue(0);
    }

    return result;
  }
};

struct ResizeExactImpl
{
  template <typename CellStateT>
  bool operator()(CellStateT& cells, svtkIdType numCells, svtkIdType connectivitySize) const
  {
    return (cells.GetOffsets()->SetNumberOfValues(numCells + 1) &&
      cells.GetConnectivity()->SetNumberOfValues(connectivitySize));
  }
};

struct FindMaxCell // SMP functor
{
  svtkCellArray* CellArray;
  svtkIdType Result{ 0 };
  svtkSMPThreadLocal<svtkIdType> LocalResult;

  FindMaxCell(svtkCellArray* array)
    : CellArray{ array }
  {
  }

  void Initialize() { this->LocalResult.Local() = 0; }

  struct Impl
  {
    template <typename CellStateT>
    svtkIdType operator()(CellStateT& cells, svtkIdType cellId, const svtkIdType endCellId) const
    {
      svtkIdType result = 0;
      for (; cellId < endCellId; ++cellId)
      {
        result = std::max(result, cells.GetCellSize(cellId));
      }
      return result;
    }
  };

  void operator()(svtkIdType cellId, svtkIdType endCellId)
  {
    svtkIdType& lval = this->LocalResult.Local();
    lval = std::max(lval, this->CellArray->Visit(Impl{}, cellId, endCellId));
  }

  void Reduce()
  {
    for (const svtkIdType lResult : this->LocalResult)
    {
      this->Result = std::max(this->Result, lResult);
    }
  }
};

struct GetActualMemorySizeImpl
{
  template <typename CellStateT>
  unsigned long operator()(CellStateT& cells) const
  {
    return (
      cells.GetOffsets()->GetActualMemorySize() + cells.GetConnectivity()->GetActualMemorySize());
  }
};

struct PrintSelfImpl
{
  template <typename CellStateT>
  void operator()(CellStateT& cells, ostream& os, svtkIndent indent) const
  {
    os << indent << "Offsets:\n";
    cells.GetOffsets()->PrintSelf(os, indent.GetNextIndent());
    os << indent << "Connectivity:\n";
    cells.GetConnectivity()->PrintSelf(os, indent.GetNextIndent());
  }
};

struct GetLegacyDataSizeImpl
{
  template <typename CellStateT>
  svtkIdType operator()(CellStateT& cells) const
  {
    return (
      (cells.GetOffsets()->GetNumberOfValues() - 1) + cells.GetConnectivity()->GetNumberOfValues());
  }
};

struct ReverseCellAtIdImpl
{
  template <typename CellStateT>
  void operator()(CellStateT& cells, svtkIdType cellId) const
  {
    auto cellRange = cells.GetCellRange(cellId);
    std::reverse(cellRange.begin(), cellRange.end());
  }
};

struct ReplaceCellAtIdImpl
{
  template <typename CellStateT>
  void operator()(
    CellStateT& cells, svtkIdType cellId, svtkIdType cellSize, const svtkIdType* cellPoints) const
  {
    using ValueType = typename CellStateT::ValueType;

    auto cellRange = cells.GetCellRange(cellId);

    assert(cellRange.size() == cellSize);
    for (svtkIdType i = 0; i < cellSize; ++i)
    {
      cellRange[i] = static_cast<ValueType>(cellPoints[i]);
    }
  }
};

struct AppendLegacyFormatImpl
{
  template <typename CellStateT>
  void operator()(
    CellStateT& cells, const svtkIdType* data, const svtkIdType len, const svtkIdType ptOffset) const
  {
    using ValueType = typename CellStateT::ValueType;

    ValueType offset = static_cast<ValueType>(cells.GetConnectivity()->GetNumberOfValues());

    const svtkIdType* const dataEnd = data + len;
    while (data < dataEnd)
    {
      svtkIdType numPts = *data++;
      offset += static_cast<ValueType>(numPts);
      cells.GetOffsets()->InsertNextValue(offset);
      while (numPts-- > 0)
      {
        cells.GetConnectivity()->InsertNextValue(static_cast<ValueType>(*data++ + ptOffset));
      }
    }
  }
};

struct AppendImpl
{
  // Call this signature:
  template <typename DstCellStateT>
  void operator()(DstCellStateT& dstcells, svtkCellArray* src, svtkIdType pointOffset) const
  { // dispatch on src:
    src->Visit(*this, dstcells, pointOffset);
  }

  // Above signature calls this operator in Visit:
  template <typename SrcCellStateT, typename DstCellStateT>
  void operator()(SrcCellStateT& src, DstCellStateT& dst, svtkIdType pointOffsets) const
  {
    this->AppendArrayWithOffset(
      src.GetOffsets(), dst.GetOffsets(), dst.GetConnectivity()->GetNumberOfValues(), true);
    this->AppendArrayWithOffset(src.GetConnectivity(), dst.GetConnectivity(), pointOffsets, false);
  }

  // Assumes both arrays are 1 component. src's data is appended to dst with
  // offset added to each value.
  template <typename SrcArrayT, typename DstArrayT>
  void AppendArrayWithOffset(
    SrcArrayT* srcArray, DstArrayT* dstArray, svtkIdType offset, bool skipFirst) const
  {
    SVTK_ASSUME(srcArray->GetNumberOfComponents() == 1);
    SVTK_ASSUME(dstArray->GetNumberOfComponents() == 1);

    using SrcValueType = svtk::GetAPIType<SrcArrayT>;
    using DstValueType = svtk::GetAPIType<DstArrayT>;

    const svtkIdType srcSize =
      skipFirst ? srcArray->GetNumberOfValues() - 1 : srcArray->GetNumberOfValues();
    const svtkIdType dstBegin = dstArray->GetNumberOfValues();
    const svtkIdType dstEnd = dstBegin + srcSize;

    // This extends the allocation of dst to ensure we have enough space
    // allocated:
    dstArray->InsertValue(dstEnd - 1, 0);

    const auto srcRange = svtk::DataArrayValueRange<1>(srcArray, skipFirst ? 1 : 0);
    auto dstRange = svtk::DataArrayValueRange<1>(dstArray, dstBegin, dstEnd);
    assert(srcRange.size() == dstRange.size());

    const DstValueType dOffset = static_cast<DstValueType>(offset);

    std::transform(srcRange.cbegin(), srcRange.cend(), dstRange.begin(),
      [&](SrcValueType x) -> DstValueType { return static_cast<DstValueType>(x) + dOffset; });
  }
};

} // end anon namespace

svtkCellArray::svtkCellArray() = default;
svtkCellArray::~svtkCellArray() = default;
svtkStandardNewMacro(svtkCellArray);

//=================== Begin Legacy Methods ===================================
// These should be deprecated at some point as they are confusing or very slow

//----------------------------------------------------------------------------
svtkIdType svtkCellArray::GetSize()
{
  // We can still compute roughly the same result, so go ahead and do that.
  return this->Visit(deprec::GetSizeImpl{});
}

//----------------------------------------------------------------------------
svtkIdType svtkCellArray::GetNumberOfConnectivityEntries()
{
  // We can still compute roughly the same result, so go ahead and do that.
  return this->Visit(GetLegacyDataSizeImpl{});
}

//----------------------------------------------------------------------------
void svtkCellArray::GetCell(svtkIdType loc, svtkIdType& npts, const svtkIdType*& pts)
{
  const svtkIdType cellId = this->Visit(deprec::LocationToCellIdFunctor{}, loc);
  if (cellId < 0)
  {
    svtkErrorMacro("Invalid location.");
    npts = 0;
    pts = nullptr;
    return;
  }

  this->GetCellAtId(cellId, this->TempCell);
  npts = this->TempCell->GetNumberOfIds();
  pts = this->TempCell->GetPointer(0);
}

//----------------------------------------------------------------------------
void svtkCellArray::GetCell(svtkIdType loc, svtkIdList* pts)
{
  const svtkIdType cellId = this->Visit(deprec::LocationToCellIdFunctor{}, loc);
  if (cellId < 0)
  {
    svtkErrorMacro("Invalid location.");
    pts->Reset();
    return;
  }

  this->GetCellAtId(cellId, pts);
}

//----------------------------------------------------------------------------
svtkIdType svtkCellArray::GetInsertLocation(int npts)
{
  // It looks like the original implementation of this actually returned the
  // location of the last cell (of size npts), not the current insert location.
  return this->Visit(deprec::GetInsertLocationImpl{}) - npts - 1;
}

//----------------------------------------------------------------------------
svtkIdType svtkCellArray::GetTraversalLocation()
{
  return this->Visit(deprec::CellIdToLocationFunctor{}, this->GetTraversalCellId());
}

//----------------------------------------------------------------------------
svtkIdType svtkCellArray::GetTraversalLocation(svtkIdType npts)
{
  return this->Visit(deprec::CellIdToLocationFunctor{}, this->GetTraversalCellId()) - npts - 1;
}

//----------------------------------------------------------------------------
void svtkCellArray::SetTraversalLocation(svtkIdType loc)
{
  const svtkIdType cellId = this->Visit(deprec::LocationToCellIdFunctor{}, loc);
  if (cellId < 0)
  {
    svtkErrorMacro("Invalid location, ignoring.");
    return;
  }

  this->SetTraversalCellId(cellId);
}

//----------------------------------------------------------------------------
svtkIdType svtkCellArray::EstimateSize(svtkIdType numCells, int maxPtsPerCell)
{
  return numCells * (1 + maxPtsPerCell);
}

//----------------------------------------------------------------------------
void svtkCellArray::SetNumberOfCells(svtkIdType)
{
  // no-op
}

//----------------------------------------------------------------------------
void svtkCellArray::ReverseCell(svtkIdType loc)
{
  const svtkIdType cellId = this->Visit(deprec::LocationToCellIdFunctor{}, loc);
  if (cellId < 0)
  {
    svtkErrorMacro("Invalid location, ignoring.");
    return;
  }

  this->ReverseCellAtId(cellId);
}

//----------------------------------------------------------------------------
void svtkCellArray::ReplaceCell(svtkIdType loc, int npts, const svtkIdType pts[])
{
  const svtkIdType cellId = this->Visit(deprec::LocationToCellIdFunctor{}, loc);
  if (cellId < 0)
  {
    svtkErrorMacro("Invalid location, ignoring.");
    return;
  }

  this->ReplaceCellAtId(cellId, static_cast<svtkIdType>(npts), pts);
}

//----------------------------------------------------------------------------
svtkIdTypeArray* svtkCellArray::GetData()
{
  this->ExportLegacyFormat(this->LegacyData);

  return this->LegacyData;
}

//----------------------------------------------------------------------------
// Specify a group of cells.
void svtkCellArray::SetCells(svtkIdType ncells, svtkIdTypeArray* cells)
{
  this->AllocateExact(ncells, cells->GetNumberOfValues() - ncells);
  this->ImportLegacyFormat(cells);
}

//=================== End Legacy Methods =====================================

//----------------------------------------------------------------------------
void svtkCellArray::DeepCopy(svtkCellArray* ca)
{
  if (ca == this)
  {
    return;
  }

  if (ca->Storage.Is64Bit())
  {
    this->Storage.Use64BitStorage();
    auto& srcStorage = ca->Storage.GetArrays64();
    auto& dstStorage = this->Storage.GetArrays64();
    dstStorage.Offsets->DeepCopy(srcStorage.Offsets);
    dstStorage.Connectivity->DeepCopy(srcStorage.Connectivity);
    this->Modified();
  }
  else
  {
    this->Storage.Use32BitStorage();
    auto& srcStorage = ca->Storage.GetArrays32();
    auto& dstStorage = this->Storage.GetArrays32();
    dstStorage.Offsets->DeepCopy(srcStorage.Offsets);
    dstStorage.Connectivity->DeepCopy(srcStorage.Connectivity);
    this->Modified();
  }
}

//----------------------------------------------------------------------------
void svtkCellArray::ShallowCopy(svtkCellArray* ca)
{
  if (ca == this)
  {
    return;
  }

  if (ca->Storage.Is64Bit())
  {
    auto& srcStorage = ca->Storage.GetArrays64();
    this->SetData(srcStorage.GetOffsets(), srcStorage.GetConnectivity());
  }
  else
  {
    auto& srcStorage = ca->Storage.GetArrays32();
    this->SetData(srcStorage.GetOffsets(), srcStorage.GetConnectivity());
  }
}

//----------------------------------------------------------------------------
void svtkCellArray::Append(svtkCellArray* src, svtkIdType pointOffset)
{
  if (src->GetNumberOfCells() > 0)
  {
    this->Visit(AppendImpl{}, src, pointOffset);
  }
}

//----------------------------------------------------------------------------
void svtkCellArray::Initialize()
{
  this->Visit(InitializeImpl{});

  this->LegacyData->Initialize();
}

//----------------------------------------------------------------------------
svtkCellArrayIterator* svtkCellArray::NewIterator()
{
  svtkCellArrayIterator* iter = svtkCellArrayIterator::New();
  iter->SetCellArray(this);
  iter->GoToFirstCell();
  return iter;
}

//----------------------------------------------------------------------------
void svtkCellArray::SetData(svtkTypeInt32Array* offsets, svtkTypeInt32Array* connectivity)
{
  if (offsets->GetNumberOfComponents() != 1 || connectivity->GetNumberOfComponents() != 1)
  {
    svtkErrorMacro("Only single component arrays may be used for svtkCellArray "
                  "storage.");
    return;
  }

  this->Storage.Use32BitStorage();
  auto& storage = this->Storage.GetArrays32();

  // svtkArrayDownCast to ensure this works when ArrayType32 is svtkIdTypeArray.
  storage.Offsets = svtkArrayDownCast<ArrayType32>(offsets);
  storage.Connectivity = svtkArrayDownCast<ArrayType32>(connectivity);
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkCellArray::SetData(svtkTypeInt64Array* offsets, svtkTypeInt64Array* connectivity)
{
  if (offsets->GetNumberOfComponents() != 1 || connectivity->GetNumberOfComponents() != 1)
  {
    svtkErrorMacro("Only single component arrays may be used for svtkCellArray "
                  "storage.");
    return;
  }

  this->Storage.Use64BitStorage();
  auto& storage = this->Storage.GetArrays64();

  // svtkArrayDownCast to ensure this works when ArrayType64 is svtkIdTypeArray.
  storage.Offsets = svtkArrayDownCast<ArrayType64>(offsets);
  storage.Connectivity = svtkArrayDownCast<ArrayType64>(connectivity);
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkCellArray::SetData(svtkIdTypeArray* offsets, svtkIdTypeArray* connectivity)
{
#ifdef SVTK_USE_64BIT_IDS
  svtkNew<svtkTypeInt64Array> o;
  svtkNew<svtkTypeInt64Array> c;
  o->ShallowCopy(offsets);
  c->ShallowCopy(connectivity);
  this->SetData(o, c);
#else  // SVTK_USE_64BIT_IDS
  svtkNew<svtkTypeInt32Array> o;
  svtkNew<svtkTypeInt32Array> c;
  o->ShallowCopy(offsets);
  c->ShallowCopy(connectivity);
  this->SetData(o, c);
#endif // SVTK_USE_64BIT_IDS
}

//----------------------------------------------------------------------------
void svtkCellArray::SetData(
  svtkAOSDataArrayTemplate<int>* offsets, svtkAOSDataArrayTemplate<int>* connectivity)
{
#if SVTK_SIZEOF_INT == 4
  svtkNew<svtkTypeInt32Array> o;
  svtkNew<svtkTypeInt32Array> c;
  o->ShallowCopy(offsets);
  c->ShallowCopy(connectivity);
  this->SetData(o, c);
#elif SVTK_SIZEOF_INT == 8
  svtkNew<svtkTypeInt64Array> o;
  svtkNew<svtkTypeInt64Array> c;
  o->ShallowCopy(offsets);
  c->ShallowCopy(connectivity);
  this->SetData(o, c);
#else
  svtkErrorMacro("`int` type is neither 32 nor 64 bits.");
#endif
}

//----------------------------------------------------------------------------
void svtkCellArray::SetData(
  svtkAOSDataArrayTemplate<long>* offsets, svtkAOSDataArrayTemplate<long>* connectivity)
{
#if SVTK_SIZEOF_LONG == 4
  svtkNew<svtkTypeInt32Array> o;
  svtkNew<svtkTypeInt32Array> c;
  o->ShallowCopy(offsets);
  c->ShallowCopy(connectivity);
  this->SetData(o, c);
#elif SVTK_SIZEOF_LONG == 8
  svtkNew<svtkTypeInt64Array> o;
  svtkNew<svtkTypeInt64Array> c;
  o->ShallowCopy(offsets);
  c->ShallowCopy(connectivity);
  this->SetData(o, c);
#else
  svtkErrorMacro("`long` type is neither 32 nor 64 bits.");
#endif
}

//----------------------------------------------------------------------------
void svtkCellArray::SetData(
  svtkAOSDataArrayTemplate<long long>* offsets, svtkAOSDataArrayTemplate<long long>* connectivity)
{
#if SVTK_SIZEOF_LONG_LONG == 4
  svtkNew<svtkTypeInt32Array> o;
  svtkNew<svtkTypeInt32Array> c;
  o->ShallowCopy(offsets);
  c->ShallowCopy(connectivity);
  this->SetData(o, c);
#elif SVTK_SIZEOF_LONG_LONG == 8
  svtkNew<svtkTypeInt64Array> o;
  svtkNew<svtkTypeInt64Array> c;
  o->ShallowCopy(offsets);
  c->ShallowCopy(connectivity);
  this->SetData(o, c);
#else
  svtkErrorMacro("`long long` type is neither 32 nor 64 bits.");
#endif
}

namespace
{

struct SetDataGenericImpl
{
  svtkCellArray* CellArray;
  svtkDataArray* ConnDA;
  bool ArraysMatch;

  template <typename ArrayT>
  void operator()(ArrayT* offsets)
  {
    ArrayT* conn = svtkArrayDownCast<ArrayT>(this->ConnDA);
    if (!conn)
    {
      this->ArraysMatch = false;
      return;
    }
    this->ArraysMatch = true;

    this->CellArray->SetData(offsets, conn);
  }
};

} // end anon namespace

//----------------------------------------------------------------------------
bool svtkCellArray::SetData(svtkDataArray* offsets, svtkDataArray* connectivity)
{
  SetDataGenericImpl worker{ this, connectivity, false };
  using SupportedArrays = svtkCellArray::InputArrayList;
  using Dispatch = svtkArrayDispatch::DispatchByArray<SupportedArrays>;
  if (!Dispatch::Execute(offsets, worker))
  {
    svtkErrorMacro("Invalid array types passed to SetData: "
      << "offsets=" << offsets->GetClassName() << ", "
      << "connectivity=" << connectivity->GetClassName());
    return false;
  }

  if (!worker.ArraysMatch)
  {
    svtkErrorMacro("Offsets and Connectivity arrays must have the same type.");
    return false;
  }

  return true;
}

//----------------------------------------------------------------------------
void svtkCellArray::Use32BitStorage()
{
  if (!this->Storage.Is64Bit())
  {
    this->Initialize();
    return;
  }
  this->Storage.Use32BitStorage();
}

//----------------------------------------------------------------------------
void svtkCellArray::Use64BitStorage()
{
  if (this->Storage.Is64Bit())
  {
    this->Initialize();
    return;
  }
  this->Storage.Use64BitStorage();
}

//----------------------------------------------------------------------------
void svtkCellArray::UseDefaultStorage()
{
#ifdef SVTK_USE_64BIT_IDS
  this->Use64BitStorage();
#else  // SVTK_USE_64BIT_IDS
  this->Use32BitStorage();
#endif // SVTK_USE_64BIT_IDS
}

//----------------------------------------------------------------------------
bool svtkCellArray::CanConvertTo32BitStorage() const
{
  if (!this->Storage.Is64Bit())
  {
    return true;
  }
  return this->Visit(CanConvert<ArrayType32::ValueType>{});
}

//----------------------------------------------------------------------------
bool svtkCellArray::CanConvertTo64BitStorage() const
{
  return true;
}

//----------------------------------------------------------------------------
bool svtkCellArray::CanConvertToDefaultStorage() const
{
#ifdef SVTK_USE_64BIT_IDS
  return this->CanConvertTo64BitStorage();
#else  // SVTK_USE_64BIT_IDS
  return this->CanConvertTo32BitStorage();
#endif // SVTK_USE_64BIT_IDS
}

//----------------------------------------------------------------------------
bool svtkCellArray::ConvertTo32BitStorage()
{
  if (!this->IsStorage64Bit())
  {
    return true;
  }
  svtkNew<ArrayType32> offsets;
  svtkNew<ArrayType32> conn;
  if (!this->Visit(ExtractAndInitialize{}, offsets.Get(), conn.Get()))
  {
    return false;
  }

  this->SetData(offsets, conn);
  return true;
}

//----------------------------------------------------------------------------
bool svtkCellArray::ConvertTo64BitStorage()
{
  if (this->IsStorage64Bit())
  {
    return true;
  }
  svtkNew<ArrayType64> offsets;
  svtkNew<ArrayType64> conn;
  if (!this->Visit(ExtractAndInitialize{}, offsets.Get(), conn.Get()))
  {
    return false;
  }

  this->SetData(offsets, conn);
  return true;
}

//----------------------------------------------------------------------------
bool svtkCellArray::ConvertToDefaultStorage()
{
#ifdef SVTK_USE_64BIT_IDS
  return this->ConvertTo64BitStorage();
#else  // SVTK_USE_64BIT_IDS
  return this->ConvertTo32BitStorage();
#endif // SVTK_USE_64BIT_IDS
}

//----------------------------------------------------------------------------
bool svtkCellArray::ConvertToSmallestStorage()
{
  if (this->IsStorage64Bit() && this->CanConvertTo32BitStorage())
  {
    return this->ConvertTo32BitStorage();
  }
  // Already at the smallest possible.
  return true;
}

//----------------------------------------------------------------------------
bool svtkCellArray::AllocateExact(svtkIdType numCells, svtkIdType connectivitySize)
{
  return this->Visit(AllocateExactImpl{}, numCells, connectivitySize);
}

//----------------------------------------------------------------------------
bool svtkCellArray::ResizeExact(svtkIdType numCells, svtkIdType connectivitySize)
{
  return this->Visit(ResizeExactImpl{}, numCells, connectivitySize);
}

//----------------------------------------------------------------------------
// Returns the size of the largest cell. The size is the number of points
// defining the cell.
int svtkCellArray::GetMaxCellSize()
{
  FindMaxCell finder{ this };

  // Grain size puts an even number of pages into each instance.
  svtkSMPTools::For(0, this->GetNumberOfCells(), finder);

  return static_cast<int>(finder.Result);
}

//----------------------------------------------------------------------------
unsigned long svtkCellArray::GetActualMemorySize() const
{
  return this->Visit(GetActualMemorySizeImpl{});
}

//----------------------------------------------------------------------------
void svtkCellArray::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "StorageIs64Bit: " << this->Storage.Is64Bit() << "\n";

  PrintSelfImpl functor;
  this->Visit(functor, os, indent);
}

//----------------------------------------------------------------------------
void svtkCellArray::PrintDebug(std::ostream& os)
{
  this->Print(os);
  this->Visit(PrintDebugImpl{}, os);
}

//----------------------------------------------------------------------------
svtkIdType svtkCellArray::GetTraversalCellId()
{
  return this->TraversalCellId;
}

//----------------------------------------------------------------------------
void svtkCellArray::SetTraversalCellId(svtkIdType cellId)
{
  this->TraversalCellId = cellId;
}

//----------------------------------------------------------------------------
void svtkCellArray::ReverseCellAtId(svtkIdType cellId)
{
  this->Visit(ReverseCellAtIdImpl{}, cellId);
}

//----------------------------------------------------------------------------
void svtkCellArray::ReplaceCellAtId(svtkIdType cellId, svtkIdList* list)
{
  this->Visit(ReplaceCellAtIdImpl{}, cellId, list->GetNumberOfIds(), list->GetPointer(0));
}

//----------------------------------------------------------------------------
void svtkCellArray::ReplaceCellAtId(
  svtkIdType cellId, svtkIdType cellSize, const svtkIdType cellPoints[])
{
  this->Visit(ReplaceCellAtIdImpl{}, cellId, cellSize, cellPoints);
}

//----------------------------------------------------------------------------
void svtkCellArray::ExportLegacyFormat(svtkIdTypeArray* data)
{
  data->Allocate(this->Visit(GetLegacyDataSizeImpl{}));

  auto it = svtk::TakeSmartPointer(this->NewIterator());

  svtkIdType cellSize;
  const svtkIdType* cellPoints;
  for (it->GoToFirstCell(); !it->IsDoneWithTraversal(); it->GoToNextCell())
  {
    it->GetCurrentCell(cellSize, cellPoints);
    data->InsertNextValue(cellSize);
    for (svtkIdType i = 0; i < cellSize; ++i)
    {
      data->InsertNextValue(cellPoints[i]);
    }
  }
}

//----------------------------------------------------------------------------
void svtkCellArray::ImportLegacyFormat(svtkIdTypeArray* data)
{
  this->ImportLegacyFormat(data->GetPointer(0), data->GetNumberOfValues());
}

//----------------------------------------------------------------------------
void svtkCellArray::ImportLegacyFormat(const svtkIdType* data, svtkIdType len)
{
  this->Reset();
  this->AppendLegacyFormat(data, len, 0);
}

//----------------------------------------------------------------------------
void svtkCellArray::AppendLegacyFormat(svtkIdTypeArray* data, svtkIdType ptOffset)
{
  this->AppendLegacyFormat(data->GetPointer(0), data->GetNumberOfValues(), ptOffset);
}

//----------------------------------------------------------------------------
void svtkCellArray::AppendLegacyFormat(const svtkIdType* data, svtkIdType len, svtkIdType ptOffset)
{
  this->Visit(AppendLegacyFormatImpl{}, data, len, ptOffset);
}

//----------------------------------------------------------------------------
void svtkCellArray::Squeeze()
{
  this->Visit(SqueezeImpl{});

  // Just delete the legacy buffer.
  this->LegacyData->Initialize();
}

//----------------------------------------------------------------------------
bool svtkCellArray::IsValid()
{
  return this->Visit(IsValidImpl{});
}

//----------------------------------------------------------------------------
svtkIdType svtkCellArray::IsHomogeneous()
{
  return this->Visit(IsHomogeneousImpl{});
}
