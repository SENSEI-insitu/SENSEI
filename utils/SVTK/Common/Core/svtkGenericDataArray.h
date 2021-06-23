/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericDataArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkGenericDataArray
 * @brief   Base interface for all typed svtkDataArray
 * subclasses.
 *
 *
 *
 * A more detailed description of this class and related tools can be found
 * \ref SVTK-7-1-ArrayDispatch "here".
 *
 * The svtkGenericDataArray class provides a generic implementation of the
 * svtkDataArray API. It relies on subclasses providing access to data
 * via 8 "concept methods", which should be implemented as non-virtual
 * methods of the subclass. These methods are:
 *
 * - ValueType GetValue(svtkIdType valueIdx) const
 * - [public] void SetValue(svtkIdType valueIdx, ValueType value)
 * - [public] void GetTypedTuple(svtkIdType tupleIdx, ValueType* tuple) const
 * - [public] void SetTypedTuple(svtkIdType tupleIdx, const ValueType* tuple)
 * - [public] ValueType GetTypedComponent(svtkIdType tupleIdx, int compIdx) const
 * - [public] void SetTypedComponent(svtkIdType tupleIdx, int compIdx,
 *                                   ValueType value)
 * - [protected] bool AllocateTuples(svtkIdType numTuples)
 * - [protected] bool ReallocateTuples(svtkIdType numTuples)
 *
 * Note that these methods use the CRTP idiom, which provides static binding to
 * avoid virtual calls. This allows the compiler to optimize away layers of
 * indirection when these methods are used. Well-designed implementations
 * of these methods will reduce to raw memory accesses, providing efficient
 * performance comparable to working with the pointer data.
 *
 * See svtkAOSDataArrayTemplate and svtkSOADataArrayTemplate for example
 * implementations.
 *
 * In practice, code should not be written to use svtkGenericDataArray objects.
 * Doing so is rather unweildy due to the CRTP pattern requiring the derived
 * class be provided as a template argument. Instead, the svtkArrayDispatch
 * framework can be used to detect a svtkDataArray's implementation type and
 * instantiate appropriate templated worker code.
 *
 * svtkArrayDispatch is also intended to replace code that currently relies on
 * the encapsulation-breaking GetVoidPointer method. Not all subclasses of
 * svtkDataArray use the memory layout assumed by GetVoidPointer; calling this
 * method on, e.g. a svtkSOADataArrayTemplate will trigger a deep copy of the
 * array data into an AOS buffer. This is very inefficient and should be
 * avoided.
 *
 * @sa
 * svtkArrayDispatcher svtkDataArrayRange
 */

#ifndef svtkGenericDataArray_h
#define svtkGenericDataArray_h

#include "svtkDataArray.h"

#include "svtkConfigure.h"
#include "svtkGenericDataArrayLookupHelper.h"
#include "svtkSmartPointer.h"
#include "svtkTypeTraits.h"

#include <cassert>

template <class DerivedT, class ValueTypeT>
class svtkGenericDataArray : public svtkDataArray
{
  typedef svtkGenericDataArray<DerivedT, ValueTypeT> SelfType;

public:
  typedef ValueTypeT ValueType;
  svtkTemplateTypeMacro(SelfType, svtkDataArray);

  /**
   * Compile time access to the SVTK type identifier.
   */
  enum
  {
    SVTK_DATA_TYPE = svtkTypeTraits<ValueType>::SVTK_TYPE_ID
  };

  /// @defgroup svtkGDAConceptMethods svtkGenericDataArray Concept Methods
  /// These signatures must be reimplemented in subclasses as public,
  /// non-virtual methods. Ideally, they should be inlined and as efficient as
  /// possible to ensure the best performance possible.

  /**
   * Get the value at @a valueIdx. @a valueIdx assumes AOS ordering.
   * @note GetTypedComponent is preferred over this method. It is faster for
   * SOA arrays, and shows equivalent performance for AOS arrays when
   * NumberOfComponents is known to the compiler (See svtkAssume.h).
   * @ingroup svtkGDAConceptMethods
   */
  inline ValueType GetValue(svtkIdType valueIdx) const
  {
    return static_cast<const DerivedT*>(this)->GetValue(valueIdx);
  }

  /**
   * Set the value at @a valueIdx to @a value. @a valueIdx assumes AOS ordering.
   * @note SetTypedComponent is preferred over this method. It is faster for
   * SOA arrays, and shows equivalent performance for AOS arrays when
   * NumberOfComponents is known to the compiler (See svtkAssume.h).
   * @ingroup svtkGDAConceptMethods
   */
  void SetValue(svtkIdType valueIdx, ValueType value)
    SVTK_EXPECTS(0 <= valueIdx && valueIdx < GetNumberOfValues())
  {
    static_cast<DerivedT*>(this)->SetValue(valueIdx, value);
  }

  /**
   * Copy the tuple at @a tupleIdx into @a tuple.
   * @note GetTypedComponent is preferred over this method. The overhead of
   * copying the tuple is significant compared to the more performant
   * component-wise access methods, which typically optimize to raw memory
   * access.
   * @ingroup svtkGDAConceptMethods
   */
  void GetTypedTuple(svtkIdType tupleIdx, ValueType* tuple) const
    SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples())
  {
    static_cast<const DerivedT*>(this)->GetTypedTuple(tupleIdx, tuple);
  }

  /**
   * Set this array's tuple at @a tupleIdx to the values in @a tuple.
   * @note SetTypedComponent is preferred over this method. The overhead of
   * copying the tuple is significant compared to the more performant
   * component-wise access methods, which typically optimize to raw memory
   * access.
   * @ingroup svtkGDAConceptMethods
   */
  void SetTypedTuple(svtkIdType tupleIdx, const ValueType* tuple)
    SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples())
  {
    static_cast<DerivedT*>(this)->SetTypedTuple(tupleIdx, tuple);
  }

  /**
   * Get component @a compIdx of the tuple at @a tupleIdx. This is typically
   * the fastest way to access array data.
   * @ingroup svtkGDAConceptMethods
   */
  ValueType GetTypedComponent(svtkIdType tupleIdx, int compIdx) const SVTK_EXPECTS(0 <= tupleIdx &&
    tupleIdx < GetNumberOfTuples()) SVTK_EXPECTS(0 <= compIdx && compIdx < GetNumberOfComponents())
  {
    return static_cast<const DerivedT*>(this)->GetTypedComponent(tupleIdx, compIdx);
  }

  /**
   * Set component @a compIdx of the tuple at @a tupleIdx to @a value. This is
   * typically the fastest way to set array data.
   * @ingroup svtkGDAConceptMethods
   */
  void SetTypedComponent(svtkIdType tupleIdx, int compIdx, ValueType value)
    SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples())
      SVTK_EXPECTS(0 <= compIdx && compIdx < GetNumberOfComponents())
  {
    static_cast<DerivedT*>(this)->SetTypedComponent(tupleIdx, compIdx, value);
  }

  //@{
  /**
   * Default implementation raises a runtime error. If subclasses keep on
   * supporting this API, they should override this method.
   */
  void* GetVoidPointer(svtkIdType valueIdx) override;
  ValueType* GetPointer(svtkIdType valueIdx);
  void SetVoidArray(void*, svtkIdType, int) override;
  void SetVoidArray(void*, svtkIdType, int, int) override;
  void SetArrayFreeFunction(void (*callback)(void*)) override;
  void* WriteVoidPointer(svtkIdType valueIdx, svtkIdType numValues) override;
  ValueType* WritePointer(svtkIdType valueIdx, svtkIdType numValues);
  //@}

  /**
   * Removes a tuple at the given index. Default implementation
   * iterates over tuples to move elements. Subclasses are
   * encouraged to reimplemented this method to support faster implementations,
   * if needed.
   */
  void RemoveTuple(svtkIdType tupleIdx) override;

  /**
   * Insert data at the end of the array. Return its location in the array.
   */
  svtkIdType InsertNextValue(ValueType value);

  /**
   * Insert data at a specified position in the array.
   */
  void InsertValue(svtkIdType valueIdx, ValueType value);

  /**
   * Insert (memory allocation performed) the tuple t at tupleIdx.
   */
  void InsertTypedTuple(svtkIdType tupleIdx, const ValueType* t);

  /**
   * Insert (memory allocation performed) the tuple onto the end of the array.
   */
  svtkIdType InsertNextTypedTuple(const ValueType* t);

  /**
   * Insert (memory allocation performed) the value at the specified tuple and
   * component location.
   */
  void InsertTypedComponent(svtkIdType tupleIdx, int compIdx, ValueType val);

  //@{
  /**
   * Get the range of array values for the given component in the
   * native data type.
   */
  void GetValueRange(ValueType range[2], int comp);
  ValueType* GetValueRange(int comp) SVTK_SIZEHINT(2);
  //@}

  /**
   * Get the range of array values for the 0th component in the
   * native data type.
   */
  ValueType* GetValueRange() SVTK_SIZEHINT(2) { return this->GetValueRange(0); }
  void GetValueRange(ValueType range[2]) { this->GetValueRange(range, 0); }

  /**
   * These methods are analogous to the GetValueRange methods, except that the
   * only consider finite values.
   * @{
   */
  void GetFiniteValueRange(ValueType range[2], int comp);
  ValueType* GetFiniteValueRange(int comp) SVTK_SIZEHINT(2);
  ValueType* GetFiniteValueRange() SVTK_SIZEHINT(2) { return this->GetFiniteValueRange(0); }
  void GetFiniteValueRange(ValueType range[2]) { this->GetFiniteValueRange(range, 0); }
  /**@}*/

  /**
   * Return the capacity in typeof T units of the current array.
   * TODO Leftover from svtkDataArrayTemplate, redundant with GetSize. Deprecate?
   */
  svtkIdType Capacity() { return this->Size; }

  /**
   * Set component @a comp of all tuples to @a value.
   */
  virtual void FillTypedComponent(int compIdx, ValueType value);

  /**
   * Set all the values in array to @a value.
   */
  virtual void FillValue(ValueType value);

  int GetDataType() const override;
  int GetDataTypeSize() const override;
  bool HasStandardMemoryLayout() const override;
  svtkTypeBool Allocate(svtkIdType size, svtkIdType ext = 1000) override;
  svtkTypeBool Resize(svtkIdType numTuples) override;
  void SetNumberOfComponents(int num) override;
  void SetNumberOfTuples(svtkIdType number) override;
  void Initialize() override;
  void Squeeze() override;
  void SetTuple(svtkIdType dstTupleIdx, svtkIdType srcTupleIdx, svtkAbstractArray* source) override;
  // MSVC doesn't like 'using' here (error C2487). Just forward instead:
  // using Superclass::SetTuple;
  void SetTuple(svtkIdType tupleIdx, const float* tuple) override
  {
    this->Superclass::SetTuple(tupleIdx, tuple);
  }
  void SetTuple(svtkIdType tupleIdx, const double* tuple) override
  {
    this->Superclass::SetTuple(tupleIdx, tuple);
  }

  void InsertTuples(svtkIdList* dstIds, svtkIdList* srcIds, svtkAbstractArray* source) override;
  // MSVC doesn't like 'using' here (error C2487). Just forward instead:
  // using Superclass::InsertTuples;
  void InsertTuples(
    svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkAbstractArray* source) override
  {
    this->Superclass::InsertTuples(dstStart, n, srcStart, source);
  }

  void InsertTuple(svtkIdType dstTupleIdx, svtkIdType srcTupleIdx, svtkAbstractArray* source) override;
  void InsertTuple(svtkIdType tupleIdx, const float* source) override;
  void InsertTuple(svtkIdType tupleIdx, const double* source) override;
  void InsertComponent(svtkIdType tupleIdx, int compIdx, double value) override;
  svtkIdType InsertNextTuple(svtkIdType srcTupleIdx, svtkAbstractArray* source) override;
  svtkIdType InsertNextTuple(const float* tuple) override;
  svtkIdType InsertNextTuple(const double* tuple) override;
  void GetTuples(svtkIdList* tupleIds, svtkAbstractArray* output) override;
  void GetTuples(svtkIdType p1, svtkIdType p2, svtkAbstractArray* output) override;
  double* GetTuple(svtkIdType tupleIdx) override;
  void GetTuple(svtkIdType tupleIdx, double* tuple) override;
  void InterpolateTuple(svtkIdType dstTupleIdx, svtkIdList* ptIndices, svtkAbstractArray* source,
    double* weights) override;
  void InterpolateTuple(svtkIdType dstTupleIdx, svtkIdType srcTupleIdx1, svtkAbstractArray* source1,
    svtkIdType srcTupleIdx2, svtkAbstractArray* source2, double t) override;
  void SetComponent(svtkIdType tupleIdx, int compIdx, double value) override;
  double GetComponent(svtkIdType tupleIdx, int compIdx) override;
  void SetVariantValue(svtkIdType valueIdx, svtkVariant value) override;
  svtkVariant GetVariantValue(svtkIdType valueIdx) override;
  void InsertVariantValue(svtkIdType valueIdx, svtkVariant value) override;
  svtkIdType LookupValue(svtkVariant value) override;
  virtual svtkIdType LookupTypedValue(ValueType value);
  void LookupValue(svtkVariant value, svtkIdList* valueIds) override;
  virtual void LookupTypedValue(ValueType value, svtkIdList* valueIds);
  void ClearLookup() override;
  void DataChanged() override;
  void FillComponent(int compIdx, double value) override;
  SVTK_NEWINSTANCE svtkArrayIterator* NewIterator() override;

protected:
  svtkGenericDataArray();
  ~svtkGenericDataArray() override;

  /**
   * Allocate space for numTuples. Old data is not preserved. If numTuples == 0,
   * all data is freed.
   * @ingroup svtkGDAConceptMethods
   */
  inline bool AllocateTuples(svtkIdType numTuples)
  {
    return static_cast<DerivedT*>(this)->AllocateTuples(numTuples);
  }

  /**
   * Allocate space for numTuples. Old data is preserved. If numTuples == 0,
   * all data is freed.
   * @ingroup svtkGDAConceptMethods
   */
  inline bool ReallocateTuples(svtkIdType numTuples)
  {
    return static_cast<DerivedT*>(this)->ReallocateTuples(numTuples);
  }

  // This method resizes the array if needed so that the given tuple index is
  // valid/accessible.
  bool EnsureAccessToTuple(svtkIdType tupleIdx);

  /**
   * Compute the range for a specific component. If comp is set -1
   * then L2 norm is computed on all components. Call ClearRange
   * to force a recomputation if it is needed. The range is copied
   * to the range argument.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  void ComputeValueRange(ValueType range[2], int comp);

  /**
   * Compute the range for a specific component. If comp is set -1
   * then L2 norm is computed on all components. Call ClearRange
   * to force a recomputation if it is needed. The range is copied
   * to the range argument.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  void ComputeFiniteValueRange(ValueType range[2], int comp);

  /**
   * Computes the range for each component of an array, the length
   * of \a ranges must be two times the number of components.
   * Returns true if the range was computed. Will return false
   * if you try to compute the range of an array of length zero.
   */
  bool ComputeScalarValueRange(ValueType* ranges);

  /**
   * Returns true if the range was computed. Will return false
   * if you try to compute the range of an array of length zero.
   */
  bool ComputeVectorValueRange(ValueType range[2]);

  /**
   * Computes the range for each component of an array, the length
   * of \a ranges must be two times the number of components.
   * Returns true if the range was computed. Will return false
   * if you try to compute the range of an array of length zero.
   */
  bool ComputeFiniteScalarValueRange(ValueType* ranges);

  /**
   * Returns true if the range was computed. Will return false
   * if you try to compute the range of an array of length zero.
   */
  bool ComputeFiniteVectorValueRange(ValueType range[2]);

  std::vector<double> LegacyTuple;
  std::vector<ValueType> LegacyValueRange;
  std::vector<ValueType> LegacyValueRangeFull;

  svtkGenericDataArrayLookupHelper<SelfType> Lookup;

private:
  svtkGenericDataArray(const svtkGenericDataArray&) = delete;
  void operator=(const svtkGenericDataArray&) = delete;
};

// these predeclarations are needed before the .txx include for MinGW
namespace svtkDataArrayPrivate
{
template <typename A, typename R, typename T>
bool DoComputeScalarRange(A*, R*, T);
template <typename A, typename R>
bool DoComputeVectorRange(A*, R[2], AllValues);
template <typename A, typename R>
bool DoComputeVectorRange(A*, R[2], FiniteValues);
} // namespace svtkDataArrayPrivate

#include "svtkGenericDataArray.txx"

// Adds an implementation of NewInstanceInternal() that returns an AoS
// (unmapped) SVTK array, if possible. This allows the pipeline to copy and
// propagate the array when the array data is not modifiable. Use this in
// combination with svtkAbstractTypeMacro or svtkAbstractTemplateTypeMacro
// (instead of svtkTypeMacro) to avoid adding the default NewInstance
// implementation.
#define svtkAOSArrayNewInstanceMacro(thisClass)                                                     \
protected:                                                                                         \
  svtkObjectBase* NewInstanceInternal() const override                                              \
  {                                                                                                \
    if (svtkDataArray* da = svtkDataArray::CreateDataArray(thisClass::SVTK_DATA_TYPE))                \
    {                                                                                              \
      return da;                                                                                   \
    }                                                                                              \
    return thisClass::New();                                                                       \
  }                                                                                                \
                                                                                                   \
public:

#endif

// This portion must be OUTSIDE the include blockers. This is used to tell
// libraries other than svtkCommonCore that instantiations of
// the GetValueRange lookups can be found externally. This prevents each library
// from instantiating these on their own.
// Additionally it helps hide implementation details that pull in system
// headers.
// We only provide these specializations for the 64-bit integer types, since
// other types can reuse the double-precision mechanism in
// svtkDataArray::GetRange without losing precision.
#ifdef SVTK_GDA_VALUERANGE_INSTANTIATING

// Forward declare necessary stuffs:
template <typename ValueType>
class svtkAOSDataArrayTemplate;
template <typename ValueType>
class svtkSOADataArrayTemplate;

#ifdef SVTK_USE_SCALED_SOA_ARRAYS
template <typename ValueType>
class svtkScaledSOADataArrayTemplate;
#endif

#define SVTK_INSTANTIATE_VALUERANGE_ARRAYTYPE(ArrayType, ValueType)                                 \
  template SVTKCOMMONCORE_EXPORT bool DoComputeScalarRange(                                         \
    ArrayType*, ValueType*, svtkDataArrayPrivate::AllValues);                                       \
  template SVTKCOMMONCORE_EXPORT bool DoComputeScalarRange(                                         \
    ArrayType*, ValueType*, svtkDataArrayPrivate::FiniteValues);                                    \
  template SVTKCOMMONCORE_EXPORT bool DoComputeVectorRange(                                         \
    ArrayType*, ValueType[2], svtkDataArrayPrivate::AllValues);                                     \
  template SVTKCOMMONCORE_EXPORT bool DoComputeVectorRange(                                         \
    ArrayType*, ValueType[2], svtkDataArrayPrivate::FiniteValues);

#ifdef SVTK_USE_SCALED_SOA_ARRAYS

#define SVTK_INSTANTIATE_VALUERANGE_VALUETYPE(ValueType)                                            \
  SVTK_INSTANTIATE_VALUERANGE_ARRAYTYPE(svtkAOSDataArrayTemplate<ValueType>, ValueType)              \
  SVTK_INSTANTIATE_VALUERANGE_ARRAYTYPE(svtkSOADataArrayTemplate<ValueType>, ValueType)              \
  SVTK_INSTANTIATE_VALUERANGE_ARRAYTYPE(svtkScaledSOADataArrayTemplate<ValueType>, ValueType)

#else // SVTK_USE_SCALED_SOA_ARRAYS

#define SVTK_INSTANTIATE_VALUERANGE_VALUETYPE(ValueType)                                            \
  SVTK_INSTANTIATE_VALUERANGE_ARRAYTYPE(svtkAOSDataArrayTemplate<ValueType>, ValueType)              \
  SVTK_INSTANTIATE_VALUERANGE_ARRAYTYPE(svtkSOADataArrayTemplate<ValueType>, ValueType)

#endif

#elif defined(SVTK_USE_EXTERN_TEMPLATE) // SVTK_GDA_VALUERANGE_INSTANTIATING

#ifndef SVTK_GDA_TEMPLATE_EXTERN
#define SVTK_GDA_TEMPLATE_EXTERN
#ifdef _MSC_VER
#pragma warning(push)
// The following is needed when the following is declared
// dllexport and is used from another class in svtkCommonCore
#pragma warning(disable : 4910) // extern and dllexport incompatible
#endif

// Forward declare necessary stuffs:
template <typename ValueType>
class svtkAOSDataArrayTemplate;
template <typename ValueType>
class svtkSOADataArrayTemplate;

#ifdef SVTK_USE_SCALED_SOA_ARRAYS
template <typename ValueType>
class svtkScaledSOADataArrayTemplate;
#endif

namespace svtkDataArrayPrivate
{
template <typename A, typename R, typename T>
bool DoComputeScalarRange(A*, R*, T);
template <typename A, typename R>
bool DoComputeVectorRange(A*, R[2], AllValues);
template <typename A, typename R>
bool DoComputeVectorRange(A*, R[2], FiniteValues);
} // namespace svtkDataArrayPrivate

#define SVTK_DECLARE_VALUERANGE_ARRAYTYPE(ArrayType, ValueType)                                     \
  extern template SVTKCOMMONCORE_EXPORT bool DoComputeScalarRange(                                  \
    ArrayType*, ValueType*, svtkDataArrayPrivate::AllValues);                                       \
  extern template SVTKCOMMONCORE_EXPORT bool DoComputeScalarRange(                                  \
    ArrayType*, ValueType*, svtkDataArrayPrivate::FiniteValues);                                    \
  extern template SVTKCOMMONCORE_EXPORT bool DoComputeVectorRange(                                  \
    ArrayType*, ValueType[2], svtkDataArrayPrivate::AllValues);                                     \
  extern template SVTKCOMMONCORE_EXPORT bool DoComputeVectorRange(                                  \
    ArrayType*, ValueType[2], svtkDataArrayPrivate::FiniteValues);

#ifdef SVTK_USE_SCALED_SOA_ARRAYS

#define SVTK_DECLARE_VALUERANGE_VALUETYPE(ValueType)                                                \
  SVTK_DECLARE_VALUERANGE_ARRAYTYPE(svtkAOSDataArrayTemplate<ValueType>, ValueType)                  \
  SVTK_DECLARE_VALUERANGE_ARRAYTYPE(svtkSOADataArrayTemplate<ValueType>, ValueType)                  \
  SVTK_DECLARE_VALUERANGE_ARRAYTYPE(svtkScaledSOADataArrayTemplate<ValueType>, ValueType)

#else // SVTK_USE_SCALED_SOA_ARRAYS

#define SVTK_DECLARE_VALUERANGE_VALUETYPE(ValueType)                                                \
  SVTK_DECLARE_VALUERANGE_ARRAYTYPE(svtkAOSDataArrayTemplate<ValueType>, ValueType)                  \
  SVTK_DECLARE_VALUERANGE_ARRAYTYPE(svtkSOADataArrayTemplate<ValueType>, ValueType)

#endif

namespace svtkDataArrayPrivate
{
SVTK_DECLARE_VALUERANGE_VALUETYPE(long)
SVTK_DECLARE_VALUERANGE_VALUETYPE(unsigned long)
SVTK_DECLARE_VALUERANGE_VALUETYPE(long long)
SVTK_DECLARE_VALUERANGE_VALUETYPE(unsigned long long)
SVTK_DECLARE_VALUERANGE_ARRAYTYPE(svtkDataArray, double)
} // namespace svtkDataArrayPrivate

#undef SVTK_DECLARE_VALUERANGE_ARRAYTYPE
#undef SVTK_DECLARE_VALUERANGE_VALUETYPE

#ifdef _MSC_VER
#pragma warning(pop)
#endif
#endif // SVTK_SOA_DATA_ARRAY_TEMPLATE_EXTERN

#endif // SVTK_GDA_VALUERANGE_INSTANTIATING

// SVTK-HeaderTest-Exclude: svtkGenericDataArray.h
