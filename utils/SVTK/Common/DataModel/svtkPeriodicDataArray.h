/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPeriodicDataArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

    This software is distributed WITHOUT ANY WARRANTY; without even
    the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
    PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkPeriodicDataArray
 * @brief   Map native an Array into an angulat
 * periodic array
 *
 *
 * Map an array into a periodic array. Data from the original array are
 * rotated (on the fly) by the specified angle along the specified axis
 * around the specified point. Lookup is not implemented.
 * Creating the array is virtually free, accessing a tuple require some
 * computation.
 */

#ifndef svtkPeriodicDataArray_h
#define svtkPeriodicDataArray_h

#include "svtkAOSDataArrayTemplate.h" // Template
#include "svtkGenericDataArray.h"     // Parent

template <class Scalar>
class svtkPeriodicDataArray : public svtkGenericDataArray<svtkPeriodicDataArray<Scalar>, Scalar>
{
  typedef svtkGenericDataArray<svtkPeriodicDataArray<Scalar>, Scalar> GenericBase;

public:
  svtkTemplateTypeMacro(svtkPeriodicDataArray<Scalar>, GenericBase);
  typedef typename Superclass::ValueType ValueType;

  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Initialize the mapped array with the original input data array.
   */
  void InitializeArray(svtkAOSDataArrayTemplate<Scalar>* inputData);

  /**
   * Initialize array with zero values
   */
  void Initialize() override;

  /**
   * Copy tuples values, selected by ptIds into provided array
   */
  void GetTuples(svtkIdList* ptIds, svtkAbstractArray* output) override;

  /**
   * Copy tuples from id p1 to id p2 included into provided array
   */
  void GetTuples(svtkIdType p1, svtkIdType p2, svtkAbstractArray* output) override;

  /**
   * No effect
   */
  void Squeeze() override;

  /**
   * Not implemented
   */
  SVTK_NEWINSTANCE svtkArrayIterator* NewIterator() override;

  /**
   * Not implemented
   */
  svtkIdType LookupValue(svtkVariant value) override;

  /**
   * Not implemented
   */
  void LookupValue(svtkVariant value, svtkIdList* ids) override;

  /**
   * Not implemented
   */
  svtkVariant GetVariantValue(svtkIdType idx) override;

  /**
   * Not implemented
   */
  void ClearLookup() override;

  /**
   * Return tuple at location i.
   * Pointer valid until next call to this object
   */
  double* GetTuple(svtkIdType i) override;

  /**
   * Copy tuple at location i into user provided array
   */
  void GetTuple(svtkIdType i, double* tuple) override;

  /**
   * Not implemented
   */
  svtkIdType LookupTypedValue(Scalar value) override;

  /**
   * Not implemented
   */
  void LookupTypedValue(Scalar value, svtkIdList* ids) override;

  /**
   * Get value at index idx.
   * Warning, it internally call GetTypedTuple,
   * so it is an inneficcient way if reading all data
   */
  ValueType GetValue(svtkIdType idx) const;

  /**
   * Get value at index idx as reference.
   * Warning, it internally call GetTypedTuple,
   * so it is an inneficcient way if reading all data
   */
  ValueType& GetValueReference(svtkIdType idx);

  /**
   * Copy tuple value at location idx into provided array
   */
  void GetTypedTuple(svtkIdType idx, Scalar* t) const;

  /**
   * Return the requested component of the specified tuple.
   * Warning, this internally calls GetTypedTuple, so it is an inefficient way
   * of reading all data.
   */
  ValueType GetTypedComponent(svtkIdType tupleIdx, int compIdx) const;

  /**
   * Return the memory in kilobytes consumed by this data array.
   */
  unsigned long GetActualMemorySize() const override;

  /**
   * Read only container, not supported.
   */
  svtkTypeBool Allocate(svtkIdType sz, svtkIdType ext) override;

  /**
   * Read only container, not supported.
   */
  svtkTypeBool Resize(svtkIdType numTuples) override;

  /**
   * Read only container, not supported.
   */
  void SetNumberOfTuples(svtkIdType number) override;

  /**
   * Read only container, not supported.
   */
  void SetTuple(svtkIdType i, svtkIdType j, svtkAbstractArray* source) override;

  /**
   * Read only container, not supported.
   */
  void SetTuple(svtkIdType i, const float* source) override;

  /**
   * Read only container, not supported.
   */
  void SetTuple(svtkIdType i, const double* source) override;

  /**
   * Read only container, not supported.
   */
  void InsertTuple(svtkIdType i, svtkIdType j, svtkAbstractArray* source) override;

  /**
   * Read only container, not supported.
   */
  void InsertTuple(svtkIdType i, const float* source) override;

  /**
   * Read only container, not supported.
   */
  void InsertTuple(svtkIdType i, const double* source) override;

  /**
   * Read only container, not supported.
   */
  void InsertTuples(svtkIdList* dstIds, svtkIdList* srcIds, svtkAbstractArray* source) override;

  /**
   * Read only container, not supported.
   */
  void InsertTuples(
    svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkAbstractArray* source) override;

  /**
   * Read only container, error.
   */
  svtkIdType InsertNextTuple(svtkIdType j, svtkAbstractArray* source) override;

  /**
   * Read only container, not supported.
   */
  svtkIdType InsertNextTuple(const float* source) override;

  /**
   * Read only container, not supported.
   */
  svtkIdType InsertNextTuple(const double* source) override;

  /**
   * Read only container, not supported.
   */
  void DeepCopy(svtkAbstractArray* aa) override;

  /**
   * Read only container, not supported.
   */
  void DeepCopy(svtkDataArray* da) override;

  /**
   * Read only container, not supported.
   */
  void InterpolateTuple(
    svtkIdType i, svtkIdList* ptIndices, svtkAbstractArray* source, double* weights) override;

  /**
   * Read only container, not supported.
   */
  void InterpolateTuple(svtkIdType i, svtkIdType id1, svtkAbstractArray* source1, svtkIdType id2,
    svtkAbstractArray* source2, double t) override;

  /**
   * Read only container, not supported.
   */
  void SetVariantValue(svtkIdType idx, svtkVariant value) override;

  /**
   * Read only container, not supported.
   */
  void InsertVariantValue(svtkIdType idx, svtkVariant value) override;

  /**
   * Read only container, not supported.
   */
  void RemoveTuple(svtkIdType id) override;

  /**
   * Read only container, not supported.
   */
  void RemoveFirstTuple() override;

  /**
   * Read only container, not supported.
   */
  void RemoveLastTuple() override;

  /**
   * Read only container, not supported.
   */
  void SetTypedTuple(svtkIdType i, const Scalar* t);

  /**
   * Read only container, not supported.
   */
  void SetTypedComponent(svtkIdType t, int c, Scalar v);

  /**
   * Read only container, not supported.
   */
  void InsertTypedTuple(svtkIdType i, const Scalar* t);

  /**
   * Read only container, not supported.
   */
  svtkIdType InsertNextTypedTuple(const Scalar* t);

  /**
   * Read only container, not supported.
   */
  void SetValue(svtkIdType idx, Scalar value);

  /**
   * Read only container, not supported.
   */
  svtkIdType InsertNextValue(Scalar v);

  /**
   * Read only container, not supported.
   */
  void InsertValue(svtkIdType idx, Scalar v);

  //@{
  /**
   * Set/Get normalize flag. Default: false
   */
  svtkSetMacro(Normalize, bool);
  svtkGetMacro(Normalize, bool);
  //@}

protected:
  svtkPeriodicDataArray();
  ~svtkPeriodicDataArray() override;

  //@{
  /**
   * Read only container, not supported.
   */
  bool AllocateTuples(svtkIdType numTuples);
  bool ReallocateTuples(svtkIdType numTuples);
  //@}

  /**
   * Transform the provided tuple
   */
  virtual void Transform(Scalar* tuple) const = 0;

  /**
   * Get the transformed range by components
   */
  bool ComputeScalarRange(double* range) override;

  /**
   * Get the transformed range on all components
   */
  bool ComputeVectorRange(double range[2]) override;

  /**
   * Update the transformed periodic range
   */
  virtual void ComputePeriodicRange();

  /**
   * Set the invalid range flag to false
   */
  void InvalidateRange();

  bool Normalize; // If transformed vector must be normalized

private:
  svtkPeriodicDataArray(const svtkPeriodicDataArray&) = delete;
  void operator=(const svtkPeriodicDataArray&) = delete;

  friend class svtkGenericDataArray<svtkPeriodicDataArray<Scalar>, Scalar>;

  Scalar* TempScalarArray;               // Temporary array used by GetTypedTuple methods
  double* TempDoubleArray;               // Temporary array used by GetTuple vethods
  svtkIdType TempTupleIdx;                // Location of currently stored Temp Tuple to use as cache
  svtkAOSDataArrayTemplate<Scalar>* Data; // Original data

  bool InvalidRange;
  double PeriodicRange[6]; // Transformed periodic range
};

#include "svtkPeriodicDataArray.txx"

#endif // svtkPeriodicDataArray_h
// SVTK-HeaderTest-Exclude: svtkPeriodicDataArray.h
