/*==============================================================================

  Program:   Visualization Toolkit
  Module:    svtkMappedDataArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

==============================================================================*/
/**
 * @class   svtkMappedDataArray
 * @brief   Map non-contiguous data structures into the
 * svtkDataArray API.
 *
 *
 * svtkMappedDataArray is a superclass for svtkDataArrays that do not use
 * the standard memory layout, and allows SVTK to interface with
 * simulation codes for in-situ analysis without repacking simulation data.
 *
 * svtkMappedDataArrayNewInstanceMacro is used by subclasses to implement
 * NewInstanceInternal such that a non-mapped svtkDataArray is returned by
 * NewInstance(). This prevents the mapped array type from propagating
 * through the pipeline.
 *
 * @attention
 * Subclasses that hold svtkIdType elements must also
 * reimplement `int GetDataType() const` (see Caveat in svtkTypedDataArray).
 */

#ifndef svtkMappedDataArray_h
#define svtkMappedDataArray_h

#include "svtkTypedDataArray.h"

template <class Scalar>
class svtkMappedDataArray : public svtkTypedDataArray<Scalar>
{
public:
  svtkTemplateTypeMacro(svtkMappedDataArray<Scalar>, svtkTypedDataArray<Scalar>);
  typedef typename Superclass::ValueType ValueType;

  /**
   * Perform a fast, safe cast from a svtkAbstractArray to a svtkMappedDataArray.
   * This method checks if:
   * - source->GetArrayType() is appropriate, and
   * - source->GetDataType() matches the Scalar template argument
   * if these conditions are met, the method performs a static_cast to return
   * source as a svtkMappedDataArray pointer. Otherwise, nullptr is returned.
   */
  static svtkMappedDataArray<Scalar>* FastDownCast(svtkAbstractArray* source);

  void PrintSelf(ostream& os, svtkIndent indent) override;

  // svtkAbstractArray virtual method that must be reimplemented.
  void DeepCopy(svtkAbstractArray* aa) override = 0;
  svtkVariant GetVariantValue(svtkIdType idx) override = 0;
  void SetVariantValue(svtkIdType idx, svtkVariant value) override = 0;
  void GetTuples(svtkIdList* ptIds, svtkAbstractArray* output) override = 0;
  void GetTuples(svtkIdType p1, svtkIdType p2, svtkAbstractArray* output) override = 0;
  void InterpolateTuple(
    svtkIdType i, svtkIdList* ptIndices, svtkAbstractArray* source, double* weights) override = 0;
  void InterpolateTuple(svtkIdType i, svtkIdType id1, svtkAbstractArray* source1, svtkIdType id2,
    svtkAbstractArray* source2, double t) override = 0;

  // svtkDataArray virtual method that must be reimplemented.
  void DeepCopy(svtkDataArray* da) override = 0;

  /**
   * Print an error and create an internal, long-lived temporary array. This
   * method should not be used on svtkMappedDataArray subclasses. See
   * svtkArrayDispatch for a better way.
   */
  void* GetVoidPointer(svtkIdType id) override;

  /**
   * Copy the internal data to the void pointer. The pointer is cast to this
   * array's Scalar type and svtkTypedDataArrayIterator is used to populate
   * the input array.
   */
  void ExportToVoidPointer(void* ptr) override;

  /**
   * Read the data from the internal temporary array (created by GetVoidPointer)
   * back into the mapped array. If GetVoidPointer has not been called (and the
   * internal array therefore does not exist), print an error and return. The
   * default implementation uses svtkTypedDataArrayIterator to extract the mapped
   * data.
   */
  void DataChanged() override;

  //@{
  /**
   * These methods don't make sense for mapped data array. Prints an error and
   * returns.
   */
  void SetVoidArray(void*, svtkIdType, int) override;
  void SetVoidArray(void*, svtkIdType, int, int) override;
  //@}

  //@{
  /**
   * Not implemented. Print error and return nullptr.
   */
  void* WriteVoidPointer(svtkIdType /*id*/, svtkIdType /*number*/) override
  {
    svtkErrorMacro(<< "WriteVoidPointer: Method not implemented.");
    return nullptr;
  }
  //@}

  /**
   * Invalidate the internal temporary array and call superclass method.
   */
  void Modified() override;

  // svtkAbstractArray override:
  bool HasStandardMemoryLayout() const override { return false; }

protected:
  svtkMappedDataArray();
  ~svtkMappedDataArray() override;

  int GetArrayType() const override { return svtkAbstractArray::MappedDataArray; }

private:
  svtkMappedDataArray(const svtkMappedDataArray&) = delete;
  void operator=(const svtkMappedDataArray&) = delete;

  //@{
  /**
   * GetVoidPointer.
   */
  ValueType* TemporaryScalarPointer;
  size_t TemporaryScalarPointerSize;
  //@}
};

// Declare svtkArrayDownCast implementations for mapped containers:
svtkArrayDownCast_TemplateFastCastMacro(svtkMappedDataArray);

#include "svtkMappedDataArray.txx"

// Adds an implementation of NewInstanceInternal() that returns an AoS
// (unmapped) SVTK array, if possible. Use this in combination with
// svtkAbstractTemplateTypeMacro when your subclass is a template class.
// Otherwise, use svtkMappedDataArrayTypeMacro.
#define svtkMappedDataArrayNewInstanceMacro(thisClass)                                              \
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

// Same as svtkTypeMacro, but adds an implementation of NewInstanceInternal()
// that returns a standard (unmapped) SVTK array, if possible.
#define svtkMappedDataArrayTypeMacro(thisClass, superClass)                                         \
  svtkAbstractTypeMacroWithNewInstanceType(thisClass, superClass, svtkDataArray);                    \
  svtkMappedDataArrayNewInstanceMacro(thisClass)

#endif // svtkMappedDataArray_h

// SVTK-HeaderTest-Exclude: svtkMappedDataArray.h
