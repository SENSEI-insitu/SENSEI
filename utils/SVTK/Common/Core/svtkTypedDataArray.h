/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTypedDataArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkTypedDataArray
 * @brief   Extend svtkDataArray with abstract type-specific API
 *
 *
 * This templated class decorates svtkDataArray with additional type-specific
 * methods that can be used to interact with the data.
 *
 * NOTE: This class has been made obsolete by the newer svtkGenericDataArray.
 *
 * @warning
 * This class uses svtkTypeTraits to implement GetDataType(). Since svtkIdType
 * is a typedef for either a 32- or 64-bit integer, subclasses that are designed
 * to hold svtkIdTypes will, by default, return an incorrect value from
 * GetDataType(). To fix this, such subclasses should override GetDataType() to
 * return SVTK_ID_TYPE.
 *
 * @sa
 * svtkGenericDataArray
 */

#ifndef svtkTypedDataArray_h
#define svtkTypedDataArray_h

#include "svtkGenericDataArray.h"

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkTypeTraits.h"       // For type metadata

template <class Scalar>
class svtkTypedDataArrayIterator;

template <class Scalar>
class svtkTypedDataArray : public svtkGenericDataArray<svtkTypedDataArray<Scalar>, Scalar>
{
  typedef svtkGenericDataArray<svtkTypedDataArray<Scalar>, Scalar> GenericDataArrayType;

public:
  svtkTemplateTypeMacro(svtkTypedDataArray<Scalar>, GenericDataArrayType);
  typedef typename Superclass::ValueType ValueType;

  /**
   * Typedef to a suitable iterator class.
   */
  typedef svtkTypedDataArrayIterator<ValueType> Iterator;

  /**
   * Return an iterator initialized to the first element of the data.
   */
  Iterator Begin();

  /**
   * Return an iterator initialized to first element past the end of the data.
   */
  Iterator End();

  /**
   * Compile time access to the SVTK type identifier.
   */
  enum
  {
    SVTK_DATA_TYPE = svtkTypeTraits<ValueType>::SVTK_TYPE_ID
  };

  /**
   * Perform a fast, safe cast from a svtkAbstractArray to a svtkTypedDataArray.
   * This method checks if:
   * - source->GetArrayType() is appropriate, and
   * - source->GetDataType() matches the Scalar template argument
   * if these conditions are met, the method performs a static_cast to return
   * source as a svtkTypedDataArray pointer. Otherwise, nullptr is returned.
   */
  static svtkTypedDataArray<Scalar>* FastDownCast(svtkAbstractArray* source);

  /**
   * Return the SVTK data type held by this array.
   */
  int GetDataType() const override;

  /**
   * Return the size of the element type in bytes.
   */
  int GetDataTypeSize() const override;

  /**
   * Set the tuple value at the ith location in the array.
   */
  virtual void SetTypedTuple(svtkIdType i, const ValueType* t) = 0;

  /**
   * Insert (memory allocation performed) the tuple into the ith location
   * in the array.
   */
  virtual void InsertTypedTuple(svtkIdType i, const ValueType* t) = 0;

  /**
   * Insert (memory allocation performed) the tuple onto the end of the array.
   */
  virtual svtkIdType InsertNextTypedTuple(const ValueType* t) = 0;

  /**
   * Get the data at a particular index.
   */
  virtual ValueType GetValue(svtkIdType idx) const = 0;

  /**
   * Get a reference to the scalar value at a particular index.
   */
  virtual ValueType& GetValueReference(svtkIdType idx) = 0;

  /**
   * Set the data at a particular index. Does not do range checking. Make sure
   * you use the method SetNumberOfValues() before inserting data.
   */
  virtual void SetValue(svtkIdType idx, ValueType value) = 0;

  /**
   * Copy the tuple value into a user-provided array.
   */
  virtual void GetTypedTuple(svtkIdType idx, ValueType* t) const = 0;

  /**
   * Insert data at the end of the array. Return its location in the array.
   */
  virtual svtkIdType InsertNextValue(ValueType v) = 0;

  /**
   * Insert data at a specified position in the array.
   */
  virtual void InsertValue(svtkIdType idx, ValueType v) = 0;

  virtual ValueType GetTypedComponent(svtkIdType tupleIdx, int comp) const;
  virtual void SetTypedComponent(svtkIdType tupleIdx, int comp, ValueType v);

  /**
   * Method for type-checking in FastDownCast implementations.
   */
  int GetArrayType() const override { return svtkAbstractArray::TypedDataArray; }

  // Reintroduced as pure virtual since the base svtkGenericDataArray method
  // requires new allocation/resize APIs, though existing MappedDataArrays
  // would just use the svtkDataArray-level virtuals.
  svtkTypeBool Allocate(svtkIdType size, svtkIdType ext = 1000) override = 0;
  svtkTypeBool Resize(svtkIdType numTuples) override = 0;

protected:
  svtkTypedDataArray();
  ~svtkTypedDataArray() override;

  /**
   * Needed for svtkGenericDataArray API, but just aborts. Override Allocate
   * instead.
   */
  virtual bool AllocateTuples(svtkIdType numTuples);

  /**
   * Needed for svtkGenericDataArray API, but just aborts. Override Resize
   * instead.
   */
  virtual bool ReallocateTuples(svtkIdType numTuples);

private:
  svtkTypedDataArray(const svtkTypedDataArray&) = delete;
  void operator=(const svtkTypedDataArray&) = delete;

  friend class svtkGenericDataArray<svtkTypedDataArray<Scalar>, Scalar>;
};

// Declare svtkArrayDownCast implementations for typed containers:
svtkArrayDownCast_TemplateFastCastMacro(svtkTypedDataArray);

// Included here to resolve chicken/egg issue with container/iterator:
#include "svtkTypedDataArrayIterator.h" // For iterator

template <class Scalar>
inline typename svtkTypedDataArray<Scalar>::Iterator svtkTypedDataArray<Scalar>::Begin()
{
  return Iterator(this, 0);
}

template <class Scalar>
inline typename svtkTypedDataArray<Scalar>::Iterator svtkTypedDataArray<Scalar>::End()
{
  return Iterator(this, this->MaxId + 1);
}

#include "svtkTypedDataArray.txx"

#endif // svtkTypedDataArray_h

// SVTK-HeaderTest-Exclude: svtkTypedDataArray.h
