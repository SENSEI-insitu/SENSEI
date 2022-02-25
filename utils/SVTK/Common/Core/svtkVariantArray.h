/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkVariantArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------*/
/**
 * @class   svtkVariantArray
 * @brief   An array holding svtkVariants.
 *
 *
 *
 * @par Thanks:
 * Thanks to Patricia Crossno, Ken Moreland, Andrew Wilson and Brian Wylie from
 * Sandia National Laboratories for their help in developing this class.
 */

#ifndef svtkVariantArray_h
#define svtkVariantArray_h

#include "svtkAbstractArray.h"
#include "svtkCommonCoreModule.h" // For export macro
#include "svtkVariant.h"          // For variant type

class svtkVariantArrayLookup;

/// Forward declaration required for Boost serialization
namespace boost
{
namespace serialization
{
class access;
}
}

class SVTKCOMMONCORE_EXPORT svtkVariantArray : public svtkAbstractArray
{

  /// Friendship required for Boost serialization
  friend class boost::serialization::access;

public:
  enum DeleteMethod
  {
    SVTK_DATA_ARRAY_FREE = svtkAbstractArray::SVTK_DATA_ARRAY_FREE,
    SVTK_DATA_ARRAY_DELETE = svtkAbstractArray::SVTK_DATA_ARRAY_DELETE,
    SVTK_DATA_ARRAY_ALIGNED_FREE = svtkAbstractArray::SVTK_DATA_ARRAY_ALIGNED_FREE,
    SVTK_DATA_ARRAY_USER_DEFINED = svtkAbstractArray::SVTK_DATA_ARRAY_USER_DEFINED
  };

  static svtkVariantArray* New();
  svtkTypeMacro(svtkVariantArray, svtkAbstractArray);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //
  // Functions required by svtkAbstractArray
  //

  /**
   * Allocate memory for this array. Delete old storage only if necessary.
   * Note that ext is no longer used.
   */
  svtkTypeBool Allocate(svtkIdType sz, svtkIdType ext = 1000) override;

  /**
   * Release storage and reset array to initial state.
   */
  void Initialize() override;

  /**
   * Return the underlying data type. An integer indicating data type is
   * returned as specified in svtkSetGet.h.
   */
  int GetDataType() const override;

  /**
   * Return the size of the underlying data type.  For a bit, 1 is
   * returned.  For string 0 is returned. Arrays with variable length
   * components return 0.
   */
  int GetDataTypeSize() const override;

  /**
   * Return the size, in bytes, of the lowest-level element of an
   * array.  For svtkDataArray and subclasses this is the size of the
   * data type.  For svtkStringArray, this is
   * sizeof(svtkStdString::value_type), which winds up being
   * sizeof(char).
   */
  int GetElementComponentSize() const override;

  /**
   * Set the number of tuples (a component group) in the array. Note that
   * this may allocate space depending on the number of components.
   */
  void SetNumberOfTuples(svtkIdType number) override;

  /**
   * Set the tuple at the ith location using the jth tuple in the source array.
   * This method assumes that the two arrays have the same type
   * and structure. Note that range checking and memory allocation is not
   * performed; use in conjunction with SetNumberOfTuples() to allocate space.
   */
  void SetTuple(svtkIdType i, svtkIdType j, svtkAbstractArray* source) override;

  /**
   * Insert the jth tuple in the source array, at ith location in this array.
   * Note that memory allocation is performed as necessary to hold the data.
   */
  void InsertTuple(svtkIdType i, svtkIdType j, svtkAbstractArray* source) override;

  /**
   * Copy the tuples indexed in srcIds from the source array to the tuple
   * locations indexed by dstIds in this array.
   * Note that memory allocation is performed as necessary to hold the data.
   */
  void InsertTuples(svtkIdList* dstIds, svtkIdList* srcIds, svtkAbstractArray* source) override;

  /**
   * Copy n consecutive tuples starting at srcStart from the source array to
   * this array, starting at the dstStart location.
   * Note that memory allocation is performed as necessary to hold the data.
   */
  void InsertTuples(
    svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkAbstractArray* source) override;

  /**
   * Insert the jth tuple in the source array, at the end in this array.
   * Note that memory allocation is performed as necessary to hold the data.
   * Returns the location at which the data was inserted.
   */
  svtkIdType InsertNextTuple(svtkIdType j, svtkAbstractArray* source) override;

  /**
   * Return a void pointer. For image pipeline interface and other
   * special pointer manipulation.
   */
  void* GetVoidPointer(svtkIdType id) override;

  /**
   * Deep copy of data. Implementation left to subclasses, which
   * should support as many type conversions as possible given the
   * data type.
   */
  void DeepCopy(svtkAbstractArray* da) override;

  /**
   * Set the ith tuple in this array as the interpolated tuple value,
   * given the ptIndices in the source array and associated
   * interpolation weights.
   * This method assumes that the two arrays are of the same type
   * and structure.
   */
  void InterpolateTuple(
    svtkIdType i, svtkIdList* ptIndices, svtkAbstractArray* source, double* weights) override;

  /**
   * Insert the ith tuple in this array as interpolated from the two values,
   * p1 and p2, and an interpolation factor, t.
   * The interpolation factor ranges from (0,1),
   * with t=0 located at p1. This method assumes that the three arrays are of
   * the same type. p1 is value at index id1 in source1, while, p2 is
   * value at index id2 in source2.
   */
  void InterpolateTuple(svtkIdType i, svtkIdType id1, svtkAbstractArray* source1, svtkIdType id2,
    svtkAbstractArray* source2, double t) override;

  /**
   * Free any unnecessary memory.
   * Description:
   * Resize object to just fit data requirement. Reclaims extra memory.
   */
  void Squeeze() override;

  /**
   * Resize the array while conserving the data.  Returns 1 if
   * resizing succeeded and 0 otherwise.
   */
  svtkTypeBool Resize(svtkIdType numTuples) override;

  //@{
  /**
   * This method lets the user specify data to be held by the array.  The
   * array argument is a pointer to the data.  size is the size of
   * the array supplied by the user.  Set save to 1 to keep the class
   * from deleting the array when it cleans up or reallocates memory.
   * The class uses the actual array provided; it does not copy the data
   * from the supplied array.
   */
  void SetVoidArray(void* arr, svtkIdType size, int save) override;
  void SetVoidArray(void* arr, svtkIdType size, int save, int deleteM) override;
  //@}

  /**
   * Return the memory in kibibytes (1024 bytes) consumed by this data array. Used to
   * support streaming and reading/writing data. The value returned is
   * guaranteed to be greater than or equal to the memory required to
   * actually represent the data represented by this object. The
   * information returned is valid only after the pipeline has
   * been updated.
   */
  unsigned long GetActualMemorySize() const override;

  /**
   * Since each item can be of a different type, we say that a variant array is not numeric.
   */
  int IsNumeric() const override;

  /**
   * Subclasses must override this method and provide the right
   * kind of templated svtkArrayIteratorTemplate.
   */
  SVTK_NEWINSTANCE svtkArrayIterator* NewIterator() override;

  //
  // Additional functions
  //

  /**
   * Get the data at a particular index.
   */
  svtkVariant& GetValue(svtkIdType id) const;

  /**
   * Set the data at a particular index. Does not do range checking. Make sure
   * you use the method SetNumberOfValues() before inserting data.
   */
  void SetValue(svtkIdType id, svtkVariant value)
    SVTK_EXPECTS(0 <= id && id < this->GetNumberOfValues());

  /**
   * If id < GetNumberOfValues(), overwrite the array at that index.
   * If id >= GetNumberOfValues(), expand the array size to id+1
   * and set the final value to the specified value.
   */
  void InsertValue(svtkIdType id, svtkVariant value) SVTK_EXPECTS(0 <= id);

  /**
   * Insert a value into the array from a variant.
   */
  void SetVariantValue(svtkIdType idx, svtkVariant value) override;

  /**
   * Safely insert a value into the array from a variant.
   */
  void InsertVariantValue(svtkIdType idx, svtkVariant value) override;

  /**
   * Expand the array by one and set the value at that location.
   * Return the array index of the inserted value.
   */
  svtkIdType InsertNextValue(svtkVariant value);

  /**
   * Return a pointer to the location in the internal array at the specified index.
   */
  svtkVariant* GetPointer(svtkIdType id);

  /**
   * Set the internal array used by this object.
   */
  void SetArray(
    svtkVariant* arr, svtkIdType size, int save, int deleteMethod = SVTK_DATA_ARRAY_DELETE);

  /**
   * This method allows the user to specify a custom free function to be
   * called when the array is deallocated. Calling this method will implicitly
   * mean that the given free function will be called when the class
   * cleans up or reallocates memory.
   **/
  void SetArrayFreeFunction(void (*callback)(void*)) override;

  /**
   * Return the number of values in the array.
   */
  svtkIdType GetNumberOfValues() { return this->MaxId + 1; }

  //@{
  /**
   * Return the indices where a specific value appears.
   */
  svtkIdType LookupValue(svtkVariant value) override;
  void LookupValue(svtkVariant value, svtkIdList* ids) override;
  //@}

  /**
   * Tell the array explicitly that the data has changed.
   * This is only necessary to call when you modify the array contents
   * without using the array's API (i.e. you retrieve a pointer to the
   * data and modify the array contents).  You need to call this so that
   * the fast lookup will know to rebuild itself.  Otherwise, the lookup
   * functions will give incorrect results.
   */
  void DataChanged() override;

  /**
   * Tell the array explicitly that a single data element has
   * changed. Like DataChanged(), then is only necessary when you
   * modify the array contents without using the array's API.
   */
  virtual void DataElementChanged(svtkIdType id);

  /**
   * Delete the associated fast lookup data structure on this array,
   * if it exists.  The lookup will be rebuilt on the next call to a lookup
   * function.
   */
  void ClearLookup() override;

  /**
   * This destructor is public to work around a bug in version 1.36.0 of
   * the Boost.Serialization library.
   */
  ~svtkVariantArray() override;

protected:
  // Construct object with default tuple dimension (number of components) of 1.
  svtkVariantArray();

  // Pointer to data

  svtkVariant* Array;

  // Function to resize data
  svtkVariant* ResizeAndExtend(svtkIdType sz);

  void (*DeleteFunction)(void*);

private:
  svtkVariantArray(const svtkVariantArray&) = delete;
  void operator=(const svtkVariantArray&) = delete;

  svtkVariantArrayLookup* Lookup;
  void UpdateLookup();
};

#endif
