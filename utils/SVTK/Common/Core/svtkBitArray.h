/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBitArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkBitArray
 * @brief   dynamic, self-adjusting array of bits
 *
 * svtkBitArray is an array of bits (0/1 data value). The array is packed
 * so that each byte stores eight bits. svtkBitArray provides methods
 * for insertion and retrieval of bits, and will automatically resize
 * itself to hold new data.
 */

#ifndef svtkBitArray_h
#define svtkBitArray_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkDataArray.h"

class svtkBitArrayLookup;

class SVTKCOMMONCORE_EXPORT svtkBitArray : public svtkDataArray
{
public:
  enum DeleteMethod
  {
    SVTK_DATA_ARRAY_FREE = svtkAbstractArray::SVTK_DATA_ARRAY_FREE,
    SVTK_DATA_ARRAY_DELETE = svtkAbstractArray::SVTK_DATA_ARRAY_DELETE,
    SVTK_DATA_ARRAY_ALIGNED_FREE = svtkAbstractArray::SVTK_DATA_ARRAY_ALIGNED_FREE,
    SVTK_DATA_ARRAY_USER_DEFINED = svtkAbstractArray::SVTK_DATA_ARRAY_USER_DEFINED
  };

  static svtkBitArray* New();
  svtkTypeMacro(svtkBitArray, svtkDataArray);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Allocate memory for this array. Delete old storage only if necessary.
   * Note that ext is no longer used.
   */
  svtkTypeBool Allocate(svtkIdType sz, svtkIdType ext = 1000) override;

  /**
   * Release storage and reset array to initial state.
   */
  void Initialize() override;

  // satisfy svtkDataArray API
  int GetDataType() const override { return SVTK_BIT; }
  int GetDataTypeSize() const override { return 0; }

  /**
   * Set the number of n-tuples in the array.
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
   * Get a pointer to a tuple at the ith location. This is a dangerous method
   * (it is not thread safe since a pointer is returned).
   */
  double* GetTuple(svtkIdType i) override;

  /**
   * Copy the tuple value into a user-provided array.
   */
  void GetTuple(svtkIdType i, double* tuple) override;

  //@{
  /**
   * Set the tuple value at the ith location in the array.
   */
  void SetTuple(svtkIdType i, const float* tuple) override;
  void SetTuple(svtkIdType i, const double* tuple) override;
  //@}

  //@{
  /**
   * Insert (memory allocation performed) the tuple into the ith location
   * in the array.
   */
  void InsertTuple(svtkIdType i, const float* tuple) override;
  void InsertTuple(svtkIdType i, const double* tuple) override;
  //@}

  //@{
  /**
   * Insert (memory allocation performed) the tuple onto the end of the array.
   */
  svtkIdType InsertNextTuple(const float* tuple) override;
  svtkIdType InsertNextTuple(const double* tuple) override;
  //@}

  //@{
  /**
   * These methods remove tuples from the data array. They shift data and
   * resize array, so the data array is still valid after this operation. Note,
   * this operation is fairly slow.
   */
  void RemoveTuple(svtkIdType id) override;
  void RemoveFirstTuple() override;
  void RemoveLastTuple() override;
  //@}

  /**
   * Set the data component at the ith tuple and jth component location.
   * Note that i is less then NumberOfTuples and j is less then
   * NumberOfComponents. Make sure enough memory has been allocated (use
   * SetNumberOfTuples() and  SetNumberOfComponents()).
   */
  void SetComponent(svtkIdType i, int j, double c) override;

  /**
   * Free any unneeded memory.
   */
  void Squeeze() override;

  /**
   * Resize the array while conserving the data.
   */
  svtkTypeBool Resize(svtkIdType numTuples) override;

  /**
   * Get the data at a particular index.
   */
  int GetValue(svtkIdType id) const;

  /**
   * Set the data at a particular index. Does not do range checking. Make sure
   * you use the method SetNumberOfValues() before inserting data.
   */
  void SetValue(svtkIdType id, int value);

  /**
   * Inserts values and checks to make sure there is enough memory
   */
  void InsertValue(svtkIdType id, int i);

  /**
   * Set a value in the array from a variant.
   */
  void SetVariantValue(svtkIdType idx, svtkVariant value) override;

  /**
   * Inserts values from a variant and checks to ensure there is enough memory
   */
  void InsertVariantValue(svtkIdType idx, svtkVariant value) override;

  svtkIdType InsertNextValue(int i);

  /**
   * Insert the data component at ith tuple and jth component location.
   * Note that memory allocation is performed as necessary to hold the data.
   */
  void InsertComponent(svtkIdType i, int j, double c) override;

  /**
   * Direct manipulation of the underlying data.
   */
  unsigned char* GetPointer(svtkIdType id) { return this->Array + id / 8; }

  /**
   * Get the address of a particular data index. Make sure data is allocated
   * for the number of items requested. Set MaxId according to the number of
   * data values requested.
   */
  unsigned char* WritePointer(svtkIdType id, svtkIdType number);

  void* WriteVoidPointer(svtkIdType id, svtkIdType number) override
  {
    return this->WritePointer(id, number);
  }

  void* GetVoidPointer(svtkIdType id) override { return static_cast<void*>(this->GetPointer(id)); }

  /**
   * Deep copy of another bit array.
   */
  void DeepCopy(svtkDataArray* da) override;
  void DeepCopy(svtkAbstractArray* aa) override { this->Superclass::DeepCopy(aa); }

  //@{
  /**
   * This method lets the user specify data to be held by the array.  The
   * array argument is a pointer to the data.  size is the size of
   * the array supplied by the user.  Set save to 1 to keep the class
   * from deleting the array when it cleans up or reallocates memory.
   * The class uses the actual array provided; it does not copy the data
   * from the supplied array.
   * If the delete method is SVTK_DATA_ARRAY_USER_DEFINED
   * a custom free function can be assigned to be called using SetArrayFreeFunction,
   * if no custom function is assigned we will default to delete[].
   */
#ifndef __SVTK_WRAP__
  void SetArray(
    unsigned char* array, svtkIdType size, int save, int deleteMethod = SVTK_DATA_ARRAY_DELETE);
#endif
  void SetVoidArray(void* array, svtkIdType size, int save) override
  {
    this->SetArray(static_cast<unsigned char*>(array), size, save);
  }
  void SetVoidArray(void* array, svtkIdType size, int save, int deleteMethod) override
  {
    this->SetArray(static_cast<unsigned char*>(array), size, save, deleteMethod);
  }
  //@}

  /**
   * This method allows the user to specify a custom free function to be
   * called when the array is deallocated. Calling this method will implicitly
   * mean that the given free function will be called when the class
   * cleans up or reallocates memory.
   **/
  void SetArrayFreeFunction(void (*callback)(void*)) override;

  /**
   * Returns a new svtkBitArrayIterator instance.
   */
  SVTK_NEWINSTANCE svtkArrayIterator* NewIterator() override;

  //@{
  /**
   * Return the indices where a specific value appears.
   */
  svtkIdType LookupValue(svtkVariant value) override;
  void LookupValue(svtkVariant value, svtkIdList* ids) override;
  svtkIdType LookupValue(int value);
  void LookupValue(int value, svtkIdList* ids);
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
   * Delete the associated fast lookup data structure on this array,
   * if it exists.  The lookup will be rebuilt on the next call to a lookup
   * function.
   */
  void ClearLookup() override;

protected:
  svtkBitArray();
  ~svtkBitArray() override;

  unsigned char* Array; // pointer to data
  unsigned char* ResizeAndExtend(svtkIdType sz);
  // function to resize data

  int TupleSize; // used for data conversion
  double* Tuple;

  void (*DeleteFunction)(void*);

private:
  // hide superclass' DeepCopy() from the user and the compiler
  void DeepCopy(svtkDataArray& da) { this->svtkDataArray::DeepCopy(&da); }

private:
  svtkBitArray(const svtkBitArray&) = delete;
  void operator=(const svtkBitArray&) = delete;

  svtkBitArrayLookup* Lookup;
  void UpdateLookup();
};

inline void svtkBitArray::SetValue(svtkIdType id, int value)
{
  if (value)
  {
    this->Array[id / 8] = static_cast<unsigned char>(this->Array[id / 8] | (0x80 >> id % 8));
  }
  else
  {
    this->Array[id / 8] = static_cast<unsigned char>(this->Array[id / 8] & (~(0x80 >> id % 8)));
  }
  this->DataChanged();
}

inline void svtkBitArray::InsertValue(svtkIdType id, int i)
{
  if (id >= this->Size)
  {
    if (!this->ResizeAndExtend(id + 1))
    {
      return;
    }
  }
  if (i)
  {
    this->Array[id / 8] = static_cast<unsigned char>(this->Array[id / 8] | (0x80 >> id % 8));
  }
  else
  {
    this->Array[id / 8] = static_cast<unsigned char>(this->Array[id / 8] & (~(0x80 >> id % 8)));
  }
  if (id > this->MaxId)
  {
    this->MaxId = id;
  }
  this->DataChanged();
}

inline void svtkBitArray::SetVariantValue(svtkIdType id, svtkVariant value)
{
  this->SetValue(id, value.ToInt());
}

inline void svtkBitArray::InsertVariantValue(svtkIdType id, svtkVariant value)
{
  this->InsertValue(id, value.ToInt());
}

inline svtkIdType svtkBitArray::InsertNextValue(int i)
{
  this->InsertValue(++this->MaxId, i);
  this->DataChanged();
  return this->MaxId;
}

inline void svtkBitArray::Squeeze()
{
  this->ResizeAndExtend(this->MaxId + 1);
}

#endif
