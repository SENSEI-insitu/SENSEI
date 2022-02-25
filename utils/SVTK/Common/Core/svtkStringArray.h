/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStringArray.h
  Language:  C++

  Copyright 2004 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
  license for use of this work by or on behalf of the
  U.S. Government. Redistribution and use in source and binary forms, with
  or without modification, are permitted provided that this Notice and any
  statement of authorship are reproduced on all copies.

=========================================================================*/

/**
 * @class   svtkStringArray
 * @brief   a svtkAbstractArray subclass for strings
 *
 * Points and cells may sometimes have associated data that are stored
 * as strings, e.g. labels for information visualization projects.
 * This class provides a clean way to store and access those strings.
 * @par Thanks:
 * Andy Wilson (atwilso@sandia.gov) wrote this class.
 */

#ifndef svtkStringArray_h
#define svtkStringArray_h

#include "svtkAbstractArray.h"
#include "svtkCommonCoreModule.h" // For export macro
#include "svtkStdString.h"        // needed for svtkStdString definition

class svtkStringArrayLookup;

class SVTKCOMMONCORE_EXPORT svtkStringArray : public svtkAbstractArray
{
public:
  enum DeleteMethod
  {
    SVTK_DATA_ARRAY_FREE = svtkAbstractArray::SVTK_DATA_ARRAY_FREE,
    SVTK_DATA_ARRAY_DELETE = svtkAbstractArray::SVTK_DATA_ARRAY_DELETE,
    SVTK_DATA_ARRAY_ALIGNED_FREE = svtkAbstractArray::SVTK_DATA_ARRAY_ALIGNED_FREE,
    SVTK_DATA_ARRAY_USER_DEFINED = svtkAbstractArray::SVTK_DATA_ARRAY_USER_DEFINED
  };

  static svtkStringArray* New();
  svtkTypeMacro(svtkStringArray, svtkAbstractArray);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //
  //
  // Functions required by svtkAbstractArray
  //
  //

  /**
   * Get the data type.
   */
  int GetDataType() const override { return SVTK_STRING; }

  int IsNumeric() const override { return 0; }

  /**
   * Release storage and reset array to initial state.
   */
  void Initialize() override;

  /**
   * Return the size of the data type.  WARNING: This may not mean
   * what you expect with strings.  It will return
   * sizeof(std::string) and not take into account the data
   * included in any particular string.
   */
  int GetDataTypeSize() const override;

  /**
   * Free any unnecessary memory.
   * Resize object to just fit data requirement. Reclaims extra memory.
   */
  void Squeeze() override { this->ResizeAndExtend(this->MaxId + 1); }

  /**
   * Resize the array while conserving the data.
   */
  svtkTypeBool Resize(svtkIdType numTuples) override;

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
   * Given a list of indices, return an array of values.  You must
   * insure that the output array has been previously allocated with
   * enough space to hold the data and that the types match
   * sufficiently to allow conversion (if necessary).
   */
  void GetTuples(svtkIdList* ptIds, svtkAbstractArray* output) override;

  /**
   * Get the values for the range of indices specified (i.e.,
   * p1->p2 inclusive). You must insure that the output array has been
   * previously allocated with enough space to hold the data and that
   * the type of the output array is compatible with the type of this
   * array.
   */
  void GetTuples(svtkIdType p1, svtkIdType p2, svtkAbstractArray* output) override;

  /**
   * Allocate memory for this array. Delete old storage only if necessary.
   * Note that ext is no longer used.
   */
  svtkTypeBool Allocate(svtkIdType sz, svtkIdType ext = 1000) override;

  /**
   * Get the data at a particular index.
   */
  svtkStdString& GetValue(svtkIdType id) SVTK_EXPECTS(0 <= id && id < this->GetNumberOfValues());

  /**
   * Set the data at a particular index. Does not do range checking. Make sure
   * you use the method SetNumberOfValues() before inserting data.
   */
  void SetValue(svtkIdType id, svtkStdString value)
    SVTK_EXPECTS(0 <= id && id < this->GetNumberOfValues())
  {
    this->Array[id] = value;
    this->DataChanged();
  }

  void SetValue(svtkIdType id, const char* value)
    SVTK_EXPECTS(0 <= id && id < this->GetNumberOfValues()) SVTK_EXPECTS(value != nullptr);

  /**
   * Set the number of tuples (a component group) in the array. Note that
   * this may allocate space depending on the number of components.
   */
  void SetNumberOfTuples(svtkIdType number) override
  {
    this->SetNumberOfValues(this->NumberOfComponents * number);
  }

  svtkIdType GetNumberOfValues() { return this->MaxId + 1; }

  int GetNumberOfElementComponents() { return 0; }
  int GetElementComponentSize() const override
  {
    return static_cast<int>(sizeof(svtkStdString::value_type));
  }

  /**
   * Insert data at a specified position in the array.
   */
  void InsertValue(svtkIdType id, svtkStdString f) SVTK_EXPECTS(0 <= id);
  void InsertValue(svtkIdType id, const char* val) SVTK_EXPECTS(0 <= id) SVTK_EXPECTS(val != nullptr);

  /**
   * Set a value in the array form a variant.
   * Insert a value into the array from a variant.
   */
  void SetVariantValue(svtkIdType idx, svtkVariant value) override;

  /**
   * Safely set a value in the array form a variant.
   * Safely insert a value into the array from a variant.
   */
  void InsertVariantValue(svtkIdType idx, svtkVariant value) override;

  /**
   * Insert data at the end of the array. Return its location in the array.
   */
  svtkIdType InsertNextValue(svtkStdString f);
  svtkIdType InsertNextValue(const char* f) SVTK_EXPECTS(f != nullptr);

  /**
   * Get the address of a particular data index. Make sure data is allocated
   * for the number of items requested. Set MaxId according to the number of
   * data values requested.
   */
  svtkStdString* WritePointer(svtkIdType id, svtkIdType number);

  /**
   * Get the address of a particular data index. Performs no checks
   * to verify that the memory has been allocated etc.
   */
  svtkStdString* GetPointer(svtkIdType id) { return this->Array + id; }
  void* GetVoidPointer(svtkIdType id) override { return this->GetPointer(id); }

  /**
   * Deep copy of another string array.  Will complain and change nothing
   * if the array passed in is not a svtkStringArray.
   */
  void DeepCopy(svtkAbstractArray* aa) override;

  /**
   * This method lets the user specify data to be held by the array.  The
   * array argument is a pointer to the data.  size is the size of
   * the array supplied by the user.  Set save to 1 to keep the class
   * from deleting the array when it cleans up or reallocates memory.
   * The class uses the actual array provided; it does not copy the data
   * from the supplied array. If save is 0, then this class is free to delete
   * the array when it cleans up or reallocates using the provided free function
   * If the delete method is SVTK_DATA_ARRAY_USER_DEFINED
   * a custom free function can be assigned to be called using SetArrayFreeFunction,
   * if no custom function is assigned we will default to delete[].
   */
  void SetArray(
    svtkStdString* array, svtkIdType size, int save, int deleteMethod = SVTK_DATA_ARRAY_DELETE);
  void SetVoidArray(void* array, svtkIdType size, int save) override
  {
    this->SetArray(static_cast<svtkStdString*>(array), size, save);
  }
  void SetVoidArray(void* array, svtkIdType size, int save, int deleteMethod) override
  {
    this->SetArray(static_cast<svtkStdString*>(array), size, save, deleteMethod);
  }

  /**
   * This method allows the user to specify a custom free function to be
   * called when the array is deallocated. Calling this method will implicitly
   * mean that the given free function will be called when the class
   * cleans up or reallocates memory.
   **/
  void SetArrayFreeFunction(void (*callback)(void*)) override;

  /**
   * Return the memory in kibibytes (1024 bytes) consumed by this data array. Used to
   * support streaming and reading/writing data. The value returned is
   * guaranteed to be greater than or equal to the memory required to
   * actually represent the data represented by this object. The
   * information returned is valid only after the pipeline has
   * been updated.

   * This function takes into account the size of the contents of the
   * strings as well as the string containers themselves.
   */
  unsigned long GetActualMemorySize() const override;

  /**
   * Returns a svtkArrayIteratorTemplate<svtkStdString>.
   */
  SVTK_NEWINSTANCE svtkArrayIterator* NewIterator() override;

  /**
   * Returns the size of the data in DataTypeSize units. Thus, the number of bytes
   * for the data can be computed by GetDataSize() * GetDataTypeSize().
   * The size computation includes the string termination character for each string.
   */
  svtkIdType GetDataSize() const override;

  //@{
  /**
   * Return the indices where a specific value appears.
   */
  svtkIdType LookupValue(svtkVariant value) override;
  void LookupValue(svtkVariant value, svtkIdList* ids) override;
  //@}

  svtkIdType LookupValue(const svtkStdString& value);
  void LookupValue(const svtkStdString& value, svtkIdList* ids);

  svtkIdType LookupValue(const char* value);
  void LookupValue(const char* value, svtkIdList* ids);

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

protected:
  svtkStringArray();
  ~svtkStringArray() override;

  svtkStdString* Array;                         // pointer to data
  svtkStdString* ResizeAndExtend(svtkIdType sz); // function to resize data

  void (*DeleteFunction)(void*);

private:
  svtkStringArray(const svtkStringArray&) = delete;
  void operator=(const svtkStringArray&) = delete;

  svtkStringArrayLookup* Lookup;
  void UpdateLookup();
};

#endif
