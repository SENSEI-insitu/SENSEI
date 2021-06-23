/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDenseArray.h

-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkDenseArray
 * @brief   Contiguous storage for N-way arrays.
 *
 *
 * svtkDenseArray is a concrete svtkArray implementation that stores values
 * using a contiguous block of memory.  Values are stored with fortran ordering,
 * meaning that if you iterated over the memory block, the left-most coordinates
 * would vary the fastest.
 *
 * In addition to the retrieval and update methods provided by svtkTypedArray,
 * svtkDenseArray provides methods to:
 *
 * Fill the entire array with a specific value.
 *
 * Retrieve a pointer to the storage memory block.
 *
 * @sa
 * svtkArray, svtkTypedArray, svtkSparseArray
 *
 * @par Thanks:
 * Developed by Timothy M. Shead (tshead@sandia.gov) at Sandia National Laboratories.
 */

#ifndef svtkDenseArray_h
#define svtkDenseArray_h

#include "svtkArrayCoordinates.h"
#include "svtkObjectFactory.h"
#include "svtkTypedArray.h"

template <typename T>
class svtkDenseArray : public svtkTypedArray<T>
{
public:
  static svtkDenseArray<T>* New();
  svtkTemplateTypeMacro(svtkDenseArray<T>, svtkTypedArray<T>);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  typedef typename svtkArray::CoordinateT CoordinateT;
  typedef typename svtkArray::DimensionT DimensionT;
  typedef typename svtkArray::SizeT SizeT;

  // svtkArray API
  bool IsDense() override;
  const svtkArrayExtents& GetExtents() override;
  SizeT GetNonNullSize() override;
  void GetCoordinatesN(const SizeT n, svtkArrayCoordinates& coordinates) override;
  svtkArray* DeepCopy() override;

  // svtkTypedArray API
  const T& GetValue(CoordinateT i) override;
  const T& GetValue(CoordinateT i, CoordinateT j) override;
  const T& GetValue(CoordinateT i, CoordinateT j, CoordinateT k) override;
  const T& GetValue(const svtkArrayCoordinates& coordinates) override;
  const T& GetValueN(const SizeT n) override;
  void SetValue(CoordinateT i, const T& value) override;
  void SetValue(CoordinateT i, CoordinateT j, const T& value) override;
  void SetValue(CoordinateT i, CoordinateT j, CoordinateT k, const T& value) override;
  void SetValue(const svtkArrayCoordinates& coordinates, const T& value) override;
  void SetValueN(const SizeT n, const T& value) override;

  // svtkDenseArray API

  /**
   * Strategy object that contains a block of memory to be used by svtkDenseArray
   * for value storage.  The MemoryBlock object is responsible for freeing
   * memory when destroyed.
   */
  class MemoryBlock
  {
  public:
    virtual ~MemoryBlock();
    //@{
    /**
     * Returns a pointer to the block of memory to be used for storage.
     */
    virtual T* GetAddress() = 0;
    //@}
  };

  //@{
  /**
   * MemoryBlock implementation that manages internally-allocated memory using
   * new[] and delete[].  Note: HeapMemoryBlock is the default used by svtkDenseArray
   * for its "normal" internal memory allocation.
   */
  class HeapMemoryBlock : public MemoryBlock
  {
  public:
    HeapMemoryBlock(const svtkArrayExtents& extents);
    ~HeapMemoryBlock() override;
    T* GetAddress() override;
    //@}

  private:
    T* Storage;
  };

  //@{
  /**
   * MemoryBlock implementation that manages a static (will not be freed) memory block.
   */
  class StaticMemoryBlock : public MemoryBlock
  {
  public:
    StaticMemoryBlock(T* const storage);
    T* GetAddress() override;
    //@}

  private:
    T* Storage;
  };

  /**
   * Initializes the array to use an externally-allocated memory block.  The supplied
   * MemoryBlock must be large enough to store extents.GetSize() values.  The contents of
   * the memory must be stored contiguously with fortran ordering,

   * Dimension-labels are undefined after calling ExternalStorage() - you should
   * initialize them accordingly.

   * The array will use the supplied memory for storage until the array goes out of
   * scope, is configured to use a different memory block by calling ExternalStorage()
   * again, or is configured to use internally-allocated memory by calling Resize().

   * Note that the array will delete the supplied memory block when it is no longer in use.
   * caller's responsibility to ensure that the memory does not go out-of-scope until
   * the array has been destroyed or is no longer using it.
   */
  void ExternalStorage(const svtkArrayExtents& extents, MemoryBlock* storage);

  /**
   * Fills every element in the array with the given value.
   */
  void Fill(const T& value);

  /**
   * Returns a value by-reference, which is useful for performance and code-clarity.
   */
  T& operator[](const svtkArrayCoordinates& coordinates);

  /**
   * Returns a read-only reference to the underlying storage.  Values are stored
   * contiguously with fortran ordering.
   */
  const T* GetStorage() const;

  /**
   * Returns a mutable reference to the underlying storage.  Values are stored
   * contiguously with fortran ordering.  Use at your own risk!
   */
  T* GetStorage();

protected:
  svtkDenseArray();
  ~svtkDenseArray() override;

private:
  svtkDenseArray(const svtkDenseArray&) = delete;
  void operator=(const svtkDenseArray&) = delete;

  void InternalResize(const svtkArrayExtents& extents) override;
  void InternalSetDimensionLabel(DimensionT i, const svtkStdString& label) override;
  svtkStdString InternalGetDimensionLabel(DimensionT i) override;
  inline svtkIdType MapCoordinates(CoordinateT i);
  inline svtkIdType MapCoordinates(CoordinateT i, CoordinateT j);
  inline svtkIdType MapCoordinates(CoordinateT i, CoordinateT j, CoordinateT k);
  inline svtkIdType MapCoordinates(const svtkArrayCoordinates& coordinates);

  void Reconfigure(const svtkArrayExtents& extents, MemoryBlock* storage);

  typedef svtkDenseArray<T> ThisT;

  /**
   * Stores the current array extents (its size along each dimension)
   */
  svtkArrayExtents Extents;

  /**
   * Stores labels for each array dimension
   */
  std::vector<svtkStdString> DimensionLabels;

  /**
   * Manages array value memory storage.
   */
  MemoryBlock* Storage;

  //@{
  /**
   * Stores array values using a contiguous range of memory
   * with constant-time value lookup.
   */
  T* Begin;
  T* End;
  //@}

  /**
   * Stores the offset along each array dimension (used for fast lookups).
   */
  std::vector<svtkIdType> Offsets;
  //@{
  /**
   * Stores the stride along each array dimension (used for fast lookups).
   */
  std::vector<svtkIdType> Strides;
  //@}
};

#include "svtkDenseArray.txx"

#endif

// SVTK-HeaderTest-Exclude: svtkDenseArray.h
