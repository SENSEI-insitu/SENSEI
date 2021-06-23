/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArray.h

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
 * @class   svtkArray
 * @brief   Abstract interface for N-dimensional arrays.
 *
 *
 * svtkArray is the root of a hierarchy of arrays that can be used to
 * store data with any number of dimensions.  It provides an abstract
 * interface for retrieving and setting array attributes that are
 * independent of the type of values stored in the array - such as the
 * number of dimensions, extents along each dimension, and number of
 * values stored in the array.
 *
 * To get and set array values, the svtkTypedArray template class derives
 * from svtkArray and provides type-specific methods for retrieval and
 * update.
 *
 * Two concrete derivatives of svtkTypedArray are provided at the moment:
 * svtkDenseArray and svtkSparseArray, which provide dense and sparse
 * storage for arbitrary-dimension data, respectively.  Toolkit users
 * can create their own concrete derivatives that implement alternative
 * storage strategies, such as compressed-sparse-row, etc.  You could
 * also create an array that provided read-only access to 'virtual' data,
 * such as an array that returned a Fibonacci sequence, etc.
 *
 * @sa
 * svtkTypedArray, svtkDenseArray, svtkSparseArray
 *
 * @par Thanks:
 * Developed by Timothy M. Shead (tshead@sandia.gov) at  Sandia National
 * Laboratories.
 */

#ifndef svtkArray_h
#define svtkArray_h

#include "svtkArrayCoordinates.h"
#include "svtkArrayExtents.h"
#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"
#include "svtkStdString.h"
#include "svtkVariant.h"

class SVTKCOMMONCORE_EXPORT svtkArray : public svtkObject
{
public:
  svtkTypeMacro(svtkArray, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  typedef svtkArrayExtents::CoordinateT CoordinateT;
  typedef svtkArrayExtents::DimensionT DimensionT;
  typedef svtkArrayExtents::SizeT SizeT;

  enum
  {
    /// Used with CreateArray() to create dense arrays
    DENSE = 0,
    /// Used with CreateArray() to create sparse arrays
    SPARSE = 1
  };

  /**
   * Creates a new array where StorageType is one of svtkArray::DENSE
   * or svtkArray::SPARSE, and ValueType is one of SVTK_CHAR,
   * SVTK_UNSIGNED_CHAR, SVTK_SHORT, SVTK_UNSIGNED_SHORT,  SVTK_INT,
   * SVTK_UNSIGNED_INT, SVTK_LONG, SVTK_UNSIGNED_LONG, SVTK_DOUBLE,
   * SVTK_ID_TYPE, or SVTK_STRING.  The caller is responsible for the
   * lifetime of the returned object.
   */
  SVTK_NEWINSTANCE
  static svtkArray* CreateArray(int StorageType, int ValueType);

  /**
   * Returns true iff the underlying array storage is "dense", i.e. that
   * GetSize() and GetNonNullSize() will always return the same value.
   * If not, the array is "sparse".
   */
  virtual bool IsDense() = 0;

  //@{
  /**
   * Resizes the array to the given extents (number of dimensions and
   * size of each dimension).  Note that concrete implementations of
   * svtkArray may place constraints on the extents that they will
   * store, so you cannot assume that GetExtents() will always return
   * the same value passed to Resize().

   * The contents of the array are undefined after calling Resize() - you
   * should initialize its contents accordingly.  In particular,
   * dimension-labels will be undefined, dense array values will be
   * undefined, and sparse arrays will be empty.
   */
  void Resize(const CoordinateT i);
  void Resize(const CoordinateT i, const CoordinateT j);
  void Resize(const CoordinateT i, const CoordinateT j, const CoordinateT k);
  void Resize(const svtkArrayRange& i);
  void Resize(const svtkArrayRange& i, const svtkArrayRange& j);
  void Resize(const svtkArrayRange& i, const svtkArrayRange& j, const svtkArrayRange& k);
  void Resize(const svtkArrayExtents& extents);
  //@}

  /**
   * Returns the extent (valid coordinate range) along the given
   * dimension.
   */
  svtkArrayRange GetExtent(DimensionT dimension);
  /**
   * Returns the extents (the number of dimensions and size along each
   * dimension) of the array.
   */
  virtual const svtkArrayExtents& GetExtents() = 0;

  /**
   * Returns the number of dimensions stored in the array.  Note that
   * this is the same as calling GetExtents().GetDimensions().
   */
  DimensionT GetDimensions();

  /**
   * Returns the number of values stored in the array.  Note that this is
   * the same as calling GetExtents().GetSize(), and represents the
   * maximum number of values that could ever be stored using the current
   * extents.  This is equal to the number of values stored in a dense
   * array, but may be larger than the number of values stored in a
   * sparse array.
   */
  SizeT GetSize();

  /**
   * Returns the number of non-null values stored in the array.  Note
   * that this value will equal GetSize() for dense arrays, and will be
   * less-than-or-equal to GetSize() for sparse arrays.
   */
  virtual SizeT GetNonNullSize() = 0;

  /**
   * Sets the array name.
   */
  void SetName(const svtkStdString& name);
  /**
   * Returns the array name.
   */
  svtkStdString GetName();

  /**
   * Sets the label for the i-th array dimension.
   */
  void SetDimensionLabel(DimensionT i, const svtkStdString& label);

  /**
   * Returns the label for the i-th array dimension.
   */
  svtkStdString GetDimensionLabel(DimensionT i);

  /**
   * Returns the coordinates of the n-th value in the array, where n is
   * in the range [0, GetNonNullSize()).  Note that the order in which
   * coordinates are visited is undefined, but is guaranteed to match the
   * order in which values are visited using svtkTypedArray::GetValueN()
   * and svtkTypedArray::SetValueN().
   */
  virtual void GetCoordinatesN(const SizeT n, svtkArrayCoordinates& coordinates) = 0;

  //@{
  /**
   * Returns the value stored in the array at the given coordinates.
   * Note that the number of dimensions in the supplied coordinates must
   * match the number of dimensions in the array.
   */
  inline svtkVariant GetVariantValue(CoordinateT i);
  inline svtkVariant GetVariantValue(CoordinateT i, CoordinateT j);
  inline svtkVariant GetVariantValue(CoordinateT i, CoordinateT j, CoordinateT k);
  virtual svtkVariant GetVariantValue(const svtkArrayCoordinates& coordinates) = 0;
  //@}

  /**
   * Returns the n-th value stored in the array, where n is in the
   * range [0, GetNonNullSize()).  This is useful for efficiently
   * visiting every value in the array.  Note that the order in which
   * values are visited is undefined, but is guaranteed to match the
   * order used by svtkArray::GetCoordinatesN().
   */
  virtual svtkVariant GetVariantValueN(const SizeT n) = 0;

  //@{
  /**
   * Overwrites the value stored in the array at the given coordinates.
   * Note that the number of dimensions in the supplied coordinates must
   * match the number of dimensions in the array.
   */
  inline void SetVariantValue(CoordinateT i, const svtkVariant& value);
  inline void SetVariantValue(CoordinateT i, CoordinateT j, const svtkVariant& value);
  inline void SetVariantValue(CoordinateT i, CoordinateT j, CoordinateT k, const svtkVariant& value);
  virtual void SetVariantValue(const svtkArrayCoordinates& coordinates, const svtkVariant& value) = 0;
  //@}

  /**
   * Overwrites the n-th value stored in the array, where n is in the
   * range [0, GetNonNullSize()).  This is useful for efficiently
   * visiting every value in the array.  Note that the order in which
   * values are visited is undefined, but is guaranteed to match the
   * order used by svtkArray::GetCoordinatesN().
   */
  virtual void SetVariantValueN(const SizeT n, const svtkVariant& value) = 0;

  //@{
  /**
   * Overwrites a value with a value retrieved from another array.  Both
   * arrays must store the same data types.
   */
  virtual void CopyValue(svtkArray* source, const svtkArrayCoordinates& source_coordinates,
    const svtkArrayCoordinates& target_coordinates) = 0;
  virtual void CopyValue(
    svtkArray* source, const SizeT source_index, const svtkArrayCoordinates& target_coordinates) = 0;
  virtual void CopyValue(
    svtkArray* source, const svtkArrayCoordinates& source_coordinates, const SizeT target_index) = 0;
  //@}

  /**
   * Returns a new array that is a deep copy of this array.
   */
  virtual svtkArray* DeepCopy() = 0;

protected:
  svtkArray();
  ~svtkArray() override;

private:
  svtkArray(const svtkArray&) = delete;
  void operator=(const svtkArray&) = delete;

  /**
   * Stores the array name.
   */
  svtkStdString Name;

  /**
   * Implemented in concrete derivatives to update their storage
   * when the array is resized.
   */
  virtual void InternalResize(const svtkArrayExtents&) = 0;

  /**
   * Implemented in concrete derivatives to set dimension labels.
   */
  virtual void InternalSetDimensionLabel(DimensionT i, const svtkStdString& label) = 0;

  //@{
  /**
   * Implemented in concrete derivatives to get dimension labels.
   */
  virtual svtkStdString InternalGetDimensionLabel(DimensionT i) = 0;
  //@}
};

svtkVariant svtkArray::GetVariantValue(CoordinateT i)
{
  return this->GetVariantValue(svtkArrayCoordinates(i));
}

svtkVariant svtkArray::GetVariantValue(CoordinateT i, CoordinateT j)
{
  return this->GetVariantValue(svtkArrayCoordinates(i, j));
}

svtkVariant svtkArray::GetVariantValue(CoordinateT i, CoordinateT j, CoordinateT k)
{
  return this->GetVariantValue(svtkArrayCoordinates(i, j, k));
}

void svtkArray::SetVariantValue(CoordinateT i, const svtkVariant& value)
{
  this->SetVariantValue(svtkArrayCoordinates(i), value);
}

void svtkArray::SetVariantValue(CoordinateT i, CoordinateT j, const svtkVariant& value)
{
  this->SetVariantValue(svtkArrayCoordinates(i, j), value);
}

void svtkArray::SetVariantValue(CoordinateT i, CoordinateT j, CoordinateT k, const svtkVariant& value)
{
  this->SetVariantValue(svtkArrayCoordinates(i, j, k), value);
}

#endif

// SVTK-HeaderTest-Exclude: svtkArray.h
