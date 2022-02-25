/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTypedArray.h

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
 * @class   svtkTypedArray
 * @brief   Provides a type-specific interface to N-way arrays
 *
 *
 * svtkTypedArray provides an interface for retrieving and updating data in an
 * arbitrary-dimension array.  It derives from svtkArray and is templated on the
 * type of value stored in the array.
 *
 * Methods are provided for retrieving and updating array values based either
 * on their array coordinates, or on a 1-dimensional integer index.  The latter
 * approach can be used to iterate over the values in an array in arbitrary order,
 * which is useful when writing filters that operate efficiently on sparse arrays
 * and arrays that can have any number of dimensions.
 *
 * Special overloaded methods provide simple access for arrays with one, two, or
 * three dimensions.
 *
 * @sa
 * svtkArray, svtkDenseArray, svtkSparseArray
 *
 * @par Thanks:
 * Developed by Timothy M. Shead (tshead@sandia.gov) at Sandia National Laboratories.
 */

#ifndef svtkTypedArray_h
#define svtkTypedArray_h

#include "svtkArray.h"

class svtkArrayCoordinates;

template <typename T>
class svtkTypedArray : public svtkArray
{
public:
  svtkTemplateTypeMacro(svtkTypedArray<T>, svtkArray);
  typedef typename svtkArray::CoordinateT CoordinateT;
  typedef typename svtkArray::SizeT SizeT;

  using svtkArray::GetVariantValue;
  using svtkArray::SetVariantValue;

  void PrintSelf(ostream& os, svtkIndent indent) override;

  // svtkArray API
  svtkVariant GetVariantValue(const svtkArrayCoordinates& coordinates) override;
  svtkVariant GetVariantValueN(const SizeT n) override;
  void SetVariantValue(const svtkArrayCoordinates& coordinates, const svtkVariant& value) override;
  void SetVariantValueN(const SizeT n, const svtkVariant& value) override;
  void CopyValue(svtkArray* source, const svtkArrayCoordinates& source_coordinates,
    const svtkArrayCoordinates& target_coordinates) override;
  void CopyValue(svtkArray* source, const SizeT source_index,
    const svtkArrayCoordinates& target_coordinates) override;
  void CopyValue(svtkArray* source, const svtkArrayCoordinates& source_coordinates,
    const SizeT target_index) override;

  //@{
  /**
   * Returns the value stored in the array at the given coordinates.
   * Note that the number of dimensions in the supplied coordinates must
   * match the number of dimensions in the array.
   */
  virtual const T& GetValue(CoordinateT i) = 0;
  virtual const T& GetValue(CoordinateT i, CoordinateT j) = 0;
  virtual const T& GetValue(CoordinateT i, CoordinateT j, CoordinateT k) = 0;
  virtual const T& GetValue(const svtkArrayCoordinates& coordinates) = 0;
  //@}

  /**
   * Returns the n-th value stored in the array, where n is in the
   * range [0, GetNonNullSize()).  This is useful for efficiently
   * visiting every value in the array.  Note that the order in which
   * values are visited is undefined, but is guaranteed to match the
   * order used by svtkArray::GetCoordinatesN().
   */
  virtual const T& GetValueN(const SizeT n) = 0;

  //@{
  /**
   * Overwrites the value stored in the array at the given coordinates.
   * Note that the number of dimensions in the supplied coordinates must
   * match the number of dimensions in the array.
   */
  virtual void SetValue(CoordinateT i, const T& value) = 0;
  virtual void SetValue(CoordinateT i, CoordinateT j, const T& value) = 0;
  virtual void SetValue(CoordinateT i, CoordinateT j, CoordinateT k, const T& value) = 0;
  virtual void SetValue(const svtkArrayCoordinates& coordinates, const T& value) = 0;
  //@}

  /**
   * Overwrites the n-th value stored in the array, where n is in the
   * range [0, GetNonNullSize()).  This is useful for efficiently
   * visiting every value in the array.  Note that the order in which
   * values are visited is undefined, but is guaranteed to match the
   * order used by svtkArray::GetCoordinatesN().
   */
  virtual void SetValueN(const SizeT n, const T& value) = 0;

protected:
  svtkTypedArray() {}
  ~svtkTypedArray() override {}

private:
  svtkTypedArray(const svtkTypedArray&) = delete;
  void operator=(const svtkTypedArray&) = delete;
};

#include "svtkTypedArray.txx"

#endif

// SVTK-HeaderTest-Exclude: svtkTypedArray.h
