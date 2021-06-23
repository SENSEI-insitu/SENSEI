/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArraySort.h

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
 * @class   svtkArraySort
 * @brief   Controls sorting of sparse array coordinates.
 *
 *
 * svtkArraySort stores an ordered set of dimensions along which the
 * values stored in a sparse array should be sorted.
 *
 * Convenience constructors are provided for specifying one, two, and
 * three dimensions.  To sort along more than three dimensions, use the
 * default constructor, SetDimensions(), and operator[] to assign each
 * dimension to be sorted.
 *
 * @sa
 * svtkSparseArray
 *
 * @par Thanks:
 * Developed by Timothy M. Shead (tshead@sandia.gov) at Sandia National
 * Laboratories.
 */

#ifndef svtkArraySort_h
#define svtkArraySort_h

#include "svtkArrayCoordinates.h"
#include "svtkCommonCoreModule.h" // For export macro
#include "svtkSystemIncludes.h"
#include <vector>

class SVTKCOMMONCORE_EXPORT svtkArraySort
{
public:
  typedef svtkArrayCoordinates::DimensionT DimensionT;

  /**
   * Create an empty set of dimensions.  Use SetDimensions() and
   * operator[] to populate them.
   */
  svtkArraySort();

  /**
   * Sorts an array along one dimension.
   */
  explicit svtkArraySort(DimensionT i);

  /**
   * Sorts an array along two dimensions.
   */
  svtkArraySort(DimensionT i, DimensionT j);

  /**
   * Sorts an array along three dimensions.
   */
  svtkArraySort(DimensionT i, DimensionT j, DimensionT k);

  /**
   * Return the number of dimensions for sorting.
   */
  DimensionT GetDimensions() const;

  /**
   * Set the number of dimensions to be sorted.  Note that this method
   * resets every dimension to zero, so you must set every dimension
   * explicitly using operator[] after calling SetDimensions().
   */
  void SetDimensions(DimensionT dimensions);

  /**
   * Returns the i-th dimension to be sorted.
   */
  DimensionT& operator[](DimensionT i);

  /**
   * Returns the i-th dimension to be sorted.
   */
  const DimensionT& operator[](DimensionT i) const;

  /**
   * Equality comparison
   */
  bool operator==(const svtkArraySort& rhs) const;

  /**
   * Inequality comparison
   */
  bool operator!=(const svtkArraySort& rhs) const;

  /**
   * Serialization
   */
  SVTKCOMMONCORE_EXPORT friend ostream& operator<<(ostream& stream, const svtkArraySort& rhs);

private:
  std::vector<DimensionT> Storage;
};

#endif

// SVTK-HeaderTest-Exclude: svtkArraySort.h
