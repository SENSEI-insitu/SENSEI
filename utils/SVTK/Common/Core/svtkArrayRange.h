/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayRange.h

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
 * @class   svtkArrayRange
 * @brief   Stores a half-open range of array coordinates.
 *
 *
 * svtkArrayRange stores a half-open range of array coordinates along a
 * single dimension of a svtkArraySlice object.
 *
 * @sa
 * svtkArray, svtkArrayRange
 *
 * @par Thanks:
 * Developed by Timothy M. Shead (tshead@sandia.gov) at Sandia National
 * Laboratories.
 */

#ifndef svtkArrayRange_h
#define svtkArrayRange_h

#include "svtkArrayCoordinates.h"
#include "svtkCommonCoreModule.h" // For export macro
#include "svtkSystemIncludes.h"

class SVTKCOMMONCORE_EXPORT svtkArrayRange
{
public:
  typedef svtkArrayCoordinates::CoordinateT CoordinateT;

  /**
   * Creates an empty range.
   */
  svtkArrayRange();

  /**
   * Creates a half-open range [begin, end).
   * Note that begin must be <= end,
   * if not, creates the empty range [begin, begin).
   */
  svtkArrayRange(CoordinateT begin, CoordinateT end);

  /**
   * Returns the beginning of the range
   */
  CoordinateT GetBegin() const;

  /**
   * Returns one-past-the-end of the range
   */
  CoordinateT GetEnd() const;

  /**
   * Returns the size of the range (the distance End - Begin).
   */
  CoordinateT GetSize() const;

  /**
   * Returns true iff the given range is a non-overlapping subset of this
   * range.
   */
  bool Contains(const svtkArrayRange& range) const;

  /**
   * Returns true iff the given coordinate falls within this range.
   */
  bool Contains(const CoordinateT coordinate) const;

  //@{
  /**
   * Equality comparisons.
   */
  SVTKCOMMONCORE_EXPORT friend bool operator==(const svtkArrayRange& lhs, const svtkArrayRange& rhs);
  SVTKCOMMONCORE_EXPORT friend bool operator!=(const svtkArrayRange& lhs, const svtkArrayRange& rhs);
  //@}

  /**
   * Serialization.
   */
  SVTKCOMMONCORE_EXPORT friend ostream& operator<<(ostream& stream, const svtkArrayRange& rhs);

private:
  /**
   * Stores the beginning of the range.
   */
  CoordinateT Begin;

  //@{
  /**
   * Stores one-past-the-end of the range.
   */
  CoordinateT End;
  //@}
};

#endif
// SVTK-HeaderTest-Exclude: svtkArrayRange.h
