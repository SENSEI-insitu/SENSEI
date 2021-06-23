/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayExtentsList.h

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
 * @class   svtkArrayExtentsList
 * @brief   Stores a collection of svtkArrayExtents objects.
 *
 *
 * svtkArrayExtentsList provides storage for a collection of svtkArrayExtents
 * instances.  Constructors are provided for creating collections
 * containing one, two, three, or four slices.  To work with larger
 * numbers of slices, use the default constructor, the SetCount() method,
 * and operator[].
 *
 * svtkArrayExtentsList is most commonly used with the svtkInterpolate()
 * function, which is used to computed weighted sums of svtkArray slices.
 *
 * @sa
 * svtkArray, svtkExtents
 *
 * @par Thanks:
 * Developed by Timothy M. Shead (tshead@sandia.gov) at Sandia National
 * Laboratories.
 */

#ifndef svtkArrayExtentsList_h
#define svtkArrayExtentsList_h

#include "svtkArrayExtents.h"
#include "svtkCommonCoreModule.h" // For export macro
#include <vector>                // STL Header

class SVTKCOMMONCORE_EXPORT svtkArrayExtentsList
{
public:
  /**
   * Creates an empty collection of slices.
   */
  svtkArrayExtentsList();

  /**
   * Creates a collection containing one slice.
   */
  svtkArrayExtentsList(const svtkArrayExtents& i);

  /**
   * Creates a collection containing two slices.
   */
  svtkArrayExtentsList(const svtkArrayExtents& i, const svtkArrayExtents& j);

  /**
   * Creates a collection containing three slices.
   */
  svtkArrayExtentsList(const svtkArrayExtents& i, const svtkArrayExtents& j, const svtkArrayExtents& k);

  /**
   * Creates a collection containing four slices.
   */
  svtkArrayExtentsList(const svtkArrayExtents& i, const svtkArrayExtents& j, const svtkArrayExtents& k,
    const svtkArrayExtents& l);

  /**
   * Returns the number of slices stored in this collection.
   */
  svtkIdType GetCount() const;

  /**
   * Sets the number of extents stored in this collection.  Note: all
   * extents will be empty after calling SetCount(), use operator[]
   * to assign extents to each item in the collection.
   */
  void SetCount(svtkIdType count);

  /**
   * Accesses the i-th slice.
   */
  svtkArrayExtents& operator[](svtkIdType i);

  /**
   * Accesses the i-th slice.
   */
  const svtkArrayExtents& operator[](svtkIdType i) const;

private:
  std::vector<svtkArrayExtents> Storage;
};

#endif

// SVTK-HeaderTest-Exclude: svtkArrayExtentsList.h
