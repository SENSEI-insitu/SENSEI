/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayWeights.h

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
 * @class   svtkArrayWeights
 * @brief   Stores a collection of weighting factors.
 *
 *
 * svtkArrayWeights provides storage for a collection of weights to be
 * used when merging / interpolating N-way arrays.  Convenience
 * constructors are provided for working with one, two, three, and four
 * weighting factors.  For arbitrary collections of weights, use
 * SetCount() and operator[] to assign values.
 *
 * svtkArrayWeights is most commonly used with the svtkInterpolate()
 * function to compute weighted sums of svtkArray objects.
 *
 * @sa
 * svtkArray, svtkArraySlices
 *
 * @par Thanks:
 * Developed by Timothy M. Shead (tshead@sandia.gov) at Sandia National
 * Laboratories.
 */

#ifndef svtkArrayWeights_h
#define svtkArrayWeights_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkSystemIncludes.h"

class svtkArrayWeightsStorage; // pimpl

class SVTKCOMMONCORE_EXPORT svtkArrayWeights
{
public:
  /**
   * Create an empty collection of weights
   */
  svtkArrayWeights();

  /**
   * Copy the weights from another object.
   */
  svtkArrayWeights(const svtkArrayWeights& other);

  /**
   * Create a collection containing one weight.
   */
  svtkArrayWeights(double i);

  /**
   * Create a collection containing two weights.
   */
  svtkArrayWeights(double i, double j);

  /**
   * Create a collection containing three weights.
   */
  svtkArrayWeights(double i, double j, double k);

  /**
   * Create a collection containing four weights.
   */
  svtkArrayWeights(double i, double j, double k, double l);

  /**
   * Destructor.
   */
  ~svtkArrayWeights();

  /**
   * Returns the number of weights stored in this container.
   */
  svtkIdType GetCount() const;

  /**
   * Sets the number of weights stored in this container.  Note that each
   * weight will be reset to 0.0 after calling SetCount(), use operator[]
   * to assign the desired value for each weight.
   */
  void SetCount(svtkIdType count);

  /**
   * Accesses the i-th weight in the collection.
   */
  double& operator[](svtkIdType);

  /**
   * Accesses the i-th weight in the collection.
   */
  const double& operator[](svtkIdType) const;

  /**
   * Assignment operator.
   */
  svtkArrayWeights& operator=(const svtkArrayWeights& other);

protected:
  svtkArrayWeightsStorage* Storage;
};

#endif

// SVTK-HeaderTest-Exclude: svtkArrayWeights.h
