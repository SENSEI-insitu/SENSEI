/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayInterpolate.txx

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

#ifndef svtkArrayInterpolate_txx
#define svtkArrayInterpolate_txx

#include "svtkArrayExtentsList.h"
#include "svtkArrayWeights.h"

template <typename T>
void svtkInterpolate(svtkTypedArray<T>* source_array, const svtkArrayExtentsList& source_slices,
  const svtkArrayWeights& source_weights, const svtkArrayExtents& target_slice,
  svtkTypedArray<T>* target_array)
{
  if (!target_array->GetExtents().Contains(target_slice))
  {
    svtkGenericWarningMacro(<< "Target array does not contain target slice.");
    return;
  }

  if (source_slices.GetCount() != source_weights.GetCount())
  {
    svtkGenericWarningMacro(<< "Source slice and weight counts must match.");
    return;
  }

  for (int i = 0; i != source_slices.GetCount(); ++i)
  {
    if (!target_slice.SameShape(source_slices[i]))
    {
      svtkGenericWarningMacro(<< "Source and target slice shapes must match: " << source_slices[i]
                             << " versus " << target_slice);
      return;
    }
  }

  // Zero-out the target storage ...
  const svtkIdType n_begin = 0;
  const svtkIdType n_end = target_slice.GetSize();
  svtkArrayCoordinates target_coordinates;
  for (svtkIdType n = n_begin; n != n_end; ++n)
  {
    target_slice.GetLeftToRightCoordinatesN(n, target_coordinates);
    target_array->SetValue(target_coordinates, 0);
  }

  // Accumulate results ...
  svtkArrayCoordinates source_coordinates;
  for (svtkIdType n = n_begin; n != n_end; ++n)
  {
    target_slice.GetLeftToRightCoordinatesN(n, target_coordinates);
    for (int source = 0; source != source_slices.GetCount(); ++source)
    {
      source_slices[source].GetLeftToRightCoordinatesN(n, source_coordinates);
      target_array->SetValue(target_coordinates,
        target_array->GetValue(target_coordinates) +
          (source_array->GetValue(source_coordinates) * source_weights[source]));
    }
  }
}

#endif
