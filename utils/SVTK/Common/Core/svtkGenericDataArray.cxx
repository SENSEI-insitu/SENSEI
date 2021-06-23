/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericDataArray.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// We can't include svtkDataArrayPrivate.txx from a header, since it pulls in
// windows.h and creates a bunch of name collisions. So we compile the range
// lookup functions into this translation unit where we can encapsulate the
// header.
// We only compile the 64-bit integer versions of these. All others just
// reuse the double-precision svtkDataArray::GetRange version, since they
// won't lose precision.

#define SVTK_GDA_VALUERANGE_INSTANTIATING
#include "svtkGenericDataArray.h"

#include "svtkDataArrayPrivate.txx"

#include "svtkAOSDataArrayTemplate.h"
#include "svtkSOADataArrayTemplate.h"

#ifdef SVTK_USE_SCALED_SOA_ARRAYS
#include "svtkScaledSOADataArrayTemplate.h"
#endif

namespace svtkDataArrayPrivate
{
SVTK_INSTANTIATE_VALUERANGE_VALUETYPE(long)
SVTK_INSTANTIATE_VALUERANGE_VALUETYPE(unsigned long)
SVTK_INSTANTIATE_VALUERANGE_VALUETYPE(long long)
SVTK_INSTANTIATE_VALUERANGE_VALUETYPE(unsigned long long)
SVTK_INSTANTIATE_VALUERANGE_ARRAYTYPE(svtkDataArray, double)
} // namespace svtkDataArrayPrivate
