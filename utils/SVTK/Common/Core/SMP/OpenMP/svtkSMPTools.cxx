/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSMPTools.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkSMPTools.h"

#include <omp.h>

#include <algorithm>

namespace
{
int svtkSMPNumberOfSpecifiedThreads = 0;
}

void svtkSMPTools::Initialize(int numThreads)
{
#pragma omp single
  if (numThreads)
  {
    svtkSMPNumberOfSpecifiedThreads = numThreads;
    omp_set_num_threads(numThreads);
  }
}

int svtkSMPTools::GetEstimatedNumberOfThreads()
{
  return svtk::detail::smp::GetNumberOfThreads();
}

int svtk::detail::smp::GetNumberOfThreads()
{
  return svtkSMPNumberOfSpecifiedThreads ? svtkSMPNumberOfSpecifiedThreads : omp_get_max_threads();
}

void svtk::detail::smp::svtkSMPTools_Impl_For_OpenMP(svtkIdType first, svtkIdType last, svtkIdType grain,
  ExecuteFunctorPtrType functorExecuter, void* functor)
{
  if (grain <= 0)
  {
    svtkIdType estimateGrain = (last - first) / (omp_get_max_threads() * 4);
    grain = (estimateGrain > 0) ? estimateGrain : 1;
  }

#pragma omp parallel for schedule(runtime)
  for (svtkIdType from = first; from < last; from += grain)
  {
    functorExecuter(functor, from, grain, last);
  }
}
