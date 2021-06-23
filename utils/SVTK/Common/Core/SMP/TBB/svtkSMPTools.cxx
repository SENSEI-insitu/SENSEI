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

#include "svtkCriticalSection.h"

#ifdef _MSC_VER
#pragma push_macro("__TBB_NO_IMPLICIT_LINKAGE")
#define __TBB_NO_IMPLICIT_LINKAGE 1
#endif

#include <tbb/task_scheduler_init.h>

#ifdef _MSC_VER
#pragma pop_macro("__TBB_NO_IMPLICIT_LINKAGE")
#endif

struct svtkSMPToolsInit
{
  tbb::task_scheduler_init Init;

  svtkSMPToolsInit(int numThreads)
    : Init(numThreads)
  {
  }
};

static bool svtkSMPToolsInitialized = 0;
static int svtkTBBNumSpecifiedThreads = 0;
static svtkSimpleCriticalSection svtkSMPToolsCS;

//--------------------------------------------------------------------------------
void svtkSMPTools::Initialize(int numThreads)
{
  svtkSMPToolsCS.Lock();
  if (!svtkSMPToolsInitialized)
  {
    // If numThreads <= 0, don't create a task_scheduler_init
    // and let TBB do the default thing.
    if (numThreads > 0)
    {
      static svtkSMPToolsInit aInit(numThreads);
      svtkTBBNumSpecifiedThreads = numThreads;
    }
    svtkSMPToolsInitialized = true;
  }
  svtkSMPToolsCS.Unlock();
}

//--------------------------------------------------------------------------------
int svtkSMPTools::GetEstimatedNumberOfThreads()
{
  return svtkTBBNumSpecifiedThreads ? svtkTBBNumSpecifiedThreads
                                   : tbb::task_scheduler_init::default_num_threads();
}
