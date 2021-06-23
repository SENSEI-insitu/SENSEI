/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGarbageCollectorManager.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkGarbageCollectorManager.h"

#include "svtkGarbageCollector.h"

// Must NOT be initialized.  Default initialization to zero is
// necessary.
static unsigned int svtkGarbageCollectorManagerCount;

svtkGarbageCollectorManager::svtkGarbageCollectorManager()
{
  if (++svtkGarbageCollectorManagerCount == 1)
  {
    svtkGarbageCollector::ClassInitialize();
  }
}

svtkGarbageCollectorManager::~svtkGarbageCollectorManager()
{
  if (--svtkGarbageCollectorManagerCount == 0)
  {
    svtkGarbageCollector::ClassFinalize();
  }
}
