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

// Simple implementation that runs everything sequentially.

//--------------------------------------------------------------------------------
void svtkSMPTools::Initialize(int) {}

int svtkSMPTools::GetEstimatedNumberOfThreads()
{
  return 1;
}
