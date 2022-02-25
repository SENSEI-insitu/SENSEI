/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGaussianRandomSequence.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
=========================================================================*/
#include "svtkGaussianRandomSequence.h"

// ----------------------------------------------------------------------------
svtkGaussianRandomSequence::svtkGaussianRandomSequence() = default;

// ----------------------------------------------------------------------------
svtkGaussianRandomSequence::~svtkGaussianRandomSequence() = default;

// ----------------------------------------------------------------------------
double svtkGaussianRandomSequence::GetScaledValue(double mean, double standardDeviation)
{
  return mean + standardDeviation * this->GetValue();
}

// ----------------------------------------------------------------------------
void svtkGaussianRandomSequence::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
