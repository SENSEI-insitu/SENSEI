/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBoxMuellerRandomSequence.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
=========================================================================*/
#include "svtkBoxMuellerRandomSequence.h"

#include "svtkMath.h"
#include "svtkMinimalStandardRandomSequence.h"
#include "svtkObjectFactory.h"
#include <cassert>

svtkStandardNewMacro(svtkBoxMuellerRandomSequence);

// ----------------------------------------------------------------------------
svtkBoxMuellerRandomSequence::svtkBoxMuellerRandomSequence()
{
  this->UniformSequence = svtkMinimalStandardRandomSequence::New();
  this->Value = 0;
}

// ----------------------------------------------------------------------------
svtkBoxMuellerRandomSequence::~svtkBoxMuellerRandomSequence()
{
  this->UniformSequence->Delete();
}

// ----------------------------------------------------------------------------
double svtkBoxMuellerRandomSequence::GetValue()
{
  return this->Value;
}

// ----------------------------------------------------------------------------
void svtkBoxMuellerRandomSequence::Next()
{
  this->UniformSequence->Next();
  double x = this->UniformSequence->GetValue();
  // Make sure x is in (0,1]
  while (x == 0.0)
  {
    this->UniformSequence->Next();
    x = this->UniformSequence->GetValue();
  }

  this->UniformSequence->Next();
  double y = this->UniformSequence->GetValue();

  // Make sure y is in (0,1]
  while (y == 0.0)
  {
    this->UniformSequence->Next();
    y = this->UniformSequence->GetValue();
  }

  this->Value = sqrt(-2.0 * log(x)) * cos(2.0 * svtkMath::Pi() * y);
}

// ----------------------------------------------------------------------------
svtkRandomSequence* svtkBoxMuellerRandomSequence::GetUniformSequence()
{
  assert("post: result_exists" && this->UniformSequence != nullptr);
  return this->UniformSequence;
}

// ----------------------------------------------------------------------------
// Description:
// Set the uniformly distributed sequence of random numbers.
// Default is a .
void svtkBoxMuellerRandomSequence::SetUniformSequence(svtkRandomSequence* uniformSequence)
{
  assert("pre: uniformSequence_exists" && uniformSequence != nullptr);

  if (this->UniformSequence != uniformSequence)
  {
    this->UniformSequence->Delete();
    this->UniformSequence = uniformSequence;
    this->UniformSequence->Register(this);
  }

  assert("post: assigned" && uniformSequence == this->GetUniformSequence());
}

// ----------------------------------------------------------------------------
void svtkBoxMuellerRandomSequence::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
