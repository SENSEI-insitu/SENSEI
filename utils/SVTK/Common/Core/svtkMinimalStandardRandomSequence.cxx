/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMinimalStandardRandomSequence.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
=========================================================================*/
#include "svtkMinimalStandardRandomSequence.h"

#include "svtkObjectFactory.h"
#include <cassert>

svtkStandardNewMacro(svtkMinimalStandardRandomSequence);

const int SVTK_K_A = 16807;
const int SVTK_K_M = 2147483647; // Mersenne prime 2^(31)-1
const int SVTK_K_Q = 127773;     // M/A
const int SVTK_K_R = 2836;       // M%A

// ----------------------------------------------------------------------------
svtkMinimalStandardRandomSequence::svtkMinimalStandardRandomSequence()
{
  this->Seed = 1;
}

// ----------------------------------------------------------------------------
svtkMinimalStandardRandomSequence::~svtkMinimalStandardRandomSequence() = default;

// ----------------------------------------------------------------------------
void svtkMinimalStandardRandomSequence::SetSeedOnly(int value)
{
  this->Seed = value;

  // fit the seed to the valid range [1,2147483646]
  if (this->Seed < 1)
  {
    this->Seed += 2147483646;
  }
  else
  {
    if (this->Seed == 2147483647)
    {
      this->Seed = 1;
    }
  }
}

// ----------------------------------------------------------------------------
void svtkMinimalStandardRandomSequence::SetSeed(int value)
{
  this->SetSeedOnly(value);

  // the first random number after setting the seed is proportional to the
  // seed value. To help solve this, call Next() a few times.
  // This doesn't ruin the repeatability of Next().
  this->Next();
  this->Next();
  this->Next();
}

// ----------------------------------------------------------------------------
int svtkMinimalStandardRandomSequence::GetSeed()
{
  return this->Seed;
}

// ----------------------------------------------------------------------------
double svtkMinimalStandardRandomSequence::GetValue()
{
  double result = static_cast<double>(this->Seed) / SVTK_K_M;

  assert("post: unit_range" && result >= 0.0 && result <= 1.0);
  return result;
}

// ----------------------------------------------------------------------------
void svtkMinimalStandardRandomSequence::Next()
{
  int hi = this->Seed / SVTK_K_Q;
  int lo = this->Seed % SVTK_K_Q;
  this->Seed = SVTK_K_A * lo - SVTK_K_R * hi;
  if (this->Seed <= 0)
  {
    this->Seed += SVTK_K_M;
  }
}

// ----------------------------------------------------------------------------
double svtkMinimalStandardRandomSequence::GetRangeValue(double rangeMin, double rangeMax)
{
  double result;
  if (rangeMin == rangeMax)
  {
    result = rangeMin;
  }
  else
  {
    result = rangeMin + this->GetValue() * (rangeMax - rangeMin);
  }
  assert("post: valid_result" &&
    ((rangeMin <= rangeMax && result >= rangeMin && result <= rangeMax) ||
      (rangeMax <= rangeMin && result >= rangeMax && result <= rangeMin)));
  return result;
}

// ----------------------------------------------------------------------------
void svtkMinimalStandardRandomSequence::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
