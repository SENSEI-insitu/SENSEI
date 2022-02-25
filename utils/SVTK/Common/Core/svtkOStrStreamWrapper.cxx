/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkOStrStreamWrapper.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkSystemIncludes.h" // Cannot include svtkOStrStreamWrapper.h directly.

// Need strcpy.
#include <string>

#include <sstream>

using std::ostringstream;

//----------------------------------------------------------------------------
svtkOStrStreamWrapper::svtkOStrStreamWrapper()
  : svtkOStreamWrapper(*(new ostringstream))
{
  this->Result = nullptr;
  this->Frozen = 0;
}

//----------------------------------------------------------------------------
svtkOStrStreamWrapper::~svtkOStrStreamWrapper()
{
  if (!this->Frozen)
  {
    delete[] this->Result;
  }
  delete &this->ostr;
}

//----------------------------------------------------------------------------
char* svtkOStrStreamWrapper::str()
{
  if (!this->Result)
  {
    std::string s = static_cast<ostringstream*>(&this->ostr)->str();
    this->Result = new char[s.length() + 1];
    strcpy(this->Result, s.c_str());
    this->freeze();
  }
  return this->Result;
}

//----------------------------------------------------------------------------
svtkOStrStreamWrapper* svtkOStrStreamWrapper::rdbuf()
{
  return this;
}

//----------------------------------------------------------------------------
void svtkOStrStreamWrapper::freeze()
{
  this->freeze(1);
}

//----------------------------------------------------------------------------
void svtkOStrStreamWrapper::freeze(int f)
{
  this->Frozen = f;
}
