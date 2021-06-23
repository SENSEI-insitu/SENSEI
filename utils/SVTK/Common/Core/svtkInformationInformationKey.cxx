/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationInformationKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationInformationKey.h"

#include "svtkInformation.h"

//----------------------------------------------------------------------------
svtkInformationInformationKey::svtkInformationInformationKey(const char* name, const char* location)
  : svtkInformationKey(name, location)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationInformationKey::~svtkInformationInformationKey() = default;

//----------------------------------------------------------------------------
void svtkInformationInformationKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
void svtkInformationInformationKey::Set(svtkInformation* info, svtkInformation* value)
{
  this->SetAsObjectBase(info, value);
}

//----------------------------------------------------------------------------
svtkInformation* svtkInformationInformationKey::Get(svtkInformation* info)
{
  return static_cast<svtkInformation*>(this->GetAsObjectBase(info));
}

//----------------------------------------------------------------------------
void svtkInformationInformationKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
{
  this->Set(to, this->Get(from));
}

//----------------------------------------------------------------------------
void svtkInformationInformationKey::DeepCopy(svtkInformation* from, svtkInformation* to)
{
  svtkInformation* toInfo = svtkInformation::New();
  toInfo->Copy(this->Get(from), 1);
  this->Set(to, toInfo);
  toInfo->Delete();
}
