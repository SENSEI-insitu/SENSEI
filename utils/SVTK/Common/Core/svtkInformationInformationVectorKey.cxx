/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationInformationVectorKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationInformationVectorKey.h"

#include "svtkInformation.h"
#include "svtkInformationVector.h"

//----------------------------------------------------------------------------
svtkInformationInformationVectorKey::svtkInformationInformationVectorKey(
  const char* name, const char* location)
  : svtkInformationKey(name, location)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationInformationVectorKey::~svtkInformationInformationVectorKey() = default;

//----------------------------------------------------------------------------
void svtkInformationInformationVectorKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
void svtkInformationInformationVectorKey::Set(svtkInformation* info, svtkInformationVector* value)
{
  this->SetAsObjectBase(info, value);
}

//----------------------------------------------------------------------------
svtkInformationVector* svtkInformationInformationVectorKey::Get(svtkInformation* info)
{
  return static_cast<svtkInformationVector*>(this->GetAsObjectBase(info));
}

//----------------------------------------------------------------------------
void svtkInformationInformationVectorKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
{
  this->Set(to, this->Get(from));
}

//----------------------------------------------------------------------------
void svtkInformationInformationVectorKey::DeepCopy(svtkInformation* from, svtkInformation* to)
{
  svtkInformationVector* fromVector = this->Get(from);
  svtkInformationVector* toVector = svtkInformationVector::New();
  svtkInformation* toInfo;
  int i;

  for (i = 0; i < fromVector->GetNumberOfInformationObjects(); i++)
  {
    toInfo = svtkInformation::New();
    toInfo->Copy(fromVector->GetInformationObject(i), 1);
    toVector->Append(toInfo);
    toInfo->FastDelete();
  }
  this->Set(to, toVector);
  toVector->FastDelete();
}

//----------------------------------------------------------------------------
void svtkInformationInformationVectorKey::Report(
  svtkInformation* info, svtkGarbageCollector* collector)
{
  this->ReportAsObjectBase(info, collector);
}
