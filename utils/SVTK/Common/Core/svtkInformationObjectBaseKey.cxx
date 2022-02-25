/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationObjectBaseKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationObjectBaseKey.h"

#include "svtkInformation.h" // For svtkErrorWithObjectMacro

//----------------------------------------------------------------------------
svtkInformationObjectBaseKey ::svtkInformationObjectBaseKey(
  const char* name, const char* location, const char* requiredClass)
  : svtkInformationKey(name, location)
{
  svtkCommonInformationKeyManager::Register(this);

  this->RequiredClass = nullptr;
  this->SetRequiredClass(requiredClass);
}

//----------------------------------------------------------------------------
svtkInformationObjectBaseKey::~svtkInformationObjectBaseKey()
{
  delete[] this->RequiredClass;
}

//----------------------------------------------------------------------------
void svtkInformationObjectBaseKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
void svtkInformationObjectBaseKey::Set(svtkInformation* info, svtkObjectBase* value)
{
  if (value && this->RequiredClass && !value->IsA(this->RequiredClass))
  {
    svtkErrorWithObjectMacro(info,
      "Cannot store object of type " << value->GetClassName() << " with key " << this->Location
                                     << "::" << this->Name << " which requires objects of type "
                                     << this->RequiredClass << ".  Removing the key instead.");
    this->SetAsObjectBase(info, nullptr);
    return;
  }
  this->SetAsObjectBase(info, value);
}

//----------------------------------------------------------------------------
svtkObjectBase* svtkInformationObjectBaseKey::Get(svtkInformation* info)
{
  return this->GetAsObjectBase(info);
}

//----------------------------------------------------------------------------
void svtkInformationObjectBaseKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
{
  this->Set(to, this->Get(from));
}

//----------------------------------------------------------------------------
void svtkInformationObjectBaseKey::Report(svtkInformation* info, svtkGarbageCollector* collector)
{
  this->ReportAsObjectBase(info, collector);
}
