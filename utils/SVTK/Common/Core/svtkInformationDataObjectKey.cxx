/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationDataObjectKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationDataObjectKey.h"

#if defined(svtkCommonDataModel_ENABLED)
#include "../DataModel/svtkDataObject.h"
#endif

//----------------------------------------------------------------------------
svtkInformationDataObjectKey::svtkInformationDataObjectKey(const char* name, const char* location)
  : svtkInformationKey(name, location)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationDataObjectKey::~svtkInformationDataObjectKey() = default;

//----------------------------------------------------------------------------
void svtkInformationDataObjectKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
void svtkInformationDataObjectKey::Set(svtkInformation* info, svtkDataObject* value)
{
#if defined(svtkCommonDataModel_ENABLED)
  this->SetAsObjectBase(info, value);
#endif
}

//----------------------------------------------------------------------------
svtkDataObject* svtkInformationDataObjectKey::Get(svtkInformation* info)
{
#if defined(svtkCommonDataModel_ENABLED)
  return static_cast<svtkDataObject*>(this->GetAsObjectBase(info));
#else
  return 0;
#endif
}

//----------------------------------------------------------------------------
void svtkInformationDataObjectKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
{
  this->Set(to, this->Get(from));
}

//----------------------------------------------------------------------------
void svtkInformationDataObjectKey::Report(svtkInformation* info, svtkGarbageCollector* collector)
{
  this->ReportAsObjectBase(info, collector);
}
