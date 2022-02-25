/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationKey.h"
#include "svtkInformationKeyLookup.h"

#include "svtkDebugLeaks.h"
#include "svtkInformation.h"

class svtkInformationKeyToInformationFriendship
{
public:
  static void SetAsObjectBase(svtkInformation* info, svtkInformationKey* key, svtkObjectBase* value)
  {
    info->SetAsObjectBase(key, value);
  }
  static const svtkObjectBase* GetAsObjectBase(
    const svtkInformation* info, const svtkInformationKey* key)
  {
    return info->GetAsObjectBase(key);
  }
  static svtkObjectBase* GetAsObjectBase(svtkInformation* info, svtkInformationKey* key)
  {
    return info->GetAsObjectBase(key);
  }
  static void ReportAsObjectBase(
    svtkInformation* info, svtkInformationKey* key, svtkGarbageCollector* collector)
  {
    info->ReportAsObjectBase(key, collector);
  }
};

//----------------------------------------------------------------------------
svtkInformationKey::svtkInformationKey(const char* name, const char* location)
{
  // Save the name and location.
  this->Name = nullptr;
  this->SetName(name);

  this->Location = nullptr;
  this->SetLocation(location);

  svtkInformationKeyLookup::RegisterKey(this, name, location);
}

//----------------------------------------------------------------------------
svtkInformationKey::~svtkInformationKey()
{
  this->SetReferenceCount(0);
  this->SetName(nullptr);
  this->SetLocation(nullptr);
}

//----------------------------------------------------------------------------
void svtkInformationKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
void svtkInformationKey::Register(svtkObjectBase*) {}

//----------------------------------------------------------------------------
void svtkInformationKey::UnRegister(svtkObjectBase*) {}

//----------------------------------------------------------------------------
const char* svtkInformationKey::GetName()
{
  return this->Name;
}

//----------------------------------------------------------------------------
const char* svtkInformationKey::GetLocation()
{
  return this->Location;
}

//----------------------------------------------------------------------------
void svtkInformationKey::SetAsObjectBase(svtkInformation* info, svtkObjectBase* value)
{
  svtkInformationKeyToInformationFriendship::SetAsObjectBase(info, this, value);
}

//----------------------------------------------------------------------------
svtkObjectBase* svtkInformationKey::GetAsObjectBase(svtkInformation* info)
{
  return svtkInformationKeyToInformationFriendship::GetAsObjectBase(info, this);
}

//----------------------------------------------------------------------------
const svtkObjectBase* svtkInformationKey::GetAsObjectBase(svtkInformation* info) const
{
  return svtkInformationKeyToInformationFriendship::GetAsObjectBase(info, this);
}

//----------------------------------------------------------------------------
int svtkInformationKey::Has(svtkInformation* info)
{
  return this->GetAsObjectBase(info) ? 1 : 0;
}

//----------------------------------------------------------------------------
void svtkInformationKey::Remove(svtkInformation* info)
{
  this->SetAsObjectBase(info, nullptr);
}

//----------------------------------------------------------------------------
void svtkInformationKey::Report(svtkInformation*, svtkGarbageCollector*)
{
  // Report nothing by default.
}

//----------------------------------------------------------------------------
void svtkInformationKey::Print(svtkInformation* info)
{
  this->Print(cout, info);
}

//----------------------------------------------------------------------------
void svtkInformationKey::Print(ostream& os, svtkInformation* info)
{
  // Just print the value type and pointer by default.
  if (svtkObjectBase* value = this->GetAsObjectBase(info))
  {
    os << value->GetClassName() << "(" << value << ")";
  }
}

//----------------------------------------------------------------------------
void svtkInformationKey::ReportAsObjectBase(svtkInformation* info, svtkGarbageCollector* collector)
{
  svtkInformationKeyToInformationFriendship::ReportAsObjectBase(info, this, collector);
}

//----------------------------------------------------------------------------
void svtkInformationKey::ConstructClass(const char*) {}
