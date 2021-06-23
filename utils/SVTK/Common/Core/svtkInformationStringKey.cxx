/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationStringKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationStringKey.h"

#include "svtkInformation.h"

#include <string>

//----------------------------------------------------------------------------
svtkInformationStringKey::svtkInformationStringKey(const char* name, const char* location)
  : svtkInformationKey(name, location)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationStringKey::~svtkInformationStringKey() = default;

//----------------------------------------------------------------------------
void svtkInformationStringKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
class svtkInformationStringValue : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkInformationStringValue, svtkObjectBase);
  std::string Value;
};

//----------------------------------------------------------------------------
void svtkInformationStringKey::Set(svtkInformation* info, const char* value)
{
  if (value)
  {
    if (svtkInformationStringValue* oldv =
          static_cast<svtkInformationStringValue*>(this->GetAsObjectBase(info)))
    {
      if (oldv->Value != value)
      {
        // Replace the existing value.
        oldv->Value = value;
        // Since this sets a value without call SetAsObjectBase(),
        // the info has to be modified here (instead of
        // svtkInformation::SetAsObjectBase()
        info->Modified(this);
      }
    }
    else
    {
      // Allocate a new value.
      svtkInformationStringValue* v = new svtkInformationStringValue;
      v->InitializeObjectBase();
      v->Value = value;
      this->SetAsObjectBase(info, v);
      v->Delete();
    }
  }
  else
  {
    this->SetAsObjectBase(info, nullptr);
  }
}

//----------------------------------------------------------------------------
void svtkInformationStringKey::Set(svtkInformation* info, const std::string& s)
{
  this->Set(info, s.c_str());
}

//----------------------------------------------------------------------------
const char* svtkInformationStringKey::Get(svtkInformation* info)
{
  svtkInformationStringValue* v =
    static_cast<svtkInformationStringValue*>(this->GetAsObjectBase(info));
  return v ? v->Value.c_str() : nullptr;
}

//----------------------------------------------------------------------------
void svtkInformationStringKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
{
  this->Set(to, this->Get(from));
}

//----------------------------------------------------------------------------
void svtkInformationStringKey::Print(ostream& os, svtkInformation* info)
{
  // Print the value.
  if (this->Has(info))
  {
    os << this->Get(info);
  }
}
