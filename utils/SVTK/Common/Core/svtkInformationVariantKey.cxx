/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationVariantKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationVariantKey.h"

#include "svtkInformation.h"
#include "svtkVariant.h"

//----------------------------------------------------------------------------
svtkInformationVariantKey::svtkInformationVariantKey(const char* name, const char* location)
  : svtkInformationKey(name, location)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationVariantKey::~svtkInformationVariantKey() = default;

//----------------------------------------------------------------------------
void svtkInformationVariantKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
class svtkInformationVariantValue : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkInformationVariantValue, svtkObjectBase);
  svtkVariant Value;
  static svtkVariant Invalid;
};

svtkVariant svtkInformationVariantValue::Invalid;

//----------------------------------------------------------------------------
void svtkInformationVariantKey::Set(svtkInformation* info, const svtkVariant& value)
{
  if (svtkInformationVariantValue* oldv =
        static_cast<svtkInformationVariantValue*>(this->GetAsObjectBase(info)))
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
    svtkInformationVariantValue* v = new svtkInformationVariantValue;
    v->InitializeObjectBase();
    v->Value = value;
    this->SetAsObjectBase(info, v);
    v->Delete();
  }
}

//----------------------------------------------------------------------------
const svtkVariant& svtkInformationVariantKey::Get(svtkInformation* info)
{
  svtkInformationVariantValue* v =
    static_cast<svtkInformationVariantValue*>(this->GetAsObjectBase(info));
  return v ? v->Value : svtkInformationVariantValue::Invalid;
}

//----------------------------------------------------------------------------
void svtkInformationVariantKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
{
  if (this->Has(from))
  {
    this->Set(to, this->Get(from));
  }
  else
  {
    this->SetAsObjectBase(to, nullptr); // doesn't exist in from, so remove the key
  }
}

//----------------------------------------------------------------------------
void svtkInformationVariantKey::Print(ostream& os, svtkInformation* info)
{
  // Print the value.
  if (this->Has(info))
  {
    os << this->Get(info);
  }
}

//----------------------------------------------------------------------------
svtkVariant* svtkInformationVariantKey::GetWatchAddress(svtkInformation* info)
{
  if (svtkInformationVariantValue* v =
        static_cast<svtkInformationVariantValue*>(this->GetAsObjectBase(info)))
  {
    return &v->Value;
  }
  return nullptr;
}
