/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationIdTypeKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationIdTypeKey.h"

#include "svtkInformation.h"

//----------------------------------------------------------------------------
svtkInformationIdTypeKey::svtkInformationIdTypeKey(const char* name, const char* location)
  : svtkInformationKey(name, location)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationIdTypeKey::~svtkInformationIdTypeKey() = default;

//----------------------------------------------------------------------------
void svtkInformationIdTypeKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
class svtkInformationIdTypeValue : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkInformationIdTypeValue, svtkObjectBase);
  svtkIdType Value;
};

//----------------------------------------------------------------------------
void svtkInformationIdTypeKey::Set(svtkInformation* info, svtkIdType value)
{
  if (svtkInformationIdTypeValue* oldv =
        static_cast<svtkInformationIdTypeValue*>(this->GetAsObjectBase(info)))
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
    svtkInformationIdTypeValue* v = new svtkInformationIdTypeValue;
    v->InitializeObjectBase();
    v->Value = value;
    this->SetAsObjectBase(info, v);
    v->Delete();
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkInformationIdTypeKey::Get(svtkInformation* info)
{
  svtkInformationIdTypeValue* v =
    static_cast<svtkInformationIdTypeValue*>(this->GetAsObjectBase(info));
  return v ? v->Value : 0;
}

//----------------------------------------------------------------------------
void svtkInformationIdTypeKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
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
void svtkInformationIdTypeKey::Print(ostream& os, svtkInformation* info)
{
  // Print the value.
  if (this->Has(info))
  {
    os << this->Get(info);
  }
}

//----------------------------------------------------------------------------
svtkIdType* svtkInformationIdTypeKey::GetWatchAddress(svtkInformation* info)
{
  if (svtkInformationIdTypeValue* v =
        static_cast<svtkInformationIdTypeValue*>(this->GetAsObjectBase(info)))
  {
    return &v->Value;
  }
  return nullptr;
}
