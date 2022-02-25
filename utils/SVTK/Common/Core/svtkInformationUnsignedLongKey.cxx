/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationUnsignedLongKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationUnsignedLongKey.h"

#include "svtkInformation.h"

//----------------------------------------------------------------------------
svtkInformationUnsignedLongKey::svtkInformationUnsignedLongKey(const char* name, const char* location)
  : svtkInformationKey(name, location)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationUnsignedLongKey::~svtkInformationUnsignedLongKey() = default;

//----------------------------------------------------------------------------
void svtkInformationUnsignedLongKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
class svtkInformationUnsignedLongValue : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkInformationUnsignedLongValue, svtkObjectBase);
  unsigned long Value;
};

//----------------------------------------------------------------------------
void svtkInformationUnsignedLongKey::Set(svtkInformation* info, unsigned long value)
{
  if (svtkInformationUnsignedLongValue* oldv =
        static_cast<svtkInformationUnsignedLongValue*>(this->GetAsObjectBase(info)))
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
    svtkInformationUnsignedLongValue* v = new svtkInformationUnsignedLongValue;
    v->InitializeObjectBase();
    v->Value = value;
    this->SetAsObjectBase(info, v);
    v->Delete();
  }
}

//----------------------------------------------------------------------------
unsigned long svtkInformationUnsignedLongKey::Get(svtkInformation* info)
{
  svtkInformationUnsignedLongValue* v =
    static_cast<svtkInformationUnsignedLongValue*>(this->GetAsObjectBase(info));
  return v ? v->Value : 0;
}

//----------------------------------------------------------------------------
void svtkInformationUnsignedLongKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
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
void svtkInformationUnsignedLongKey::Print(ostream& os, svtkInformation* info)
{
  // Print the value.
  if (this->Has(info))
  {
    os << this->Get(info);
  }
}

//----------------------------------------------------------------------------
unsigned long* svtkInformationUnsignedLongKey::GetWatchAddress(svtkInformation* info)
{
  if (svtkInformationUnsignedLongValue* v =
        static_cast<svtkInformationUnsignedLongValue*>(this->GetAsObjectBase(info)))
  {
    return &v->Value;
  }
  return nullptr;
}
