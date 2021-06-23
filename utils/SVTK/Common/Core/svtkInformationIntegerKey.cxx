/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationIntegerKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationIntegerKey.h"

#include "svtkInformation.h"

//----------------------------------------------------------------------------
svtkInformationIntegerKey::svtkInformationIntegerKey(const char* name, const char* location)
  : svtkInformationKey(name, location)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationIntegerKey::~svtkInformationIntegerKey() = default;

//----------------------------------------------------------------------------
void svtkInformationIntegerKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
class svtkInformationIntegerValue : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkInformationIntegerValue, svtkObjectBase);
  int Value;
};

//----------------------------------------------------------------------------
void svtkInformationIntegerKey::Set(svtkInformation* info, int value)
{
  if (svtkInformationIntegerValue* oldv =
        static_cast<svtkInformationIntegerValue*>(this->GetAsObjectBase(info)))
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
    svtkInformationIntegerValue* v = new svtkInformationIntegerValue;
    v->InitializeObjectBase();
    v->Value = value;
    this->SetAsObjectBase(info, v);
    v->Delete();
  }
}

//----------------------------------------------------------------------------
int svtkInformationIntegerKey::Get(svtkInformation* info)
{
  svtkInformationIntegerValue* v =
    static_cast<svtkInformationIntegerValue*>(this->GetAsObjectBase(info));
  return v ? v->Value : 0;
}

//----------------------------------------------------------------------------
void svtkInformationIntegerKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
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
void svtkInformationIntegerKey::Print(ostream& os, svtkInformation* info)
{
  // Print the value.
  if (this->Has(info))
  {
    os << this->Get(info);
  }
}

//----------------------------------------------------------------------------
int* svtkInformationIntegerKey::GetWatchAddress(svtkInformation* info)
{
  if (svtkInformationIntegerValue* v =
        static_cast<svtkInformationIntegerValue*>(this->GetAsObjectBase(info)))
  {
    return &v->Value;
  }
  return nullptr;
}
