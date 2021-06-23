/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationDoubleKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationDoubleKey.h"

#include "svtkInformation.h"

//----------------------------------------------------------------------------
svtkInformationDoubleKey::svtkInformationDoubleKey(const char* name, const char* location)
  : svtkInformationKey(name, location)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationDoubleKey::~svtkInformationDoubleKey() = default;

//----------------------------------------------------------------------------
void svtkInformationDoubleKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
class svtkInformationDoubleValue : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkInformationDoubleValue, svtkObjectBase);
  double Value;
};

//----------------------------------------------------------------------------
void svtkInformationDoubleKey::Set(svtkInformation* info, double value)
{
  if (svtkInformationDoubleValue* oldv =
        static_cast<svtkInformationDoubleValue*>(this->GetAsObjectBase(info)))
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
    svtkInformationDoubleValue* v = new svtkInformationDoubleValue;
    v->InitializeObjectBase();
    v->Value = value;
    this->SetAsObjectBase(info, v);
    v->Delete();
  }
}

//----------------------------------------------------------------------------
double svtkInformationDoubleKey::Get(svtkInformation* info)
{
  svtkInformationDoubleValue* v =
    static_cast<svtkInformationDoubleValue*>(this->GetAsObjectBase(info));
  return v ? v->Value : 0;
}

//----------------------------------------------------------------------------
void svtkInformationDoubleKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
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
void svtkInformationDoubleKey::Print(ostream& os, svtkInformation* info)
{
  // Print the value.
  if (this->Has(info))
  {
    os << this->Get(info);
  }
}

//----------------------------------------------------------------------------
double* svtkInformationDoubleKey::GetWatchAddress(svtkInformation* info)
{
  if (svtkInformationDoubleValue* v =
        static_cast<svtkInformationDoubleValue*>(this->GetAsObjectBase(info)))
  {
    return &v->Value;
  }
  return nullptr;
}
