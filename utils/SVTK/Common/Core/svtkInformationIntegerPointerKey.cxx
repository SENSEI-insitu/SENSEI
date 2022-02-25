/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationIntegerPointerKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationIntegerPointerKey.h"

#include "svtkInformation.h" // For svtkErrorWithObjectMacro

#include <algorithm>
#include <vector>

//----------------------------------------------------------------------------
svtkInformationIntegerPointerKey ::svtkInformationIntegerPointerKey(
  const char* name, const char* location, int length)
  : svtkInformationKey(name, location)
  , RequiredLength(length)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationIntegerPointerKey::~svtkInformationIntegerPointerKey() = default;

//----------------------------------------------------------------------------
void svtkInformationIntegerPointerKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
class svtkInformationIntegerPointerValue : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkInformationIntegerPointerValue, svtkObjectBase);
  int* Value;
  unsigned int Length;
};

//----------------------------------------------------------------------------
void svtkInformationIntegerPointerKey::Set(svtkInformation* info, int* value, int length)
{
  if (value)
  {
    if (this->RequiredLength >= 0 && length != this->RequiredLength)
    {
      svtkErrorWithObjectMacro(info,
        "Cannot store integer vector of length "
          << length << " with key " << this->Location << "::" << this->Name
          << " which requires a vector of length " << this->RequiredLength
          << ".  Removing the key instead.");
      this->SetAsObjectBase(info, nullptr);
      return;
    }

    // Allocate a new value.
    svtkInformationIntegerPointerValue* v = new svtkInformationIntegerPointerValue;
    v->InitializeObjectBase();
    v->Value = value;
    v->Length = length;
    this->SetAsObjectBase(info, v);
    v->Delete();
  }
  else
  {
    this->SetAsObjectBase(info, nullptr);
  }
}

//----------------------------------------------------------------------------
int* svtkInformationIntegerPointerKey::Get(svtkInformation* info)
{
  svtkInformationIntegerPointerValue* v =
    static_cast<svtkInformationIntegerPointerValue*>(this->GetAsObjectBase(info));
  return v->Value;
}

//----------------------------------------------------------------------------
void svtkInformationIntegerPointerKey::Get(svtkInformation* info, int* value)
{
  svtkInformationIntegerPointerValue* v =
    static_cast<svtkInformationIntegerPointerValue*>(this->GetAsObjectBase(info));
  if (v && value)
  {
    memcpy(value, v->Value, v->Length * sizeof(int));
  }
}

//----------------------------------------------------------------------------
int svtkInformationIntegerPointerKey::Length(svtkInformation* info)
{
  svtkInformationIntegerPointerValue* v =
    static_cast<svtkInformationIntegerPointerValue*>(this->GetAsObjectBase(info));
  return v->Length;
}

//----------------------------------------------------------------------------
void svtkInformationIntegerPointerKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
{
  this->Set(to, this->Get(from), this->Length(from));
}

//----------------------------------------------------------------------------
void svtkInformationIntegerPointerKey::Print(ostream& os, svtkInformation* info)
{
  // Print the value.
  if (this->Has(info))
  {
    int* value = this->Get(info);
    int length = this->Length(info);
    const char* sep = "";
    for (int i = 0; i < length; ++i)
    {
      os << sep << value[i];
      sep = " ";
    }
  }
}

//----------------------------------------------------------------------------
int* svtkInformationIntegerPointerKey::GetWatchAddress(svtkInformation* info)
{
  if (svtkInformationIntegerPointerValue* v =
        static_cast<svtkInformationIntegerPointerValue*>(this->GetAsObjectBase(info)))
  {
    return v->Value;
  }
  return nullptr;
}
