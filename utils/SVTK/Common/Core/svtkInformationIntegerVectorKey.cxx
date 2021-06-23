/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationIntegerVectorKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationIntegerVectorKey.h"

#include "svtkInformation.h" // For svtkErrorWithObjectMacro

#include <algorithm>
#include <vector>

//----------------------------------------------------------------------------
svtkInformationIntegerVectorKey ::svtkInformationIntegerVectorKey(
  const char* name, const char* location, int length)
  : svtkInformationKey(name, location)
  , RequiredLength(length)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationIntegerVectorKey::~svtkInformationIntegerVectorKey() = default;

//----------------------------------------------------------------------------
void svtkInformationIntegerVectorKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
class svtkInformationIntegerVectorValue : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkInformationIntegerVectorValue, svtkObjectBase);
  std::vector<int> Value;
};

//----------------------------------------------------------------------------
void svtkInformationIntegerVectorKey::Append(svtkInformation* info, int value)
{
  svtkInformationIntegerVectorValue* v =
    static_cast<svtkInformationIntegerVectorValue*>(this->GetAsObjectBase(info));
  if (v)
  {
    v->Value.push_back(value);
  }
  else
  {
    this->Set(info, &value, 1);
  }
}

//----------------------------------------------------------------------------
void svtkInformationIntegerVectorKey::Set(svtkInformation* info)
{
  int someVal;
  this->Set(info, &someVal, 0);
}

//----------------------------------------------------------------------------
void svtkInformationIntegerVectorKey::Set(svtkInformation* info, const int* value, int length)
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

    svtkInformationIntegerVectorValue* oldv =
      static_cast<svtkInformationIntegerVectorValue*>(this->GetAsObjectBase(info));
    if (oldv && static_cast<int>(oldv->Value.size()) == length)
    {
      // Replace the existing value.
      std::copy(value, value + length, oldv->Value.begin());
      // Since this sets a value without call SetAsObjectBase(),
      // the info has to be modified here (instead of
      // svtkInformation::SetAsObjectBase()
      info->Modified(this);
    }
    else
    {
      // Allocate a new value.
      svtkInformationIntegerVectorValue* v = new svtkInformationIntegerVectorValue;
      v->InitializeObjectBase();
      v->Value.insert(v->Value.begin(), value, value + length);
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
int* svtkInformationIntegerVectorKey::Get(svtkInformation* info)
{
  svtkInformationIntegerVectorValue* v =
    static_cast<svtkInformationIntegerVectorValue*>(this->GetAsObjectBase(info));
  return (v && !v->Value.empty()) ? (&v->Value[0]) : nullptr;
}

//----------------------------------------------------------------------------
int svtkInformationIntegerVectorKey::Get(svtkInformation* info, int idx)
{
  if (idx >= this->Length(info))
  {
    svtkErrorWithObjectMacro(info,
      "Information does not contain " << idx << " elements. Cannot return information value.");
    return 0;
  }
  int* values = this->Get(info);
  return values[idx];
}

//----------------------------------------------------------------------------
void svtkInformationIntegerVectorKey::Get(svtkInformation* info, int* value)
{
  svtkInformationIntegerVectorValue* v =
    static_cast<svtkInformationIntegerVectorValue*>(this->GetAsObjectBase(info));
  if (v && value)
  {
    for (std::vector<int>::size_type i = 0; i < v->Value.size(); ++i)
    {
      value[i] = v->Value[i];
    }
  }
}

//----------------------------------------------------------------------------
int svtkInformationIntegerVectorKey::Length(svtkInformation* info)
{
  svtkInformationIntegerVectorValue* v =
    static_cast<svtkInformationIntegerVectorValue*>(this->GetAsObjectBase(info));
  return v ? static_cast<int>(v->Value.size()) : 0;
}

//----------------------------------------------------------------------------
void svtkInformationIntegerVectorKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
{
  this->Set(to, this->Get(from), this->Length(from));
}

//----------------------------------------------------------------------------
void svtkInformationIntegerVectorKey::Print(ostream& os, svtkInformation* info)
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
int* svtkInformationIntegerVectorKey::GetWatchAddress(svtkInformation* info)
{
  svtkInformationIntegerVectorValue* v =
    static_cast<svtkInformationIntegerVectorValue*>(this->GetAsObjectBase(info));
  return (v && !v->Value.empty()) ? (&v->Value[0]) : nullptr;
}
