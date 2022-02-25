/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationVariantVectorKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationVariantVectorKey.h"

#include "svtkInformation.h" // For svtkErrorWithObjectMacro
#include "svtkVariant.h"

#include <vector>

//----------------------------------------------------------------------------
svtkInformationVariantVectorKey ::svtkInformationVariantVectorKey(
  const char* name, const char* location, int length)
  : svtkInformationKey(name, location)
  , RequiredLength(length)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationVariantVectorKey::~svtkInformationVariantVectorKey() = default;

//----------------------------------------------------------------------------
void svtkInformationVariantVectorKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
class svtkInformationVariantVectorValue : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkInformationVariantVectorValue, svtkObjectBase);
  std::vector<svtkVariant> Value;
  static svtkVariant Invalid;
};

svtkVariant svtkInformationVariantVectorValue::Invalid;

//----------------------------------------------------------------------------
void svtkInformationVariantVectorKey::Append(svtkInformation* info, const svtkVariant& value)
{
  svtkInformationVariantVectorValue* v =
    static_cast<svtkInformationVariantVectorValue*>(this->GetAsObjectBase(info));
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
void svtkInformationVariantVectorKey::Set(svtkInformation* info, const svtkVariant* value, int length)
{
  if (value)
  {
    if (this->RequiredLength >= 0 && length != this->RequiredLength)
    {
      svtkErrorWithObjectMacro(info,
        "Cannot store svtkVariant vector of length "
          << length << " with key " << this->Location << "::" << this->Name
          << " which requires a vector of length " << this->RequiredLength
          << ".  Removing the key instead.");
      this->SetAsObjectBase(info, nullptr);
      return;
    }
    svtkInformationVariantVectorValue* v = new svtkInformationVariantVectorValue;
    v->InitializeObjectBase();
    v->Value.insert(v->Value.begin(), value, value + length);
    this->SetAsObjectBase(info, v);
    v->Delete();
  }
  else
  {
    this->SetAsObjectBase(info, nullptr);
  }
}

//----------------------------------------------------------------------------
const svtkVariant* svtkInformationVariantVectorKey::Get(svtkInformation* info) const
{
  const svtkInformationVariantVectorValue* v =
    static_cast<const svtkInformationVariantVectorValue*>(this->GetAsObjectBase(info));
  return (v && !v->Value.empty()) ? (&v->Value[0]) : nullptr;
}

//----------------------------------------------------------------------------
const svtkVariant& svtkInformationVariantVectorKey::Get(svtkInformation* info, int idx) const
{
  if (idx >= this->Length(info))
  {
    svtkErrorWithObjectMacro(info,
      "Information does not contain " << idx << " elements. Cannot return information value.");
    return svtkInformationVariantVectorValue::Invalid;
  }
  const svtkVariant* values = this->Get(info);
  return values[idx];
}

//----------------------------------------------------------------------------
void svtkInformationVariantVectorKey::Get(svtkInformation* info, svtkVariant* value) const
{
  const svtkInformationVariantVectorValue* v =
    static_cast<const svtkInformationVariantVectorValue*>(this->GetAsObjectBase(info));
  if (v && value)
  {
    for (std::vector<svtkVariant>::size_type i = 0; i < v->Value.size(); ++i)
    {
      value[i] = v->Value[i];
    }
  }
}

//----------------------------------------------------------------------------
int svtkInformationVariantVectorKey::Length(svtkInformation* info) const
{
  const svtkInformationVariantVectorValue* v =
    static_cast<const svtkInformationVariantVectorValue*>(this->GetAsObjectBase(info));
  return v ? static_cast<int>(v->Value.size()) : 0;
}

//----------------------------------------------------------------------------
void svtkInformationVariantVectorKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
{
  this->Set(to, this->Get(from), this->Length(from));
}

//----------------------------------------------------------------------------
void svtkInformationVariantVectorKey::Print(ostream& os, svtkInformation* info)
{
  // Print the value.
  if (this->Has(info))
  {
    const svtkVariant* value = this->Get(info);
    int length = this->Length(info);
    const char* sep = "";
    for (int i = 0; i < length; ++i)
    {
      os << sep << value[i];
      sep = " ";
    }
  }
}
