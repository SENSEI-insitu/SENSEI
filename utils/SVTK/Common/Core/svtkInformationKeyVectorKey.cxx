/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationKeyVectorKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationKeyVectorKey.h"

#include "svtkInformation.h"
#include <algorithm> // find()
#include <vector>

//----------------------------------------------------------------------------
svtkInformationKeyVectorKey::svtkInformationKeyVectorKey(const char* name, const char* location)
  : svtkInformationKey(name, location)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationKeyVectorKey::~svtkInformationKeyVectorKey() = default;

//----------------------------------------------------------------------------
void svtkInformationKeyVectorKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
class svtkInformationKeyVectorValue : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkInformationKeyVectorValue, svtkObjectBase);
  std::vector<svtkInformationKey*> Value;
};

//----------------------------------------------------------------------------
void svtkInformationKeyVectorKey::Append(svtkInformation* info, svtkInformationKey* value)
{
  svtkInformationKeyVectorValue* v =
    static_cast<svtkInformationKeyVectorValue*>(this->GetAsObjectBase(info));
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
void svtkInformationKeyVectorKey::AppendUnique(svtkInformation* info, svtkInformationKey* value)
{
  svtkInformationKeyVectorValue* v =
    static_cast<svtkInformationKeyVectorValue*>(this->GetAsObjectBase(info));
  if (v)
  {
    int found = 0;
    size_t len = v->Value.size();
    for (size_t i = 0; i < len; i++)
    {
      if (v->Value[i] == value)
      {
        found = 1;
        break;
      }
    }
    if (!found)
    {
      v->Value.push_back(value);
    }
  }
  else
  {
    this->Set(info, &value, 1);
  }
}

//----------------------------------------------------------------------------
void svtkInformationKeyVectorKey::Set(
  svtkInformation* info, svtkInformationKey* const* value, int length)
{
  if (value)
  {
    svtkInformationKeyVectorValue* v = new svtkInformationKeyVectorValue;
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
void svtkInformationKeyVectorKey::RemoveItem(svtkInformation* info, svtkInformationKey* value)
{
  svtkInformationKeyVectorValue* v =
    static_cast<svtkInformationKeyVectorValue*>(this->GetAsObjectBase(info));

  if (v)
  {
    std::vector<svtkInformationKey*>::iterator it =
      std::find(v->Value.begin(), v->Value.end(), value);
    if (it != v->Value.end())
    {
      v->Value.erase(it);
    }
  }
}

//----------------------------------------------------------------------------
svtkInformationKey** svtkInformationKeyVectorKey::Get(svtkInformation* info)
{
  svtkInformationKeyVectorValue* v =
    static_cast<svtkInformationKeyVectorValue*>(this->GetAsObjectBase(info));
  return (v && !v->Value.empty()) ? (&v->Value[0]) : nullptr;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformationKeyVectorKey::Get(svtkInformation* info, int idx)
{
  if (idx >= this->Length(info))
  {
    svtkErrorWithObjectMacro(info,
      "Information does not contain " << idx << " elements. Cannot return information value.");
    return nullptr;
  }
  svtkInformationKey** values = this->Get(info);
  return values[idx];
}

//----------------------------------------------------------------------------
void svtkInformationKeyVectorKey::Get(svtkInformation* info, svtkInformationKey** value)
{
  svtkInformationKeyVectorValue* v =
    static_cast<svtkInformationKeyVectorValue*>(this->GetAsObjectBase(info));
  if (v && value)
  {
    for (std::vector<svtkInformationKey*>::size_type i = 0; i < v->Value.size(); ++i)
    {
      value[i] = v->Value[i];
    }
  }
}

//----------------------------------------------------------------------------
int svtkInformationKeyVectorKey::Length(svtkInformation* info)
{
  svtkInformationKeyVectorValue* v =
    static_cast<svtkInformationKeyVectorValue*>(this->GetAsObjectBase(info));
  return v ? static_cast<int>(v->Value.size()) : 0;
}

//----------------------------------------------------------------------------
void svtkInformationKeyVectorKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
{
  this->Set(to, this->Get(from), this->Length(from));
}

//----------------------------------------------------------------------------
void svtkInformationKeyVectorKey::Print(ostream& os, svtkInformation* info)
{
  // Print the value.
  if (this->Has(info))
  {
    svtkInformationKey** value = this->Get(info);
    int length = this->Length(info);
    const char* sep = "";
    for (int i = 0; i < length; ++i)
    {
      os << sep << (value[i] ? value[i]->GetName() : "(nullptr)");
      sep = " ";
    }
  }
}
