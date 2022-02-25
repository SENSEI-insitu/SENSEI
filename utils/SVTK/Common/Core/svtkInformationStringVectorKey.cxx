/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationStringVectorKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationStringVectorKey.h"

#include "svtkInformation.h" // For svtkErrorWithObjectMacro
#include "svtkStdString.h"

#include <algorithm>
#include <vector>

//----------------------------------------------------------------------------
svtkInformationStringVectorKey ::svtkInformationStringVectorKey(
  const char* name, const char* location, int length)
  : svtkInformationKey(name, location)
  , RequiredLength(length)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationStringVectorKey::~svtkInformationStringVectorKey() = default;

//----------------------------------------------------------------------------
void svtkInformationStringVectorKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
class svtkInformationStringVectorValue : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkInformationStringVectorValue, svtkObjectBase);
  std::vector<std::string> Value;
};

//----------------------------------------------------------------------------
void svtkInformationStringVectorKey::Append(svtkInformation* info, const char* value)
{
  svtkInformationStringVectorValue* v =
    static_cast<svtkInformationStringVectorValue*>(this->GetAsObjectBase(info));
  if (v)
  {
    v->Value.push_back(value);
  }
  else
  {
    this->Set(info, value, 0);
  }
}

//----------------------------------------------------------------------------
void svtkInformationStringVectorKey::Set(svtkInformation* info, const char* value, int index)
{
  svtkInformationStringVectorValue* oldv =
    static_cast<svtkInformationStringVectorValue*>(this->GetAsObjectBase(info));
  if (oldv)
  {
    if ((static_cast<int>(oldv->Value.size()) <= index) || (oldv->Value[index] != value))
    {
      while (static_cast<int>(oldv->Value.size()) <= index)
      {
        oldv->Value.push_back("");
      }
      oldv->Value[index] = value;
      // Since this sets a value without call SetAsObjectBase(),
      // the info has to be modified here (instead of
      // svtkInformation::SetAsObjectBase()
      info->Modified(this);
    }
  }
  else
  {
    svtkInformationStringVectorValue* v = new svtkInformationStringVectorValue;
    v->InitializeObjectBase();
    while (static_cast<int>(v->Value.size()) <= index)
    {
      v->Value.push_back("");
    }
    v->Value[index] = value;
    this->SetAsObjectBase(info, v);
    v->Delete();
  }
}

//----------------------------------------------------------------------------
void svtkInformationStringVectorKey::Append(svtkInformation* info, const std::string& value)
{
  this->Append(info, value.c_str());
}

//----------------------------------------------------------------------------
void svtkInformationStringVectorKey::Set(svtkInformation* info, const std::string& value, int idx)
{
  this->Set(info, value.c_str(), idx);
}

//----------------------------------------------------------------------------
const char* svtkInformationStringVectorKey::Get(svtkInformation* info, int idx)
{
  if (idx < 0 || idx >= this->Length(info))
  {
    return nullptr;
  }
  svtkInformationStringVectorValue* v =
    static_cast<svtkInformationStringVectorValue*>(this->GetAsObjectBase(info));
  return v->Value[idx].c_str();
}

//----------------------------------------------------------------------------
int svtkInformationStringVectorKey::Length(svtkInformation* info)
{
  svtkInformationStringVectorValue* v =
    static_cast<svtkInformationStringVectorValue*>(this->GetAsObjectBase(info));
  return v ? static_cast<int>(v->Value.size()) : 0;
}

//----------------------------------------------------------------------------
void svtkInformationStringVectorKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
{
  int length = this->Length(from);
  for (int i = 0; i < length; ++i)
  {
    this->Set(to, this->Get(from, i), i);
  }
}

//----------------------------------------------------------------------------
void svtkInformationStringVectorKey::Print(ostream& os, svtkInformation* info)
{
  // Print the value.
  if (this->Has(info))
  {
    int length = this->Length(info);
    const char* sep = "";
    for (int i = 0; i < length; ++i)
    {
      os << sep << this->Get(info, i);
      sep = " ";
    }
  }
}
