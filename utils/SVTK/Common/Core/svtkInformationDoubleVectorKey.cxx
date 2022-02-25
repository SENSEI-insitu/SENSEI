/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationDoubleVectorKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationDoubleVectorKey.h"

#include "svtkInformation.h" // For svtkErrorWithObjectMacro

#include <vector>

//----------------------------------------------------------------------------
svtkInformationDoubleVectorKey ::svtkInformationDoubleVectorKey(
  const char* name, const char* location, int length)
  : svtkInformationKey(name, location)
  , RequiredLength(length)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationDoubleVectorKey::~svtkInformationDoubleVectorKey() = default;

//----------------------------------------------------------------------------
void svtkInformationDoubleVectorKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
class svtkInformationDoubleVectorValue : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkInformationDoubleVectorValue, svtkObjectBase);
  std::vector<double> Value;
};

//----------------------------------------------------------------------------
void svtkInformationDoubleVectorKey::Append(svtkInformation* info, double value)
{
  svtkInformationDoubleVectorValue* v =
    static_cast<svtkInformationDoubleVectorValue*>(this->GetAsObjectBase(info));
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
void svtkInformationDoubleVectorKey::Set(svtkInformation* info, const double* value, int length)
{
  if (value)
  {
    if (this->RequiredLength >= 0 && length != this->RequiredLength)
    {
      svtkErrorWithObjectMacro(info,
        "Cannot store double vector of length "
          << length << " with key " << this->Location << "::" << this->Name
          << " which requires a vector of length " << this->RequiredLength
          << ".  Removing the key instead.");
      this->SetAsObjectBase(info, nullptr);
      return;
    }
    svtkInformationDoubleVectorValue* v = new svtkInformationDoubleVectorValue;
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
double* svtkInformationDoubleVectorKey::Get(svtkInformation* info)
{
  svtkInformationDoubleVectorValue* v =
    static_cast<svtkInformationDoubleVectorValue*>(this->GetAsObjectBase(info));
  return (v && !v->Value.empty()) ? (&v->Value[0]) : nullptr;
}

//----------------------------------------------------------------------------
double svtkInformationDoubleVectorKey::Get(svtkInformation* info, int idx)
{
  if (idx >= this->Length(info))
  {
    svtkErrorWithObjectMacro(info,
      "Information does not contain " << idx << " elements. Cannot return information value.");
    return 0;
  }
  double* values = this->Get(info);
  return values[idx];
}

//----------------------------------------------------------------------------
void svtkInformationDoubleVectorKey::Get(svtkInformation* info, double* value)
{
  svtkInformationDoubleVectorValue* v =
    static_cast<svtkInformationDoubleVectorValue*>(this->GetAsObjectBase(info));
  if (v && value)
  {
    for (std::vector<double>::size_type i = 0; i < v->Value.size(); ++i)
    {
      value[i] = v->Value[i];
    }
  }
}

//----------------------------------------------------------------------------
int svtkInformationDoubleVectorKey::Length(svtkInformation* info)
{
  svtkInformationDoubleVectorValue* v =
    static_cast<svtkInformationDoubleVectorValue*>(this->GetAsObjectBase(info));
  return v ? static_cast<int>(v->Value.size()) : 0;
}

//----------------------------------------------------------------------------
void svtkInformationDoubleVectorKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
{
  this->Set(to, this->Get(from), this->Length(from));
}

//----------------------------------------------------------------------------
void svtkInformationDoubleVectorKey::Print(ostream& os, svtkInformation* info)
{
  // Print the value.
  if (this->Has(info))
  {
    double* value = this->Get(info);
    int length = this->Length(info);
    const char* sep = "";
    for (int i = 0; i < length; ++i)
    {
      os << sep << value[i];
      sep = " ";
    }
  }
}
