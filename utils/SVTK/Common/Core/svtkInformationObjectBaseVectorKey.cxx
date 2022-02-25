/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationObjectBaseVectorKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationObjectBaseVectorKey.h"
#include "svtkInformation.h" // For svtkErrorWithObjectMacro
#include "svtkSmartPointer.h"
#include <algorithm>
#include <vector>

//============================================================================
class svtkInformationObjectBaseVectorValue : public svtkObjectBase
{
public:
  typedef std::vector<svtkSmartPointer<svtkObjectBase> > VectorType;

  svtkBaseTypeMacro(svtkInformationObjectBaseVectorValue, svtkObjectBase);
  std::vector<svtkSmartPointer<svtkObjectBase> >& GetVector() { return this->Vector; }

private:
  std::vector<svtkSmartPointer<svtkObjectBase> > Vector;
};

//============================================================================

//----------------------------------------------------------------------------
svtkInformationObjectBaseVectorKey::svtkInformationObjectBaseVectorKey(
  const char* name, const char* location, const char* requiredClass)
  : svtkInformationKey(name, location)
  , RequiredClass(requiredClass)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationObjectBaseVectorKey::~svtkInformationObjectBaseVectorKey() = default;

//----------------------------------------------------------------------------
void svtkInformationObjectBaseVectorKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
svtkInformationObjectBaseVectorValue* svtkInformationObjectBaseVectorKey::GetObjectBaseVector(
  svtkInformation* info)
{
  // Grab the vector associated with this key.
  svtkInformationObjectBaseVectorValue* base =
    static_cast<svtkInformationObjectBaseVectorValue*>(this->GetAsObjectBase(info));

  // If we don't already have a vector then associated,
  // we will create it here.
  if (base == nullptr)
  {
    base = new svtkInformationObjectBaseVectorValue;
    base->InitializeObjectBase();
    this->SetAsObjectBase(info, base);
    base->Delete();
  }

  return base;
}

//----------------------------------------------------------------------------
bool svtkInformationObjectBaseVectorKey::ValidateDerivedType(
  svtkInformation* info, svtkObjectBase* aValue)
{
  // verify that type of aValue is compatible with
  // this container.
  if (aValue != nullptr && this->RequiredClass != nullptr && !aValue->IsA(this->RequiredClass))
  {
    svtkErrorWithObjectMacro(info,
      "Cannot store object of type " << aValue->GetClassName() << " with key " << this->Location
                                     << "::" << this->Name << " which requires objects of type "
                                     << this->RequiredClass << ".");
    return false;
  }
  return true;
}

//----------------------------------------------------------------------------
void svtkInformationObjectBaseVectorKey::Append(svtkInformation* info, svtkObjectBase* aValue)
{
  if (!this->ValidateDerivedType(info, aValue))
  {
    return;
  }
  //
  svtkInformationObjectBaseVectorValue* base = this->GetObjectBaseVector(info);
  //
  if (aValue != nullptr)
  {
    aValue->Register(base);
  }
  //
  base->GetVector().push_back(aValue);
}

//----------------------------------------------------------------------------
void svtkInformationObjectBaseVectorKey::Set(svtkInformation* info, svtkObjectBase* aValue, int i)
{
  if (!this->ValidateDerivedType(info, aValue))
  {
    return;
  }
  // Get the vector associated with this key, resize if this
  // set would run off the end.
  svtkInformationObjectBaseVectorValue* base = this->GetObjectBaseVector(info);
  int n = static_cast<int>(base->GetVector().size());
  if (i >= n)
  {
    base->GetVector().resize(i + 1);
  }
  // Set.
  base->GetVector()[i] = aValue;
}

//----------------------------------------------------------------------------
void svtkInformationObjectBaseVectorKey::Remove(svtkInformation* info, svtkObjectBase* val)
{
  if (!this->ValidateDerivedType(info, val))
  {
    return;
  }
  svtkInformationObjectBaseVectorValue* base = this->GetObjectBaseVector(info);

  typedef svtkInformationObjectBaseVectorValue::VectorType Vector;
  Vector& vector = base->GetVector();
  Vector::iterator end = std::remove(vector.begin(), vector.end(), val);
  if (end != vector.end())
  {
    vector.resize(std::distance(vector.begin(), end));
    if (val)
    {
      val->UnRegister(base);
    }
  }
}

//----------------------------------------------------------------------------
void svtkInformationObjectBaseVectorKey::Remove(svtkInformation* info, int idx)
{
  typedef svtkInformationObjectBaseVectorValue::VectorType Vector;
  svtkInformationObjectBaseVectorValue* base = this->GetObjectBaseVector(info);
  Vector& vector = base->GetVector();
  if (idx >= static_cast<int>(vector.size()))
  {
    return;
  }

  svtkObjectBase* val = vector[idx];
  if (val)
  {
    val->UnRegister(base);
  }

  vector.erase(vector.begin() + idx);
}

//----------------------------------------------------------------------------
void svtkInformationObjectBaseVectorKey::SetRange(
  svtkInformation* info, svtkObjectBase** sourceVec, int from, int to, int n)
{
  // Get the vector associated with this key, resize if this
  // set would run off the end.
  svtkInformationObjectBaseVectorValue* base = this->GetObjectBaseVector(info);
  int m = static_cast<int>(base->GetVector().size());
  int reqsz = to + n;
  if (reqsz > m)
  {
    base->GetVector().resize(reqsz);
  }
  // Set.
  for (int i = 0; i < n; ++i, ++from, ++to)
  {
    base->GetVector()[to] = sourceVec[from];
  }
}

// //----------------------------------------------------------------------------
// svtkSmartPointer<svtkObjectBase> *svtkInformationObjectBaseVectorKey::Get(
//         svtkInformation* info)
// {
//   svtkInformationObjectBaseVectorValue* base =
//     static_cast<svtkInformationObjectBaseVectorValue *>(this->GetAsObjectBase(info));
//
//   return
//     (base!=nullptr && !base->GetVector().empty())?(&base->GetVector()[0]):0;
// }

//----------------------------------------------------------------------------
void svtkInformationObjectBaseVectorKey::GetRange(
  svtkInformation* info, svtkObjectBase** dest, int from, int to, int n)
{
  svtkInformationObjectBaseVectorValue* base =
    static_cast<svtkInformationObjectBaseVectorValue*>(this->GetAsObjectBase(info));

  // Source vector exists?
  if (base == nullptr)
  {
    svtkErrorWithObjectMacro(info, "Copy of empty vector has been requested.");
    return;
  }

  int m = static_cast<int>(base->GetVector().size());
  // check source start.
  if (from >= m)
  {
    svtkErrorWithObjectMacro(info, "Copy starting past the end of the vector has been requested.");
    return;
  }

  // limit copy to whats there.
  if (n > m - from + 1)
  {
    svtkErrorWithObjectMacro(info, "Copy past the end of the vector has been requested.");
    n = m - from + 1;
  }

  // copy
  for (int i = 0; i < n; ++i, ++from, ++to)
  {
    dest[to] = base->GetVector()[from];
  }
}

//----------------------------------------------------------------------------
svtkObjectBase* svtkInformationObjectBaseVectorKey::Get(svtkInformation* info, int idx)
{
  svtkInformationObjectBaseVectorValue* base =
    static_cast<svtkInformationObjectBaseVectorValue*>(this->GetAsObjectBase(info));

  if (base == nullptr || idx >= static_cast<int>(base->GetVector().size()))
  {
    svtkErrorWithObjectMacro(info,
      "Information does not contain " << idx << " elements. Cannot return information value.");
    return nullptr;
  }

  return base->GetVector()[idx];
}

//----------------------------------------------------------------------------
int svtkInformationObjectBaseVectorKey::Size(svtkInformation* info)
{
  svtkInformationObjectBaseVectorValue* base =
    static_cast<svtkInformationObjectBaseVectorValue*>(this->GetAsObjectBase(info));

  return (base == nullptr ? 0 : static_cast<int>(base->GetVector().size()));
}

//----------------------------------------------------------------------------
void svtkInformationObjectBaseVectorKey::Resize(svtkInformation* info, int size)
{
  svtkInformationObjectBaseVectorValue* base = this->GetObjectBaseVector(info);
  base->GetVector().resize(size);
}

//----------------------------------------------------------------------------
void svtkInformationObjectBaseVectorKey::Clear(svtkInformation* info)
{
  svtkInformationObjectBaseVectorValue* base = this->GetObjectBaseVector(info);
  base->GetVector().clear();
}

//----------------------------------------------------------------------------
void svtkInformationObjectBaseVectorKey::ShallowCopy(svtkInformation* source, svtkInformation* dest)
{
  svtkInformationObjectBaseVectorValue* sourceBase =
    static_cast<svtkInformationObjectBaseVectorValue*>(this->GetAsObjectBase(source));

  if (sourceBase == nullptr)
  {
    this->SetAsObjectBase(dest, nullptr);
    return;
  }

  int sourceSize = static_cast<int>(sourceBase->GetVector().size());
  svtkInformationObjectBaseVectorValue* destBase = this->GetObjectBaseVector(dest);

  destBase->GetVector().resize(sourceSize);
  destBase->GetVector() = sourceBase->GetVector();
}

//----------------------------------------------------------------------------
void svtkInformationObjectBaseVectorKey::Print(ostream& os, svtkInformation* info)
{
  svtkIndent indent;
  // Grab the vector associated with this key.
  svtkInformationObjectBaseVectorValue* base =
    static_cast<svtkInformationObjectBaseVectorValue*>(this->GetAsObjectBase(info));
  // Print each valid item.
  if (base != nullptr)
  {
    int n = static_cast<int>(base->GetVector().size());
    if (n > 0)
    {
      svtkObjectBase* itemBase = base->GetVector()[0];
      os << indent << "item " << 0 << "=";
      itemBase->PrintSelf(os, indent);
      os << endl;
    }
    for (int i = 1; i < n; ++i)
    {
      os << indent << "item " << i << "=";
      svtkObjectBase* itemBase = base->GetVector()[i];
      if (itemBase != nullptr)
      {
        itemBase->PrintSelf(os, indent);
      }
      else
      {
        os << "nullptr;";
      }
      os << endl;
    }
  }
}
