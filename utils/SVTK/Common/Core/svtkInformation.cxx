/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformation.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformation.h"

#include "svtkCommand.h"
#include "svtkGarbageCollector.h"
#include "svtkInformationDataObjectKey.h"
#include "svtkInformationDoubleKey.h"
#include "svtkInformationDoubleVectorKey.h"
#include "svtkInformationIdTypeKey.h"
#include "svtkInformationInformationKey.h"
#include "svtkInformationInformationVectorKey.h"
#include "svtkInformationIntegerKey.h"
#include "svtkInformationIntegerPointerKey.h"
#include "svtkInformationIntegerVectorKey.h"
#include "svtkInformationIterator.h"
#include "svtkInformationKeyVectorKey.h"
#include "svtkInformationObjectBaseKey.h"
#include "svtkInformationObjectBaseVectorKey.h"
#include "svtkInformationRequestKey.h"
#include "svtkInformationStringKey.h"
#include "svtkInformationStringVectorKey.h"
#include "svtkInformationUnsignedLongKey.h"
#include "svtkInformationVariantKey.h"
#include "svtkInformationVariantVectorKey.h"
#include "svtkObjectFactory.h"
#include "svtkSmartPointer.h"
#include "svtkVariant.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "svtkInformationInternals.h"

svtkStandardNewMacro(svtkInformation);

//----------------------------------------------------------------------------
svtkInformation::svtkInformation()
{
  // Allocate the internal representation.
  this->Internal = new svtkInformationInternals;

  // There is no request key stored initially.
  this->Request = nullptr;
}

//----------------------------------------------------------------------------
svtkInformation::~svtkInformation()
{
  // Delete the internal representation.
  delete this->Internal;
}

//----------------------------------------------------------------------------
void svtkInformation::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  // Print the request if one is set.
  if (this->Request)
  {
    os << indent << "Request: " << this->Request->GetName() << "\n";
  }
  this->PrintKeys(os, indent);
}

//----------------------------------------------------------------------------
void svtkInformation::PrintKeys(ostream& os, svtkIndent indent)
{
  typedef svtkInformationInternals::MapType MapType;
  for (MapType::const_iterator i = this->Internal->Map.begin(); i != this->Internal->Map.end(); ++i)
  {
    // Print the key name first.
    svtkInformationKey* key = i->first;
    os << indent << key->GetName() << ": ";

    // Ask the key to print its value.
    key->Print(os, this);
    os << "\n";
  }
}

//----------------------------------------------------------------------------
// call modified on superclass
void svtkInformation::Modified()
{
  this->Superclass::Modified();
}

//----------------------------------------------------------------------------
// Update MTime and invoke a modified event with
// the information key as call data
void svtkInformation::Modified(svtkInformationKey* key)
{
  this->MTime.Modified();
  this->InvokeEvent(svtkCommand::ModifiedEvent, key);
}

//----------------------------------------------------------------------------
// Return the number of keys as a result of iteration.
int svtkInformation::GetNumberOfKeys()
{
  svtkSmartPointer<svtkInformationIterator> infoIterator =
    svtkSmartPointer<svtkInformationIterator>::New();
  infoIterator->SetInformation(this);

  int numberOfKeys = 0;
  for (infoIterator->InitTraversal(); !infoIterator->IsDoneWithTraversal();
       infoIterator->GoToNextItem())
  {
    numberOfKeys++;
  }
  return numberOfKeys;
}

//----------------------------------------------------------------------------
void svtkInformation::SetAsObjectBase(svtkInformationKey* key, svtkObjectBase* newvalue)
{
  if (!key)
  {
    return;
  }
  typedef svtkInformationInternals::MapType MapType;
  MapType::iterator i = this->Internal->Map.find(key);
  if (i != this->Internal->Map.end())
  {
    svtkObjectBase* oldvalue = i->second;
    if (newvalue)
    {
      i->second = newvalue;
      newvalue->Register(nullptr);
    }
    else
    {
      this->Internal->Map.erase(i);
    }
    oldvalue->UnRegister(nullptr);
  }
  else if (newvalue)
  {
    MapType::value_type entry(key, newvalue);
    this->Internal->Map.insert(entry);
    newvalue->Register(nullptr);
  }
  this->Modified(key);
}

//----------------------------------------------------------------------------
const svtkObjectBase* svtkInformation::GetAsObjectBase(const svtkInformationKey* key) const
{
  if (key)
  {
    typedef svtkInformationInternals::MapType MapType;
    MapType::const_iterator i = this->Internal->Map.find(const_cast<svtkInformationKey*>(key));
    if (i != this->Internal->Map.end())
    {
      return i->second;
    }
  }
  return nullptr;
}

//----------------------------------------------------------------------------
svtkObjectBase* svtkInformation::GetAsObjectBase(svtkInformationKey* key)
{
  if (key)
  {
    typedef svtkInformationInternals::MapType MapType;
    MapType::const_iterator i = this->Internal->Map.find(key);
    if (i != this->Internal->Map.end())
    {
      return i->second;
    }
  }
  return nullptr;
}

//----------------------------------------------------------------------------
void svtkInformation::Clear()
{
  this->Copy(nullptr);
}

//----------------------------------------------------------------------------
void svtkInformation::Copy(svtkInformation* from, int deep)
{
  svtkInformationInternals* oldInternal = this->Internal;
  this->Internal = new svtkInformationInternals;
  if (from)
  {
    typedef svtkInformationInternals::MapType MapType;
    for (MapType::const_iterator i = from->Internal->Map.begin(); i != from->Internal->Map.end();
         ++i)
    {
      this->CopyEntry(from, i->first, deep);
    }
  }
  delete oldInternal;
}

//----------------------------------------------------------------------------
void svtkInformation::Append(svtkInformation* from, int deep)
{
  if (from)
  {
    typedef svtkInformationInternals::MapType MapType;
    for (MapType::const_iterator i = from->Internal->Map.begin(); i != from->Internal->Map.end();
         ++i)
    {
      this->CopyEntry(from, i->first, deep);
    }
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntry(svtkInformation* from, svtkInformationKey* key, int deep)
{
  if (!deep)
  {
    key->ShallowCopy(from, this);
  }
  else
  {
    key->DeepCopy(from, this);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntry(svtkInformation* from, svtkInformationDataObjectKey* key, int deep)
{
  if (!deep)
  {
    key->ShallowCopy(from, this);
  }
  else
  {
    key->DeepCopy(from, this);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntry(svtkInformation* from, svtkInformationInformationKey* key, int deep)
{
  if (!deep)
  {
    key->ShallowCopy(from, this);
  }
  else
  {
    key->DeepCopy(from, this);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntry(
  svtkInformation* from, svtkInformationInformationVectorKey* key, int deep)
{
  if (!deep)
  {
    key->ShallowCopy(from, this);
  }
  else
  {
    key->DeepCopy(from, this);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntry(svtkInformation* from, svtkInformationIntegerKey* key, int deep)
{
  if (!deep)
  {
    key->ShallowCopy(from, this);
  }
  else
  {
    key->DeepCopy(from, this);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntry(svtkInformation* from, svtkInformationRequestKey* key, int deep)
{
  if (!deep)
  {
    key->ShallowCopy(from, this);
  }
  else
  {
    key->DeepCopy(from, this);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntry(svtkInformation* from, svtkInformationIntegerVectorKey* key, int deep)
{
  if (!deep)
  {
    key->ShallowCopy(from, this);
  }
  else
  {
    key->DeepCopy(from, this);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntry(
  svtkInformation* from, svtkInformationObjectBaseVectorKey* key, int deep)
{
  if (!deep)
  {
    key->ShallowCopy(from, this);
  }
  else
  {
    key->DeepCopy(from, this);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntry(svtkInformation* from, svtkInformationDoubleVectorKey* key, int deep)
{
  if (!deep)
  {
    key->ShallowCopy(from, this);
  }
  else
  {
    key->DeepCopy(from, this);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntry(svtkInformation* from, svtkInformationVariantKey* key, int deep)
{
  if (!deep)
  {
    key->ShallowCopy(from, this);
  }
  else
  {
    key->DeepCopy(from, this);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntry(svtkInformation* from, svtkInformationVariantVectorKey* key, int deep)
{
  if (!deep)
  {
    key->ShallowCopy(from, this);
  }
  else
  {
    key->DeepCopy(from, this);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntry(svtkInformation* from, svtkInformationStringKey* key, int deep)
{
  if (!deep)
  {
    key->ShallowCopy(from, this);
  }
  else
  {
    key->DeepCopy(from, this);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntry(svtkInformation* from, svtkInformationUnsignedLongKey* key, int deep)
{
  if (!deep)
  {
    key->ShallowCopy(from, this);
  }
  else
  {
    key->DeepCopy(from, this);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntry(svtkInformation* from, svtkInformationStringVectorKey* key, int deep)
{
  if (!deep)
  {
    key->ShallowCopy(from, this);
  }
  else
  {
    key->DeepCopy(from, this);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::CopyEntries(svtkInformation* from, svtkInformationKeyVectorKey* key, int deep)
{
  int numberOfKeys = from->Length(key);
  svtkInformationKey** keys = from->Get(key);
  for (int i = 0; i < numberOfKeys; ++i)
  {
    this->CopyEntry(from, keys[i], deep);
  }
}

//----------------------------------------------------------------------------
int svtkInformation::Has(svtkInformationKey* key)
{
  // Use the virtual interface in case this is a special-cased key.
  return key->Has(this) ? 1 : 0;
}

//----------------------------------------------------------------------------
void svtkInformation::Remove(svtkInformationKey* key)
{
  // Use the virtual interface in case this is a special-cased key.
  key->Remove(this);
}

void svtkInformation::Set(svtkInformationRequestKey* key)
{
  key->Set(this);
}
void svtkInformation::Remove(svtkInformationRequestKey* key)
{
  key->svtkInformationRequestKey::Remove(this);
}
int svtkInformation::Has(svtkInformationRequestKey* key)
{
  return key->svtkInformationRequestKey::Has(this);
}

//----------------------------------------------------------------------------
#define SVTK_INFORMATION_DEFINE_SCALAR_PROPERTY(name, type)                                         \
  void svtkInformation::Set(svtkInformation##name##Key* key, type value) { key->Set(this, value); }  \
  void svtkInformation::Remove(svtkInformation##name##Key* key)                                      \
  {                                                                                                \
    key->svtkInformation##name##Key::Remove(this);                                                  \
  }                                                                                                \
  type svtkInformation::Get(svtkInformation##name##Key* key) { return key->Get(this); }              \
  int svtkInformation::Has(svtkInformation##name##Key* key)                                          \
  {                                                                                                \
    return key->svtkInformation##name##Key::Has(this);                                              \
  }
SVTK_INFORMATION_DEFINE_SCALAR_PROPERTY(IdType, svtkIdType);
SVTK_INFORMATION_DEFINE_SCALAR_PROPERTY(Integer, int);
SVTK_INFORMATION_DEFINE_SCALAR_PROPERTY(Double, double);
SVTK_INFORMATION_DEFINE_SCALAR_PROPERTY(UnsignedLong, unsigned long);
SVTK_INFORMATION_DEFINE_SCALAR_PROPERTY(String, const char*);
SVTK_INFORMATION_DEFINE_SCALAR_PROPERTY(DataObject, svtkDataObject*);
SVTK_INFORMATION_DEFINE_SCALAR_PROPERTY(Information, svtkInformation*);
SVTK_INFORMATION_DEFINE_SCALAR_PROPERTY(InformationVector, svtkInformationVector*);
SVTK_INFORMATION_DEFINE_SCALAR_PROPERTY(ObjectBase, svtkObjectBase*);
SVTK_INFORMATION_DEFINE_SCALAR_PROPERTY(Variant, const svtkVariant&);
#undef SVTK_INFORMATION_DEFINE_SCALAR_PROPERTY

//----------------------------------------------------------------------------
#define SVTK_INFORMATION_DEFINE_VECTOR_PROPERTY(name, type)                                         \
  void svtkInformation::Append(svtkInformation##name##VectorKey* key, type value)                    \
  {                                                                                                \
    key->Append(this, value);                                                                      \
  }                                                                                                \
  void svtkInformation::Set(svtkInformation##name##VectorKey* key, type const* value, int length)    \
  {                                                                                                \
    key->Set(this, value, length);                                                                 \
  }                                                                                                \
  type* svtkInformation::Get(svtkInformation##name##VectorKey* key) { return key->Get(this); }       \
  type svtkInformation::Get(svtkInformation##name##VectorKey* key, int idx)                          \
  {                                                                                                \
    return key->Get(this, idx);                                                                    \
  }                                                                                                \
  void svtkInformation::Get(svtkInformation##name##VectorKey* key, type* value)                      \
  {                                                                                                \
    key->Get(this, value);                                                                         \
  }                                                                                                \
  int svtkInformation::Length(svtkInformation##name##VectorKey* key) { return key->Length(this); }   \
  void svtkInformation::Remove(svtkInformation##name##VectorKey* key)                                \
  {                                                                                                \
    key->svtkInformation##name##VectorKey::Remove(this);                                            \
  }                                                                                                \
  int svtkInformation::Has(svtkInformation##name##VectorKey* key)                                    \
  {                                                                                                \
    return key->svtkInformation##name##VectorKey::Has(this);                                        \
  }
SVTK_INFORMATION_DEFINE_VECTOR_PROPERTY(Integer, int);
SVTK_INFORMATION_DEFINE_VECTOR_PROPERTY(Double, double);

// String keys can accept std::string.
void svtkInformation::Append(svtkInformationStringVectorKey* key, const std::string& value)
{
  this->Append(key, value.c_str());
}
void svtkInformation::Set(svtkInformationStringVectorKey* key, const std::string& value, int idx)
{
  this->Set(key, value.c_str(), idx);
}
void svtkInformation::Set(svtkInformationStringKey* key, const std::string& value)
{
  this->Set(key, value.c_str());
}

// Variant vector key is slightly different to accommodate efficient
// pass-by-reference instead of pass-by-value calls.
void svtkInformation::Append(svtkInformationVariantVectorKey* key, const svtkVariant& value)
{
  key->Append(this, value);
}
void svtkInformation::Set(svtkInformationVariantVectorKey* key, const svtkVariant* value, int length)
{
  key->Set(this, value, length);
}
const svtkVariant* svtkInformation::Get(svtkInformationVariantVectorKey* key)
{
  return key->Get(this);
}
const svtkVariant& svtkInformation::Get(svtkInformationVariantVectorKey* key, int idx)
{
  return key->Get(this, idx);
}
void svtkInformation::Get(svtkInformationVariantVectorKey* key, svtkVariant* value)
{
  key->Get(this, value);
}
int svtkInformation::Length(svtkInformationVariantVectorKey* key)
{
  return key->Length(this);
}
void svtkInformation::Remove(svtkInformationVariantVectorKey* key)
{
  key->svtkInformationVariantVectorKey::Remove(this);
}
int svtkInformation::Has(svtkInformationVariantVectorKey* key)
{
  return key->svtkInformationVariantVectorKey::Has(this);
}

// String vector key is slightly different to make it backwards compatible with
// the scalar string key.
void svtkInformation::Append(svtkInformationStringVectorKey* key, const char* value)
{
  key->Append(this, value);
}
void svtkInformation::Set(svtkInformationStringVectorKey* key, const char* value, int idx)
{
  key->Set(this, value, idx);
}
const char* svtkInformation::Get(svtkInformationStringVectorKey* key, int idx)
{
  return key->Get(this, idx);
}
int svtkInformation::Length(svtkInformationStringVectorKey* key)
{
  return key->Length(this);
}
void svtkInformation::Remove(svtkInformationStringVectorKey* key)
{
  key->svtkInformationStringVectorKey::Remove(this);
}
int svtkInformation::Has(svtkInformationStringVectorKey* key)
{
  return key->svtkInformationStringVectorKey::Has(this);
}

//------------------------------------------------------------------------------
void svtkInformation::Append(svtkInformationObjectBaseVectorKey* key, svtkObjectBase* data)
{
  key->Append(this, data);
}

//------------------------------------------------------------------------------
void svtkInformation::Set(svtkInformationObjectBaseVectorKey* key, svtkObjectBase* value, int idx)
{
  key->Set(this, value, idx);
}

//------------------------------------------------------------------------------
svtkObjectBase* svtkInformation::Get(svtkInformationObjectBaseVectorKey* key, int idx)
{
  return key->Get(this, idx);
}

//------------------------------------------------------------------------------
int svtkInformation::Length(svtkInformationObjectBaseVectorKey* key)
{
  return key->Length(this);
}

//------------------------------------------------------------------------------
void svtkInformation::Remove(svtkInformationObjectBaseVectorKey* key)
{
  key->Remove(this);
}

//------------------------------------------------------------------------------
void svtkInformation::Remove(svtkInformationObjectBaseVectorKey* key, svtkObjectBase* objectToRemove)
{
  key->Remove(this, objectToRemove);
}

//------------------------------------------------------------------------------
void svtkInformation::Remove(svtkInformationObjectBaseVectorKey* key, int indexToRemove)
{
  key->Remove(this, indexToRemove);
}

//------------------------------------------------------------------------------
int svtkInformation::Has(svtkInformationObjectBaseVectorKey* key)
{
  return key->Has(this);
}

SVTK_INFORMATION_DEFINE_VECTOR_PROPERTY(Key, svtkInformationKey*);
#define SVTK_INFORMATION_DEFINE_VECTOR_VALUE2_PROPERTY(name, type, atype)                           \
  void svtkInformation::Set(svtkInformation##name##VectorKey* key, atype value1, atype value2,       \
    atype value3, atype value4, atype value5, atype value6)                                        \
  {                                                                                                \
    type value[6];                                                                                 \
    value[0] = value1;                                                                             \
    value[1] = value2;                                                                             \
    value[2] = value3;                                                                             \
    value[3] = value4;                                                                             \
    value[4] = value5;                                                                             \
    value[5] = value6;                                                                             \
    key->Set(this, value, 6);                                                                      \
  }                                                                                                \
  void svtkInformation::Set(                                                                        \
    svtkInformation##name##VectorKey* key, atype value1, atype value2, atype value3)                \
  {                                                                                                \
    type value[3];                                                                                 \
    value[0] = value1;                                                                             \
    value[1] = value2;                                                                             \
    value[2] = value3;                                                                             \
    key->Set(this, value, 3);                                                                      \
  }
#define SVTK_INFORMATION_DEFINE_VECTOR_VALUE_PROPERTY(name, type)                                   \
  SVTK_INFORMATION_DEFINE_VECTOR_VALUE2_PROPERTY(name, type, type)
SVTK_INFORMATION_DEFINE_VECTOR_VALUE_PROPERTY(Integer, int);
SVTK_INFORMATION_DEFINE_VECTOR_VALUE_PROPERTY(Double, double);
SVTK_INFORMATION_DEFINE_VECTOR_VALUE2_PROPERTY(Variant, svtkVariant, const svtkVariant&);
#undef SVTK_INFORMATION_DEFINE_VECTOR_VALUE_PROPERTY

#undef SVTK_INFORMATION_DEFINE_VECTOR_PROPERTY

//----------------------------------------------------------------------------
#define SVTK_INFORMATION_DEFINE_POINTER_PROPERTY(name, type)                                        \
  void svtkInformation::Set(svtkInformation##name##PointerKey* key, type* value, int length)         \
  {                                                                                                \
    key->Set(this, value, length);                                                                 \
  }                                                                                                \
  type* svtkInformation::Get(svtkInformation##name##PointerKey* key) { return key->Get(this); }      \
  void svtkInformation::Get(svtkInformation##name##PointerKey* key, type* value)                     \
  {                                                                                                \
    key->Get(this, value);                                                                         \
  }                                                                                                \
  int svtkInformation::Length(svtkInformation##name##PointerKey* key) { return key->Length(this); }  \
  void svtkInformation::Remove(svtkInformation##name##PointerKey* key)                               \
  {                                                                                                \
    key->svtkInformation##name##PointerKey::Remove(this);                                           \
  }                                                                                                \
  int svtkInformation::Has(svtkInformation##name##PointerKey* key)                                   \
  {                                                                                                \
    return key->svtkInformation##name##PointerKey::Has(this);                                       \
  }
SVTK_INFORMATION_DEFINE_POINTER_PROPERTY(Integer, int);
#undef SVTK_INFORMATION_DEFINE_POINTER_PROPERTY

//----------------------------------------------------------------------------
void svtkInformation::Append(svtkInformationKeyVectorKey* key, svtkInformationDataObjectKey* value)
{
  key->Append(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::Append(svtkInformationKeyVectorKey* key, svtkInformationDoubleKey* value)
{
  key->Append(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::Append(svtkInformationKeyVectorKey* key, svtkInformationDoubleVectorKey* value)
{
  key->Append(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::Append(svtkInformationKeyVectorKey* key, svtkInformationInformationKey* value)
{
  key->Append(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::Remove(svtkInformationKeyVectorKey* key, svtkInformationKey* value)
{
  key->RemoveItem(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::Append(
  svtkInformationKeyVectorKey* key, svtkInformationInformationVectorKey* value)
{
  key->Append(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::Append(svtkInformationKeyVectorKey* key, svtkInformationIntegerKey* value)
{
  key->Append(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::Append(svtkInformationKeyVectorKey* key, svtkInformationIntegerVectorKey* value)
{
  key->Append(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::Append(svtkInformationKeyVectorKey* key, svtkInformationStringKey* value)
{
  key->Append(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::Append(svtkInformationKeyVectorKey* key, svtkInformationUnsignedLongKey* value)
{
  key->Append(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::Append(svtkInformationKeyVectorKey* key, svtkInformationObjectBaseKey* value)
{
  key->Append(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::Append(svtkInformationKeyVectorKey* key, svtkInformationStringVectorKey* value)
{
  key->Append(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::AppendUnique(
  svtkInformationKeyVectorKey* key, svtkInformationDataObjectKey* value)
{
  key->AppendUnique(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationKey* value)
{
  key->AppendUnique(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationDoubleKey* value)
{
  key->AppendUnique(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::AppendUnique(
  svtkInformationKeyVectorKey* key, svtkInformationDoubleVectorKey* value)
{
  key->AppendUnique(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::AppendUnique(
  svtkInformationKeyVectorKey* key, svtkInformationInformationKey* value)
{
  key->AppendUnique(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::AppendUnique(
  svtkInformationKeyVectorKey* key, svtkInformationInformationVectorKey* value)
{
  key->AppendUnique(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationIntegerKey* value)
{
  key->AppendUnique(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::AppendUnique(
  svtkInformationKeyVectorKey* key, svtkInformationIntegerVectorKey* value)
{
  key->AppendUnique(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationStringKey* value)
{
  key->AppendUnique(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::AppendUnique(
  svtkInformationKeyVectorKey* key, svtkInformationUnsignedLongKey* value)
{
  key->AppendUnique(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::AppendUnique(
  svtkInformationKeyVectorKey* key, svtkInformationObjectBaseKey* value)
{
  key->AppendUnique(this, value);
}

//----------------------------------------------------------------------------
void svtkInformation::AppendUnique(
  svtkInformationKeyVectorKey* key, svtkInformationStringVectorKey* value)
{
  key->AppendUnique(this, value);
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformation::GetKey(svtkInformationKey* key)
{
  return key;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformation::GetKey(svtkInformationDataObjectKey* key)
{
  return key;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformation::GetKey(svtkInformationInformationKey* key)
{
  return key;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformation::GetKey(svtkInformationInformationVectorKey* key)
{
  return key;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformation::GetKey(svtkInformationIntegerKey* key)
{
  return key;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformation::GetKey(svtkInformationRequestKey* key)
{
  return key;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformation::GetKey(svtkInformationDoubleKey* key)
{
  return key;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformation::GetKey(svtkInformationIntegerVectorKey* key)
{
  return key;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformation::GetKey(svtkInformationDoubleVectorKey* key)
{
  return key;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformation::GetKey(svtkInformationStringKey* key)
{
  return key;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformation::GetKey(svtkInformationStringVectorKey* key)
{
  return key;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformation::GetKey(svtkInformationUnsignedLongKey* key)
{
  return key;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformation::GetKey(svtkInformationVariantKey* key)
{
  return key;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformation::GetKey(svtkInformationVariantVectorKey* key)
{
  return key;
}

//----------------------------------------------------------------------------
void svtkInformation::Register(svtkObjectBase* o)
{
  this->RegisterInternal(o, 1);
}

//----------------------------------------------------------------------------
void svtkInformation::UnRegister(svtkObjectBase* o)
{
  this->UnRegisterInternal(o, 1);
}

//----------------------------------------------------------------------------
void svtkInformation::ReportReferences(svtkGarbageCollector* collector)
{
  this->Superclass::ReportReferences(collector);
  // Ask each key/value pair to report any references it holds.
  typedef svtkInformationInternals::MapType MapType;
  for (MapType::const_iterator i = this->Internal->Map.begin(); i != this->Internal->Map.end(); ++i)
  {
    i->first->Report(this, collector);
  }
}

//----------------------------------------------------------------------------
void svtkInformation::ReportAsObjectBase(svtkInformationKey* key, svtkGarbageCollector* collector)
{
  if (key)
  {
    typedef svtkInformationInternals::MapType MapType;
    MapType::iterator i = this->Internal->Map.find(key);
    if (i != this->Internal->Map.end())
    {
      svtkGarbageCollectorReport(collector, i->second, key->GetName());
    }
  }
}

//----------------------------------------------------------------------------
void svtkInformation::SetRequest(svtkInformationRequestKey* request)
{
  this->Request = request;
}

//----------------------------------------------------------------------------
svtkInformationRequestKey* svtkInformation::GetRequest()
{
  return this->Request;
}
