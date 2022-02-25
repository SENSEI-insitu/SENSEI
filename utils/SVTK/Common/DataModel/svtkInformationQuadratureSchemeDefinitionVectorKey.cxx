/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationQuadratureSchemeDefinitionVectorKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationQuadratureSchemeDefinitionVectorKey.h"
#include "svtkCellType.h"
#include "svtkInformation.h"
#include "svtkQuadratureSchemeDefinition.h"
#include "svtkSmartPointer.h"
#include "svtkXMLDataElement.h"
#include <vector>

//============================================================================
class svtkInformationQuadratureSchemeDefinitionVectorValue : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkInformationQuadratureSchemeDefinitionVectorValue, svtkObjectBase);
  //
  svtkInformationQuadratureSchemeDefinitionVectorValue()
  {
    // Typically we have one definition per cell type.
    this->Vector.resize(SVTK_NUMBER_OF_CELL_TYPES);
  }
  // Accessor.
  std::vector<svtkSmartPointer<svtkQuadratureSchemeDefinition> >& GetVector() { return this->Vector; }

private:
  std::vector<svtkSmartPointer<svtkQuadratureSchemeDefinition> > Vector;
};

//============================================================================

//----------------------------------------------------------------------------
svtkInformationQuadratureSchemeDefinitionVectorKey::
  svtkInformationQuadratureSchemeDefinitionVectorKey(const char* name, const char* location)
  : svtkInformationKey(name, location)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationQuadratureSchemeDefinitionVectorKey::
  ~svtkInformationQuadratureSchemeDefinitionVectorKey() = default;

//----------------------------------------------------------------------------
svtkInformationQuadratureSchemeDefinitionVectorValue*
svtkInformationQuadratureSchemeDefinitionVectorKey::GetQuadratureSchemeDefinitionVector(
  svtkInformation* info)
{
  // Grab the vector associated with this key.
  svtkInformationQuadratureSchemeDefinitionVectorValue* base =
    static_cast<svtkInformationQuadratureSchemeDefinitionVectorValue*>(this->GetAsObjectBase(info));

  // If we don't already have a vector then associated,
  // we will create it here.
  if (base == nullptr)
  {
    base = new svtkInformationQuadratureSchemeDefinitionVectorValue;
    base->InitializeObjectBase();
    this->SetAsObjectBase(info, base);
    base->Delete();
  }

  return base;
}

//----------------------------------------------------------------------------
void svtkInformationQuadratureSchemeDefinitionVectorKey::Append(
  svtkInformation* info, svtkQuadratureSchemeDefinition* aValue)
{
  //
  svtkInformationQuadratureSchemeDefinitionVectorValue* base =
    this->GetQuadratureSchemeDefinitionVector(info);
  //
  base->GetVector().push_back(aValue);
}

//----------------------------------------------------------------------------
void svtkInformationQuadratureSchemeDefinitionVectorKey::Set(
  svtkInformation* info, svtkQuadratureSchemeDefinition* aValue, int i)
{
  // Get the vector associated with this key, resize if this
  // set would run off the end.
  svtkInformationQuadratureSchemeDefinitionVectorValue* base =
    this->GetQuadratureSchemeDefinitionVector(info);
  int n = static_cast<int>(base->GetVector().size());
  if (i >= n)
  {
    base->GetVector().resize(i + 1);
  }
  // Set.
  base->GetVector()[i] = aValue;
}

//----------------------------------------------------------------------------
void svtkInformationQuadratureSchemeDefinitionVectorKey::SetRange(
  svtkInformation* info, svtkQuadratureSchemeDefinition** sourceVec, int from, int to, int n)
{
  // Get the vector associated with this key, resize if this
  // set would run off the end.
  svtkInformationQuadratureSchemeDefinitionVectorValue* base =
    this->GetQuadratureSchemeDefinitionVector(info);
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
// svtkSmartPointer<svtkQuadratureSchemeDefinition>
// *svtkInformationQuadratureSchemeDefinitionVectorKey::Get(
//         svtkInformation* info)
// {
//   svtkInformationQuadratureSchemeDefinitionVectorValue* base =
//     static_cast<svtkInformationQuadratureSchemeDefinitionVectorValue
//     *>(this->GetAsObjectBase(info));
//
//   return
//     (base!=nullptr && !base->GetVector().empty())?(&base->GetVector()[0]):0;
// }

//----------------------------------------------------------------------------
void svtkInformationQuadratureSchemeDefinitionVectorKey::GetRange(
  svtkInformation* info, svtkQuadratureSchemeDefinition** dest, int from, int to, int n)
{
  svtkInformationQuadratureSchemeDefinitionVectorValue* base =
    static_cast<svtkInformationQuadratureSchemeDefinitionVectorValue*>(this->GetAsObjectBase(info));

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

  // limit copy to what's there.
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
svtkQuadratureSchemeDefinition* svtkInformationQuadratureSchemeDefinitionVectorKey::Get(
  svtkInformation* info, int idx)
{
  svtkInformationQuadratureSchemeDefinitionVectorValue* base =
    static_cast<svtkInformationQuadratureSchemeDefinitionVectorValue*>(this->GetAsObjectBase(info));

  if (base == nullptr || idx >= static_cast<int>(base->GetVector().size()))
  {
    svtkErrorWithObjectMacro(info,
      "Information does not contain " << idx << " elements. Cannot return information value.");
    return nullptr;
  }

  return base->GetVector()[idx];
}

//----------------------------------------------------------------------------
int svtkInformationQuadratureSchemeDefinitionVectorKey::Size(svtkInformation* info)
{
  svtkInformationQuadratureSchemeDefinitionVectorValue* base =
    static_cast<svtkInformationQuadratureSchemeDefinitionVectorValue*>(this->GetAsObjectBase(info));

  return (base == nullptr ? 0 : static_cast<int>(base->GetVector().size()));
}

//----------------------------------------------------------------------------
void svtkInformationQuadratureSchemeDefinitionVectorKey::Resize(svtkInformation* info, int size)
{
  svtkInformationQuadratureSchemeDefinitionVectorValue* base =
    this->GetQuadratureSchemeDefinitionVector(info);
  base->GetVector().resize(size);
}

//----------------------------------------------------------------------------
void svtkInformationQuadratureSchemeDefinitionVectorKey::Clear(svtkInformation* info)
{
  svtkInformationQuadratureSchemeDefinitionVectorValue* base =
    this->GetQuadratureSchemeDefinitionVector(info);
  base->GetVector().clear();
}

//----------------------------------------------------------------------------
void svtkInformationQuadratureSchemeDefinitionVectorKey::ShallowCopy(
  svtkInformation* source, svtkInformation* dest)
{
  // grab the source vector
  svtkInformationQuadratureSchemeDefinitionVectorValue* sourceBase =
    static_cast<svtkInformationQuadratureSchemeDefinitionVectorValue*>(
      this->GetAsObjectBase(source));
  // grab failed, just set dest to 0
  if (sourceBase == nullptr)
  {
    this->SetAsObjectBase(dest, nullptr);
    return;
  }
  // Grab the dest vector
  svtkInformationQuadratureSchemeDefinitionVectorValue* destBase =
    this->GetQuadratureSchemeDefinitionVector(dest);
  // Explicitly size the dest
  int sourceSize = static_cast<int>(sourceBase->GetVector().size());
  destBase->GetVector().resize(sourceSize);
  // Vector operator= copy, using smart ptrs we get ref counted.
  destBase->GetVector() = sourceBase->GetVector();
}

//----------------------------------------------------------------------------
void svtkInformationQuadratureSchemeDefinitionVectorKey::DeepCopy(
  svtkInformation* source, svtkInformation* dest)
{
  // Grab the source vector.
  svtkInformationQuadratureSchemeDefinitionVectorValue* sourceBase =
    static_cast<svtkInformationQuadratureSchemeDefinitionVectorValue*>(
      this->GetAsObjectBase(source));
  // Grab failed, set dest to 0 and bail.
  if (sourceBase == nullptr)
  {
    this->SetAsObjectBase(dest, nullptr);
    return;
  }
  // Grab the dest vector.
  svtkInformationQuadratureSchemeDefinitionVectorValue* destBase =
    this->GetQuadratureSchemeDefinitionVector(dest);
  // Explicitly size the dest.
  int sourceSize = static_cast<int>(sourceBase->GetVector().size());
  destBase->GetVector().resize(sourceSize);
  // Deep copy each definition.
  for (int i = 0; i < sourceSize; ++i)
  {
    svtkQuadratureSchemeDefinition* srcDef = sourceBase->GetVector()[i];
    if (srcDef)
    {
      svtkQuadratureSchemeDefinition* destDef = svtkQuadratureSchemeDefinition::New();
      destDef->DeepCopy(srcDef);
      destBase->GetVector()[i] = destDef;
      destDef->Delete();
    }
  }
}

//-----------------------------------------------------------------------------
int svtkInformationQuadratureSchemeDefinitionVectorKey::SaveState(
  svtkInformation* info, svtkXMLDataElement* root)
{
  // Grab the vector associated with this key.
  svtkInformationQuadratureSchemeDefinitionVectorValue* base =
    static_cast<svtkInformationQuadratureSchemeDefinitionVectorValue*>(this->GetAsObjectBase(info));

  // If it doesn't exist or it's empty then we do nothing.
  int dictSize;
  if (base == nullptr || (dictSize = static_cast<int>(base->GetVector().size())) == 0)
  {
    svtkGenericWarningMacro("Attempting to save an empty or non-existent key/value.");
    return 0;
  }

  // Quick sanity check, we're not nesting rather treating
  // this as a root, to be nested by the caller as needed.
  if (root->GetName() != nullptr || root->GetNumberOfNestedElements() > 0)
  {
    svtkGenericWarningMacro("Can't save state to non-empty element.");
    return 0;
  }

  // Initialize the key
  root->SetName("InformationKey");
  root->SetAttribute("name", "DICTIONARY");
  root->SetAttribute("location", "svtkQuadratureSchemeDefinition");
  // For each item in the array.
  for (int defnId = 0; defnId < dictSize; ++defnId)
  {
    // Grab a definition.
    svtkQuadratureSchemeDefinition* def = base->GetVector()[defnId];
    if (def == nullptr)
    {
      continue;
    }
    // Nest XML representation.
    svtkXMLDataElement* e = svtkXMLDataElement::New();
    def->SaveState(e);
    root->AddNestedElement(e);
    e->Delete();
  }
  return 1;
}

//-----------------------------------------------------------------------------
int svtkInformationQuadratureSchemeDefinitionVectorKey::RestoreState(
  svtkInformation* info, svtkXMLDataElement* root)
{
  // Grab or create the vector associated with this key.
  svtkInformationQuadratureSchemeDefinitionVectorValue* base =
    this->GetQuadratureSchemeDefinitionVector(info);
  // clear what ever state we have to avoid confusion.
  base->GetVector().clear();
  base->GetVector().resize(SVTK_NUMBER_OF_CELL_TYPES);

  // Quick sanity check to validate that we were passed a
  // valid dictionary tag.
  if ((strcmp(root->GetName(), "InformationKey") != 0) ||
    (strcmp(root->GetAttribute("name"), "DICTIONARY") != 0) ||
    (strcmp(root->GetAttribute("location"), "svtkQuadratureSchemeDefinition") != 0))
  {
    svtkGenericWarningMacro("State cannot be loaded from <"
      << root->GetName() << " "
      << "name=\"" << root->GetAttribute("name") << "\" "
      << "location=\"" << root->GetAttribute("location") << "\".");
    return 0;
  }
  // Process all nested tags. Each is assumed to be a valid definition
  // tag. If any of the tags are invalid or not definition tags they
  // will be skipped, and warnings will be generated.
  int nDefns = root->GetNumberOfNestedElements();
  for (int defnId = 0; defnId < nDefns; ++defnId)
  {
    svtkXMLDataElement* e = root->GetNestedElement(defnId);
    svtkQuadratureSchemeDefinition* def = svtkQuadratureSchemeDefinition::New();
    if (def->RestoreState(e))
    {
      base->GetVector()[def->GetCellType()] = def;
    }
    def->Delete();
  }

  return 1;
}

//----------------------------------------------------------------------------
void svtkInformationQuadratureSchemeDefinitionVectorKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//-----------------------------------------------------------------------------
void svtkInformationQuadratureSchemeDefinitionVectorKey::Print(ostream& os, svtkInformation* info)
{
  svtkIndent indent;
  // Grab the vector associated with this key.
  svtkInformationQuadratureSchemeDefinitionVectorValue* base =
    static_cast<svtkInformationQuadratureSchemeDefinitionVectorValue*>(this->GetAsObjectBase(info));
  // Print each valid item.
  if (base != nullptr)
  {
    int n = static_cast<int>(base->GetVector().size());
    for (int i = 0; i < n; ++i)
    {
      os << indent << "item " << i << "=";
      svtkQuadratureSchemeDefinition* itemBase = base->GetVector()[i];
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
