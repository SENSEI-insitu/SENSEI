/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArraySelection.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkDataArraySelection.h"
#include "svtkObjectFactory.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

svtkStandardNewMacro(svtkDataArraySelection);

class svtkDataArraySelectionInternals
{
public:
  using ArraySettingPair = std::pair<std::string, bool>;
  using ArraysType = std::vector<ArraySettingPair>;
  ArraysType Arrays;

  ArraysType::iterator Find(const char* name)
  {
    return name != nullptr ? this->Find(std::string(name)) : this->Arrays.end();
  }

  ArraysType::iterator Find(const std::string& name)
  {
    return std::find_if(this->Arrays.begin(), this->Arrays.end(),
      [&](const std::pair<std::string, bool>& item) -> bool { return (item.first == name); });
  }
};

//----------------------------------------------------------------------------
svtkDataArraySelection::svtkDataArraySelection()
{
  this->Internal = new svtkDataArraySelectionInternals;
  this->UnknownArraySetting = 0;
}

//----------------------------------------------------------------------------
svtkDataArraySelection::~svtkDataArraySelection()
{
  delete this->Internal;
}

//----------------------------------------------------------------------------
void svtkDataArraySelection::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "UnknownArraySetting: " << this->UnknownArraySetting << endl;
  os << indent << "Number of Arrays: " << this->GetNumberOfArrays() << "\n";
  svtkIndent nindent = indent.GetNextIndent();
  int cc;
  for (cc = 0; cc < this->GetNumberOfArrays(); cc++)
  {
    os << nindent << "Array: " << this->GetArrayName(cc)
       << " is: " << (this->GetArraySetting(cc) ? "enabled" : "disabled") << " ("
       << this->ArrayIsEnabled(this->GetArrayName(cc)) << ")" << endl;
  }
}

//----------------------------------------------------------------------------
void svtkDataArraySelection::EnableArray(const char* name)
{
  svtkDebugMacro("Enabling array \"" << name << "\".");
  this->SetArraySetting(name, 1);
}

//----------------------------------------------------------------------------
void svtkDataArraySelection::DisableArray(const char* name)
{
  svtkDebugMacro("Disabling array \"" << name << "\".");
  this->SetArraySetting(name, 0);
}

//----------------------------------------------------------------------------
void svtkDataArraySelection::SetArraySetting(const char* name, int setting)
{
  svtkDebugMacro("Setting array \"" << name << " = " << setting << "\".");
  const bool status = (setting > 0);
  auto& internal = *this->Internal;
  auto iter = internal.Find(name);
  if (iter != internal.Arrays.end())
  {
    if (iter->second != status)
    {
      iter->second = status;
      this->Modified();
    }
  }
  else if (name)
  {
    internal.Arrays.push_back(svtkDataArraySelectionInternals::ArraySettingPair(name, status));
    this->Modified();
  }
}

//----------------------------------------------------------------------------
int svtkDataArraySelection::ArrayIsEnabled(const char* name) const
{
  auto iter = this->Internal->Find(name);
  if (iter != this->Internal->Arrays.end())
  {
    return iter->second ? 1 : 0;
  }

  // The array does not have an entry.  Return `UnknownArraySetting`.
  return this->UnknownArraySetting;
}

//----------------------------------------------------------------------------
int svtkDataArraySelection::ArrayExists(const char* name) const
{
  // Check if there is a specific entry for this array.
  return this->Internal->Find(name) != this->Internal->Arrays.end() ? 1 : 0;
}

//----------------------------------------------------------------------------
void svtkDataArraySelection::EnableAllArrays()
{
  svtkDebugMacro("Enabling all arrays.");
  int modified = 0;
  for (auto& apair : this->Internal->Arrays)
  {
    if (!apair.second)
    {
      apair.second = true;
      modified = 1;
    }
  }

  if (modified)
  {
    this->Modified();
  }
}

//----------------------------------------------------------------------------
void svtkDataArraySelection::DisableAllArrays()
{
  svtkDebugMacro("Disabling all arrays.");
  int modified = 0;
  for (auto& apair : this->Internal->Arrays)
  {
    if (apair.second)
    {
      apair.second = false;
      modified = 1;
    }
  }

  if (modified)
  {
    this->Modified();
  }
}

//----------------------------------------------------------------------------
int svtkDataArraySelection::GetNumberOfArrays() const
{
  return static_cast<int>(this->Internal->Arrays.size());
}

//----------------------------------------------------------------------------
int svtkDataArraySelection::GetNumberOfArraysEnabled() const
{
  int numArrays = 0;
  for (const auto& apair : this->Internal->Arrays)
  {
    numArrays += (apair.second) ? 1 : 0;
  }

  return numArrays;
}

//----------------------------------------------------------------------------
const char* svtkDataArraySelection::GetArrayName(int index) const
{
  try
  {
    return this->Internal->Arrays.at(index).first.c_str();
  }
  catch (std::out_of_range&)
  {
    return nullptr;
  }
}

//----------------------------------------------------------------------------
int svtkDataArraySelection::GetArrayIndex(const char* name) const
{
  auto iter = this->Internal->Find(name);
  if (iter != this->Internal->Arrays.end())
  {
    return static_cast<int>(iter - this->Internal->Arrays.begin());
  }

  return -1;
}

//----------------------------------------------------------------------------
int svtkDataArraySelection::GetEnabledArrayIndex(const char* name) const
{
  int index = 0;
  for (const auto& apair : this->Internal->Arrays)
  {
    if (apair.first == name)
    {
      return index;
    }
    else if (apair.second)
    {
      ++index;
    }
  }
  return -1;
}

//----------------------------------------------------------------------------
int svtkDataArraySelection::GetArraySetting(int index) const
{
  if (index >= 0 && index < this->GetNumberOfArrays())
  {
    return this->Internal->Arrays[index].second ? 1 : 0;
  }

  return 0;
}

//----------------------------------------------------------------------------
void svtkDataArraySelection::RemoveAllArrays()
{
  svtkDebugMacro("Removing all arrays.");
  if (!this->Internal->Arrays.empty())
  {
    this->Internal->Arrays.clear();
    this->Modified();
  }
}

//----------------------------------------------------------------------------
int svtkDataArraySelection::AddArray(const char* name, bool state)
{
  svtkDebugMacro("Adding array \"" << name << "\".");
  // This function is called only by the filter owning the selection.
  // It should not call Modified() because array settings are not
  // changed.
  if (this->ArrayExists(name))
  {
    return 0;
  }
  this->Internal->Arrays.push_back(svtkDataArraySelectionInternals::ArraySettingPair(name, state));
  return 1;
}

//----------------------------------------------------------------------------
void svtkDataArraySelection::RemoveArrayByIndex(int index)
{
  if (index >= 0 && index < static_cast<int>(this->Internal->Arrays.size()))
  {
    this->Internal->Arrays.erase(this->Internal->Arrays.begin() + index);
  }
}

//----------------------------------------------------------------------------
void svtkDataArraySelection::RemoveArrayByName(const char* name)
{
  auto iter = this->Internal->Find(name);
  if (iter != this->Internal->Arrays.end())
  {
    this->Internal->Arrays.erase(iter);
  }
}

//----------------------------------------------------------------------------
void svtkDataArraySelection::SetArrays(const char* const* names, int numArrays)
{
  this->SetArraysWithDefault(names, numArrays, 1);
}

//----------------------------------------------------------------------------
void svtkDataArraySelection::SetArraysWithDefault(
  const char* const* names, int numArrays, int defaultStatus)
{
  // This function is called only by the filter owning the selection.
  // It should not call Modified() because array settings are not
  // changed.

  svtkDebugMacro("Settings arrays to given list of " << numArrays << " arrays.");

  // Create a new map for this set of arrays.
  svtkDataArraySelectionInternals* newInternal = new svtkDataArraySelectionInternals;

  newInternal->Arrays.reserve(numArrays);

  // Fill with settings for all arrays.
  for (int i = 0; i < numArrays; ++i)
  {
    // Fill in the setting.  Use the old value if available.
    // Otherwise, use the given default.
    bool setting = defaultStatus ? true : false;
    auto iter = this->Internal->Find(names[i]);
    if (iter != this->Internal->Arrays.end())
    {
      setting = iter->second;
    }
    newInternal->Arrays.push_back(
      svtkDataArraySelectionInternals::ArraySettingPair(names[i], setting));
  }

  // Delete the old map and save the new one.
  delete this->Internal;
  this->Internal = newInternal;
}

//----------------------------------------------------------------------------
void svtkDataArraySelection::CopySelections(svtkDataArraySelection* selections)
{
  if (this == selections)
  {
    return;
  }

  int needUpdate = 0;
  int i;
  const char* arrayName;
  if (this->GetNumberOfArrays() != selections->GetNumberOfArrays())
  {
    needUpdate = 1;
  }
  else
  {
    for (i = 0; i < this->GetNumberOfArrays(); i++)
    {
      arrayName = this->GetArrayName(i);
      if (!selections->ArrayExists(arrayName))
      {
        needUpdate = 1;
        break;
      }
      if (selections->ArrayIsEnabled(arrayName) != this->ArrayIsEnabled(arrayName))
      {
        needUpdate = 1;
        break;
      }
    }
  }

  if (!needUpdate)
  {
    return;
  }

  svtkDebugMacro("Copying arrays and settings from " << selections << ".");
  this->Internal->Arrays = selections->Internal->Arrays;
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkDataArraySelection::Union(svtkDataArraySelection* other)
{
  auto& internal = *this->Internal;
  const auto& ointernal = *other->Internal;

  bool modified = false;
  for (const auto& apair : ointernal.Arrays)
  {
    auto iter = internal.Find(apair.first);
    if (iter == internal.Arrays.end())
    {
      internal.Arrays.push_back(apair);
      modified = true;
    }
  }

  if (modified)
  {
    this->Modified();
  }
}
