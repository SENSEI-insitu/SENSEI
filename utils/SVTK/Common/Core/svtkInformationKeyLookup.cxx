/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationKeyLookup.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkInformationKeyLookup.h"

#include "svtkInformationKey.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkInformationKeyLookup);

//------------------------------------------------------------------------------
void svtkInformationKeyLookup::PrintSelf(std::ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Registered Keys:\n";
  indent = indent.GetNextIndent();
  KeyMap& keys = Keys();
  for (KeyMap::iterator i = keys.begin(), iEnd = keys.end(); i != iEnd; ++i)
  {
    os << indent << i->first.first << "::" << i->first.second << " @" << i->second << " ("
       << i->second->GetClassName() << ")\n";
  }
}

//------------------------------------------------------------------------------
svtkInformationKey* svtkInformationKeyLookup::Find(
  const std::string& name, const std::string& location)
{
  KeyMap& keys = Keys();
  KeyMap::iterator it = keys.find(std::make_pair(location, name));
  return it != keys.end() ? it->second : nullptr;
}

//------------------------------------------------------------------------------
svtkInformationKeyLookup::svtkInformationKeyLookup() = default;

//------------------------------------------------------------------------------
svtkInformationKeyLookup::~svtkInformationKeyLookup()
{
  // Keys are owned / cleaned up by the svtk*InformationKeyManagers.
}

//------------------------------------------------------------------------------
void svtkInformationKeyLookup::RegisterKey(
  svtkInformationKey* key, const std::string& name, const std::string& location)
{
  svtkInformationKeyLookup::Keys().insert(std::make_pair(std::make_pair(location, name), key));
}

//------------------------------------------------------------------------------
svtkInformationKeyLookup::KeyMap& svtkInformationKeyLookup::Keys()
{
  // Ensure that the map is initialized before using from other static
  // initializations:
  static svtkInformationKeyLookup::KeyMap keys;
  return keys;
}
