/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationInternals.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationInternals
 * @brief   internal structure for svtkInformation
 *
 * svtkInformationInternals is used in internal implementation of
 * svtkInformation. This should only be accessed by friends
 * and sub-classes of that class.
 */

#ifndef svtkInformationInternals_h
#define svtkInformationInternals_h

#include "svtkInformationKey.h"
#include "svtkObjectBase.h"

#define SVTK_INFORMATION_USE_HASH_MAP
#ifdef SVTK_INFORMATION_USE_HASH_MAP
#include <unordered_map>
#else
#include <map>
#endif

//----------------------------------------------------------------------------
class svtkInformationInternals
{
public:
  typedef svtkInformationKey* KeyType;
  typedef svtkObjectBase* DataType;
#ifdef SVTK_INFORMATION_USE_HASH_MAP
  struct HashFun
  {
    size_t operator()(KeyType key) const {
// TODO : clang warns about undefined behavior below.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnull-pointer-subtraction"
        return static_cast<size_t>(key - KeyType(nullptr)); }
#pragma clang diagnostic pop
  };
  typedef std::unordered_map<KeyType, DataType, HashFun> MapType;
#else
  typedef std::map<KeyType, DataType> MapType;
#endif
  MapType Map;

#ifdef SVTK_INFORMATION_USE_HASH_MAP
  svtkInformationInternals()
    : Map(33)
  {
  }
#endif

  ~svtkInformationInternals()
  {
    for (MapType::iterator i = this->Map.begin(); i != this->Map.end(); ++i)
    {
      if (svtkObjectBase* value = i->second)
      {
        value->UnRegister(nullptr);
      }
    }
  }

private:
  svtkInformationInternals(svtkInformationInternals const&) = delete;
};

#undef SVTK_INFORMATION_USE_HASH_MAP

#endif
// SVTK-HeaderTest-Exclude: svtkInformationInternals.h
