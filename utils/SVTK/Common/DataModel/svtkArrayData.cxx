/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayData.cxx

-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkArrayData.h"
#include "svtkArray.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"

#include <algorithm>
#include <vector>

//
// Standard functions
//

svtkStandardNewMacro(svtkArrayData);

class svtkArrayData::implementation
{
public:
  std::vector<svtkArray*> Arrays;
};

//----------------------------------------------------------------------------

svtkArrayData::svtkArrayData()
  : Implementation(new implementation())
{
}

//----------------------------------------------------------------------------

svtkArrayData::~svtkArrayData()
{
  this->ClearArrays();
  delete this->Implementation;
}

//----------------------------------------------------------------------------

void svtkArrayData::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  for (unsigned int i = 0; i != this->Implementation->Arrays.size(); ++i)
  {
    os << indent << "Array: " << this->Implementation->Arrays[i] << endl;
    this->Implementation->Arrays[i]->PrintSelf(os, indent.GetNextIndent());
  }
}

svtkArrayData* svtkArrayData::GetData(svtkInformation* info)
{
  return info ? svtkArrayData::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

svtkArrayData* svtkArrayData::GetData(svtkInformationVector* v, int i)
{
  return svtkArrayData::GetData(v->GetInformationObject(i));
}

void svtkArrayData::AddArray(svtkArray* array)
{
  if (!array)
  {
    svtkErrorMacro(<< "Cannot add nullptr array.");
    return;
  }

  // See http://developers.sun.com/solaris/articles/cmp_stlport_libCstd.html
  // Language Feature: Partial Specializations
  // Workaround

  int n = 0;
#ifdef _RWSTD_NO_CLASS_PARTIAL_SPEC
  std::count(this->Implementation->Arrays.begin(), this->Implementation->Arrays.end(), array, n);
#else
  n = std::count(this->Implementation->Arrays.begin(), this->Implementation->Arrays.end(), array);
#endif

  if (n != 0)
  {
    svtkErrorMacro(<< "Cannot add array twice.");
    return;
  }

  this->Implementation->Arrays.push_back(array);
  array->Register(nullptr);

  this->Modified();
}

void svtkArrayData::ClearArrays()
{
  for (unsigned int i = 0; i != this->Implementation->Arrays.size(); ++i)
  {
    this->Implementation->Arrays[i]->Delete();
  }

  this->Implementation->Arrays.clear();

  this->Modified();
}

svtkIdType svtkArrayData::GetNumberOfArrays()
{
  return static_cast<svtkIdType>(this->Implementation->Arrays.size());
}

svtkArray* svtkArrayData::GetArray(svtkIdType index)
{
  if (index < 0 || static_cast<size_t>(index) >= this->Implementation->Arrays.size())
  {
    svtkErrorMacro(<< "Array index out-of-range.");
    return nullptr;
  }

  return this->Implementation->Arrays[static_cast<size_t>(index)];
}

svtkArray* svtkArrayData::GetArrayByName(const char* name)
{
  if (!name || name[0] == '\0')
  {
    svtkErrorMacro(<< "No name passed into routine.");
    return nullptr;
  }

  svtkArray* temp = nullptr;
  for (svtkIdType ctr = 0; ctr < this->GetNumberOfArrays(); ctr++)
  {
    temp = this->GetArray(ctr);
    if (temp && !strcmp(name, temp->GetName()))
    {
      break;
    }
    temp = nullptr;
  }
  return temp;
}

void svtkArrayData::ShallowCopy(svtkDataObject* other)
{
  if (svtkArrayData* const array_data = svtkArrayData::SafeDownCast(other))
  {
    this->ClearArrays();
    this->Implementation->Arrays = array_data->Implementation->Arrays;
    for (size_t i = 0; i != this->Implementation->Arrays.size(); ++i)
    {
      this->Implementation->Arrays[i]->Register(this);
    }
    this->Modified();
  }

  Superclass::ShallowCopy(other);
}

void svtkArrayData::DeepCopy(svtkDataObject* other)
{
  if (svtkArrayData* const array_data = svtkArrayData::SafeDownCast(other))
  {
    this->ClearArrays();
    for (size_t i = 0; i != array_data->Implementation->Arrays.size(); ++i)
    {
      this->Implementation->Arrays.push_back(array_data->Implementation->Arrays[i]->DeepCopy());
    }
    this->Modified();
  }

  Superclass::DeepCopy(other);
}
