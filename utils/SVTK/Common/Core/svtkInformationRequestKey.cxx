/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationRequestKey.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationRequestKey.h"

#include "svtkInformation.h"

//----------------------------------------------------------------------------
svtkInformationRequestKey::svtkInformationRequestKey(const char* name, const char* location)
  : svtkInformationKey(name, location)
{
  svtkCommonInformationKeyManager::Register(this);
}

//----------------------------------------------------------------------------
svtkInformationRequestKey::~svtkInformationRequestKey() = default;

//----------------------------------------------------------------------------
void svtkInformationRequestKey::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
void svtkInformationRequestKey::Set(svtkInformation* info)
{
  if (info->GetRequest() != this)
  {
    if (info->GetRequest())
    {
      svtkGenericWarningMacro("Setting request key when one is already set. Current request is "
        << info->GetRequest()->GetName() << " while setting " << this->GetName() << "\n");
    }
    info->SetRequest(this);
    info->Modified(this);
  }
}

//----------------------------------------------------------------------------
int svtkInformationRequestKey::Has(svtkInformation* info)
{
  return (info->GetRequest() == this) ? 1 : 0;
}

//----------------------------------------------------------------------------
void svtkInformationRequestKey::Remove(svtkInformation* info)
{
  info->SetRequest(nullptr);
}

//----------------------------------------------------------------------------
void svtkInformationRequestKey::ShallowCopy(svtkInformation* from, svtkInformation* to)
{
  to->SetRequest(from->GetRequest());
}

//----------------------------------------------------------------------------
void svtkInformationRequestKey::Print(ostream& os, svtkInformation* info)
{
  // Print the value.
  if (this->Has(info))
  {
    os << "1\n";
  }
}
