/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationIterator.h"

#include "svtkInformation.h"
#include "svtkInformationInternals.h"
#include "svtkInformationKey.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkInformationIterator);

class svtkInformationIteratorInternals
{
public:
  svtkInformationInternals::MapType::iterator Iterator;
};

//----------------------------------------------------------------------------
svtkInformationIterator::svtkInformationIterator()
{
  this->Internal = new svtkInformationIteratorInternals;
  this->Information = nullptr;
  this->ReferenceIsWeak = false;
}

//----------------------------------------------------------------------------
svtkInformationIterator::~svtkInformationIterator()
{
  if (this->ReferenceIsWeak)
  {
    this->Information = nullptr;
  }
  if (this->Information)
  {
    this->Information->Delete();
  }
  delete this->Internal;
}

//----------------------------------------------------------------------------
void svtkInformationIterator::SetInformation(svtkInformation* inf)
{
  if (this->ReferenceIsWeak)
  {
    this->Information = nullptr;
  }
  this->ReferenceIsWeak = false;
  svtkSetObjectBodyMacro(Information, svtkInformation, inf);
}

//----------------------------------------------------------------------------
void svtkInformationIterator::SetInformationWeak(svtkInformation* inf)
{
  if (!this->ReferenceIsWeak)
  {
    this->SetInformation(nullptr);
  }

  this->ReferenceIsWeak = true;

  if (this->Information != inf)
  {
    this->Information = inf;
    this->Modified();
  }
}

//----------------------------------------------------------------------------
void svtkInformationIterator::GoToFirstItem()
{
  if (!this->Information)
  {
    svtkErrorMacro("No information has been set.");
    return;
  }
  this->Internal->Iterator = this->Information->Internal->Map.begin();
}

//----------------------------------------------------------------------------
void svtkInformationIterator::GoToNextItem()
{
  if (!this->Information)
  {
    svtkErrorMacro("No information has been set.");
    return;
  }

  ++this->Internal->Iterator;
}

//----------------------------------------------------------------------------
int svtkInformationIterator::IsDoneWithTraversal()
{
  if (!this->Information)
  {
    svtkErrorMacro("No information has been set.");
    return 1;
  }

  if (this->Internal->Iterator == this->Information->Internal->Map.end())
  {
    return 1;
  }
  return 0;
}

//----------------------------------------------------------------------------
svtkInformationKey* svtkInformationIterator::GetCurrentKey()
{
  if (this->IsDoneWithTraversal())
  {
    return nullptr;
  }

  return this->Internal->Iterator->first;
}

//----------------------------------------------------------------------------
void svtkInformationIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Information: ";
  if (this->Information)
  {
    os << endl;
    this->Information->PrintSelf(os, indent.GetNextIndent());
  }
  else
  {
    os << "(none)" << endl;
  }
}
