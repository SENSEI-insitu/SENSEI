/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAnnotation.cxx

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

#include "svtkAnnotation.h"
#include "svtkInformation.h"
#include "svtkInformationDataObjectKey.h"
#include "svtkInformationDoubleKey.h"
#include "svtkInformationDoubleVectorKey.h"
#include "svtkInformationIntegerKey.h"
#include "svtkInformationStringKey.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"
#include "svtkSelection.h"
#include "svtkSmartPointer.h"

svtkStandardNewMacro(svtkAnnotation);

svtkCxxSetObjectMacro(svtkAnnotation, Selection, svtkSelection);

svtkInformationKeyMacro(svtkAnnotation, LABEL, String);
svtkInformationKeyRestrictedMacro(svtkAnnotation, COLOR, DoubleVector, 3);
svtkInformationKeyMacro(svtkAnnotation, OPACITY, Double);
svtkInformationKeyMacro(svtkAnnotation, ICON_INDEX, Integer);
svtkInformationKeyMacro(svtkAnnotation, ENABLE, Integer);
svtkInformationKeyMacro(svtkAnnotation, HIDE, Integer);
svtkInformationKeyMacro(svtkAnnotation, DATA, DataObject);

svtkAnnotation::svtkAnnotation()
{
  this->Selection = nullptr;
}

svtkAnnotation::~svtkAnnotation()
{
  if (this->Selection)
  {
    this->Selection->Delete();
  }
}

void svtkAnnotation::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Selection: ";
  if (this->Selection)
  {
    os << "\n";
    this->Selection->PrintSelf(os, indent.GetNextIndent());
  }
  else
  {
    os << "(none)\n";
  }
}

void svtkAnnotation::Initialize()
{
  this->Superclass::Initialize();
}

void svtkAnnotation::ShallowCopy(svtkDataObject* other)
{
  this->Superclass::ShallowCopy(other);
  svtkAnnotation* obj = svtkAnnotation::SafeDownCast(other);
  if (!obj)
  {
    return;
  }
  this->SetSelection(obj->GetSelection());

  svtkInformation* info = this->GetInformation();
  svtkInformation* otherInfo = obj->GetInformation();
  if (otherInfo->Has(svtkAnnotation::ENABLE()))
  {
    info->CopyEntry(otherInfo, svtkAnnotation::ENABLE());
  }
  if (otherInfo->Has(svtkAnnotation::HIDE()))
  {
    info->CopyEntry(otherInfo, svtkAnnotation::HIDE());
  }
  if (otherInfo->Has(svtkAnnotation::LABEL()))
  {
    info->CopyEntry(otherInfo, svtkAnnotation::LABEL());
  }
  if (otherInfo->Has(svtkAnnotation::COLOR()))
  {
    info->CopyEntry(otherInfo, svtkAnnotation::COLOR());
  }
  if (otherInfo->Has(svtkAnnotation::OPACITY()))
  {
    info->CopyEntry(otherInfo, svtkAnnotation::OPACITY());
  }
  if (otherInfo->Has(svtkAnnotation::DATA()))
  {
    info->CopyEntry(otherInfo, svtkAnnotation::DATA());
  }
  if (otherInfo->Has(svtkAnnotation::ICON_INDEX()))
  {
    info->CopyEntry(otherInfo, svtkAnnotation::ICON_INDEX());
  }
}

void svtkAnnotation::DeepCopy(svtkDataObject* other)
{
  this->Superclass::DeepCopy(other);
  svtkAnnotation* obj = svtkAnnotation::SafeDownCast(other);
  if (!obj)
  {
    return;
  }
  svtkSmartPointer<svtkSelection> sel = svtkSmartPointer<svtkSelection>::New();
  sel->DeepCopy(obj->GetSelection());
  this->SetSelection(sel);

  svtkInformation* info = this->GetInformation();
  svtkInformation* otherInfo = obj->GetInformation();
  if (otherInfo->Has(svtkAnnotation::ENABLE()))
  {
    info->CopyEntry(otherInfo, svtkAnnotation::ENABLE());
  }
  if (otherInfo->Has(svtkAnnotation::HIDE()))
  {
    info->CopyEntry(otherInfo, svtkAnnotation::HIDE());
  }
  if (otherInfo->Has(svtkAnnotation::LABEL()))
  {
    info->CopyEntry(otherInfo, svtkAnnotation::LABEL());
  }
  if (otherInfo->Has(svtkAnnotation::COLOR()))
  {
    info->CopyEntry(otherInfo, svtkAnnotation::COLOR());
  }
  if (otherInfo->Has(svtkAnnotation::OPACITY()))
  {
    info->CopyEntry(otherInfo, svtkAnnotation::OPACITY());
  }
  if (otherInfo->Has(svtkAnnotation::DATA()))
  {
    info->CopyEntry(otherInfo, svtkAnnotation::DATA());
  }
  if (otherInfo->Has(svtkAnnotation::ICON_INDEX()))
  {
    info->CopyEntry(otherInfo, svtkAnnotation::ICON_INDEX());
  }
}

svtkMTimeType svtkAnnotation::GetMTime()
{
  svtkMTimeType mtime = this->Superclass::GetMTime();
  if (this->Selection)
  {
    svtkMTimeType stime = this->Selection->GetMTime();
    if (stime > mtime)
    {
      mtime = stime;
    }
  }
  return mtime;
}

svtkAnnotation* svtkAnnotation::GetData(svtkInformation* info)
{
  return info ? svtkAnnotation::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

svtkAnnotation* svtkAnnotation::GetData(svtkInformationVector* v, int i)
{
  return svtkAnnotation::GetData(v->GetInformationObject(i));
}
