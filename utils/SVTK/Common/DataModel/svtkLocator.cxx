/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLocator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkLocator.h"

#include "svtkDataSet.h"
#include "svtkGarbageCollector.h"

svtkCxxSetObjectMacro(svtkLocator, DataSet, svtkDataSet);

svtkLocator::svtkLocator()
{
  this->DataSet = nullptr;
  this->Tolerance = 0.001;
  this->Automatic = 1;
  this->MaxLevel = 8;
  this->Level = 8;
}

svtkLocator::~svtkLocator()
{
  // commented out because of compiler problems in g++
  //  this->FreeSearchStructure();
  this->SetDataSet(nullptr);
}

void svtkLocator::Initialize()
{
  // free up hash table
  this->FreeSearchStructure();
}

void svtkLocator::Update()
{
  if (!this->DataSet)
  {
    svtkErrorMacro(<< "Input not set!");
    return;
  }
  if ((this->MTime > this->BuildTime) || (this->DataSet->GetMTime() > this->BuildTime))
  {
    this->BuildLocator();
  }
}

void svtkLocator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  if (this->DataSet)
  {
    os << indent << "DataSet: " << this->DataSet << "\n";
  }
  else
  {
    os << indent << "DataSet: (none)\n";
  }

  os << indent << "Automatic: " << (this->Automatic ? "On\n" : "Off\n");
  os << indent << "Tolerance: " << this->Tolerance << "\n";
  os << indent << "Build Time: " << this->BuildTime.GetMTime() << "\n";
  os << indent << "MaxLevel: " << this->MaxLevel << "\n";
  os << indent << "Level: " << this->Level << "\n";
}

//----------------------------------------------------------------------------
void svtkLocator::Register(svtkObjectBase* o)
{
  this->RegisterInternal(o, 1);
}

//----------------------------------------------------------------------------
void svtkLocator::UnRegister(svtkObjectBase* o)
{
  this->UnRegisterInternal(o, 1);
}

//----------------------------------------------------------------------------
void svtkLocator::ReportReferences(svtkGarbageCollector* collector)
{
  this->Superclass::ReportReferences(collector);
  svtkGarbageCollectorReport(collector, this->DataSet, "DataSet");
}
