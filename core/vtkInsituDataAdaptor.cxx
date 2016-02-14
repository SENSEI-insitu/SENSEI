/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkInsituDataAdaptor.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkInsituDataAdaptor.h"

#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"

//----------------------------------------------------------------------------
vtkInsituDataAdaptor::vtkInsituDataAdaptor()
{
  this->Information = vtkInformation::New();
}

//----------------------------------------------------------------------------
vtkInsituDataAdaptor::~vtkInsituDataAdaptor()
{
  this->Information->Delete();
}

//----------------------------------------------------------------------------
double vtkInsituDataAdaptor::GetDataTime(vtkInformation* info)
{
  return info->Has(vtkDataObject::DATA_TIME_STEP())?
    info->Get(vtkDataObject::DATA_TIME_STEP()) : 0.0;
}

//----------------------------------------------------------------------------
void vtkInsituDataAdaptor::SetDataTime(vtkInformation* info, double time)
{
  info->Set(vtkDataObject::DATA_TIME_STEP(), time);
}

//----------------------------------------------------------------------------
void vtkInsituDataAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
