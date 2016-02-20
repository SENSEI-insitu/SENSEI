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

#include "vtkDataObject.h"
#include "vtkInformation.h"
#include "vtkInformationIntegerKey.h"
#include "vtkObjectFactory.h"

vtkInformationKeyMacro(vtkInsituDataAdaptor, DATA_TIME_STEP_INDEX, Integer);
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
int vtkInsituDataAdaptor::GetDataTimeStep(vtkInformation* info)
{
  return info->Has(vtkInsituDataAdaptor::DATA_TIME_STEP_INDEX())?
    info->Get(vtkInsituDataAdaptor::DATA_TIME_STEP_INDEX()) : 0;
}

//----------------------------------------------------------------------------
void vtkInsituDataAdaptor::SetDataTimeStep(vtkInformation* info, int index)
{
  info->Set(vtkInsituDataAdaptor::DATA_TIME_STEP_INDEX(), index);
}

//----------------------------------------------------------------------------
vtkDataObject* vtkInsituDataAdaptor::GetCompleteMesh()
{
  vtkDataObject* mesh = this->GetMesh();
  for (int attr=vtkDataObject::FIELD_ASSOCIATION_POINTS;
    attr < vtkDataObject::NUMBER_OF_ASSOCIATIONS; ++attr)
    {
    for (unsigned int cc=0, max=this->GetNumberOfArrays(attr); cc < max; ++cc)
      {
      this->AddArray(mesh, attr, this->GetArrayName(attr, cc));
      }
    }
  return mesh;
}

//----------------------------------------------------------------------------
void vtkInsituDataAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
