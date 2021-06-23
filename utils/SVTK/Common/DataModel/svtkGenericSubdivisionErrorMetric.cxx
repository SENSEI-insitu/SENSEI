/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericSubdivisionErrorMetric.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkGenericSubdivisionErrorMetric.h"

#include "svtkGenericAdaptorCell.h"
#include "svtkGenericAttribute.h"
#include "svtkGenericAttributeCollection.h"
#include "svtkGenericDataSet.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include <cassert>

//-----------------------------------------------------------------------------
svtkGenericSubdivisionErrorMetric::svtkGenericSubdivisionErrorMetric()
{
  this->GenericCell = nullptr;
  this->DataSet = nullptr;
}

//-----------------------------------------------------------------------------
svtkGenericSubdivisionErrorMetric::~svtkGenericSubdivisionErrorMetric() = default;

//-----------------------------------------------------------------------------
// Avoid reference loop
void svtkGenericSubdivisionErrorMetric::SetGenericCell(svtkGenericAdaptorCell* c)
{
  this->GenericCell = c;
  this->Modified();
}

//-----------------------------------------------------------------------------
// Avoid reference loop
void svtkGenericSubdivisionErrorMetric::SetDataSet(svtkGenericDataSet* ds)
{
  this->DataSet = ds;
  this->Modified();
}

//-----------------------------------------------------------------------------
void svtkGenericSubdivisionErrorMetric::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "GenericCell: " << this->GenericCell << endl;
  os << indent << "DataSet: " << this->DataSet << endl;
}
