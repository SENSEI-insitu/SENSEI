/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPoints.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkPoints.h"

#include "svtkBitArray.h"
#include "svtkCharArray.h"
#include "svtkDoubleArray.h"
#include "svtkFloatArray.h"
#include "svtkIdList.h"
#include "svtkIdTypeArray.h"
#include "svtkIntArray.h"
#include "svtkLongArray.h"
#include "svtkObjectFactory.h"
#include "svtkShortArray.h"
#include "svtkUnsignedCharArray.h"
#include "svtkUnsignedIntArray.h"
#include "svtkUnsignedLongArray.h"
#include "svtkUnsignedShortArray.h"

//----------------------------------------------------------------------------
svtkPoints* svtkPoints::New(int dataType)
{
  // First try to create the object from the svtkObjectFactory
  svtkObject* ret = svtkObjectFactory::CreateInstance("svtkPoints");
  if (ret)
  {
    if (dataType != SVTK_FLOAT)
    {
      static_cast<svtkPoints*>(ret)->SetDataType(dataType);
    }
    return static_cast<svtkPoints*>(ret);
  }
  // If the factory was unable to create the object, then create it here.
  svtkPoints* result = new svtkPoints(dataType);
  result->InitializeObjectBase();
  return result;
}

svtkPoints* svtkPoints::New()
{
  return svtkPoints::New(SVTK_FLOAT);
}

// Construct object with an initial data array of type float.
svtkPoints::svtkPoints(int dataType)
{
  this->Data = svtkFloatArray::New();
  this->Data->Register(this);
  this->Data->Delete();
  this->SetDataType(dataType);

  this->Data->SetNumberOfComponents(3);
  this->Data->SetName("Points");

  this->Bounds[0] = this->Bounds[2] = this->Bounds[4] = SVTK_DOUBLE_MAX;
  this->Bounds[1] = this->Bounds[3] = this->Bounds[5] = -SVTK_DOUBLE_MAX;
}

svtkPoints::~svtkPoints()
{
  this->Data->UnRegister(this);
}

// Given a list of pt ids, return an array of points.
void svtkPoints::GetPoints(svtkIdList* ptIds, svtkPoints* outPoints)
{
  outPoints->Data->SetNumberOfTuples(ptIds->GetNumberOfIds());
  this->Data->GetTuples(ptIds, outPoints->Data);
}

// Determine (xmin,xmax, ymin,ymax, zmin,zmax) bounds of points.
void svtkPoints::ComputeBounds()
{
  if (this->GetMTime() > this->ComputeTime)
  {
    this->Data->ComputeScalarRange(this->Bounds);
    this->ComputeTime.Modified();
  }
}

// Return the bounds of the points.
double* svtkPoints::GetBounds()
{
  this->ComputeBounds();
  return this->Bounds;
}

// Return the bounds of the points.
void svtkPoints::GetBounds(double bounds[6])
{
  this->ComputeBounds();
  memcpy(bounds, this->Bounds, 6 * sizeof(double));
}

svtkMTimeType svtkPoints::GetMTime()
{
  svtkMTimeType doTime = this->Superclass::GetMTime();
  if (this->Data->GetMTime() > doTime)
  {
    doTime = this->Data->GetMTime();
  }
  return doTime;
}

svtkTypeBool svtkPoints::Allocate(svtkIdType sz, svtkIdType ext)
{
  int numComp = this->Data->GetNumberOfComponents();
  return this->Data->Allocate(sz * numComp, ext * numComp);
}

void svtkPoints::Initialize()
{
  this->Data->Initialize();
  this->Modified();
}

void svtkPoints::Modified()
{
  this->Superclass::Modified();
  if (this->Data)
  {
    this->Data->Modified();
  }
}

int svtkPoints::GetDataType() const
{
  return this->Data->GetDataType();
}

// Specify the underlying data type of the object.
void svtkPoints::SetDataType(int dataType)
{
  if (dataType == this->Data->GetDataType())
  {
    return;
  }

  this->Data->Delete();
  this->Data = svtkDataArray::CreateDataArray(dataType);
  this->Data->SetNumberOfComponents(3);
  this->Data->SetName("Points");
  this->Modified();
}

// Set the data for this object. The tuple dimension must be consistent with
// the object.
void svtkPoints::SetData(svtkDataArray* data)
{
  if (data != this->Data && data != nullptr)
  {
    if (data->GetNumberOfComponents() != this->Data->GetNumberOfComponents())
    {
      svtkErrorMacro(<< "Number of components is different...can't set data");
      return;
    }
    this->Data->UnRegister(this);
    this->Data = data;
    this->Data->Register(this);
    if (!this->Data->GetName())
    {
      this->Data->SetName("Points");
    }
    this->Modified();
  }
}

// Deep copy of data. Checks consistency to make sure this operation
// makes sense.
void svtkPoints::DeepCopy(svtkPoints* da)
{
  if (da == nullptr)
  {
    return;
  }
  if (da->Data != this->Data && da->Data != nullptr)
  {
    if (da->Data->GetNumberOfComponents() != this->Data->GetNumberOfComponents())
    {
      svtkErrorMacro(<< "Number of components is different...can't copy");
      return;
    }
    this->Data->DeepCopy(da->Data);
    this->Modified();
  }
}

// Shallow copy of data (i.e. via reference counting). Checks
// consistency to make sure this operation makes sense.
void svtkPoints::ShallowCopy(svtkPoints* da)
{
  this->SetData(da->GetData());
}

unsigned long svtkPoints::GetActualMemorySize()
{
  return this->Data->GetActualMemorySize();
}

void svtkPoints::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Data: " << this->Data << "\n";
  os << indent << "Data Array Name: ";
  if (this->Data->GetName())
  {
    os << this->Data->GetName() << "\n";
  }
  else
  {
    os << "(none)\n";
  }

  os << indent << "Number Of Points: " << this->GetNumberOfPoints() << "\n";
  const double* bounds = this->GetBounds();
  os << indent << "Bounds: \n";
  os << indent << "  Xmin,Xmax: (" << bounds[0] << ", " << bounds[1] << ")\n";
  os << indent << "  Ymin,Ymax: (" << bounds[2] << ", " << bounds[3] << ")\n";
  os << indent << "  Zmin,Zmax: (" << bounds[4] << ", " << bounds[5] << ")\n";
}
