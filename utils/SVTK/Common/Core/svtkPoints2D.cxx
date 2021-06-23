/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPoints2D.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkPoints2D.h"

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
svtkPoints2D* svtkPoints2D::New(int dataType)
{
  // First try to create the object from the svtkObjectFactory
  svtkObject* ret = svtkObjectFactory::CreateInstance("svtkPoints2D");
  if (ret)
  {
    if (dataType != SVTK_FLOAT)
    {
      static_cast<svtkPoints2D*>(ret)->SetDataType(dataType);
    }
    return static_cast<svtkPoints2D*>(ret);
  }
  // If the factory was unable to create the object, then create it here.
  svtkPoints2D* result = new svtkPoints2D(dataType);
  result->InitializeObjectBase();
  return result;
}

svtkPoints2D* svtkPoints2D::New()
{
  return svtkPoints2D::New(SVTK_FLOAT);
}

// Construct object with an initial data array of type float.
svtkPoints2D::svtkPoints2D(int dataType)
{
  this->Data = svtkFloatArray::New();
  this->Data->Register(this);
  this->Data->Delete();
  this->SetDataType(dataType);

  this->Data->SetNumberOfComponents(2);
  this->Data->SetName("Points2D");

  this->Bounds[0] = this->Bounds[2] = SVTK_DOUBLE_MAX;
  this->Bounds[1] = this->Bounds[3] = -SVTK_DOUBLE_MAX;
}

svtkPoints2D::~svtkPoints2D()
{
  this->Data->UnRegister(this);
}

// Given a list of pt ids, return an array of points.
void svtkPoints2D::GetPoints(svtkIdList* ptIds, svtkPoints2D* fp)
{
  svtkIdType num = ptIds->GetNumberOfIds();
  for (svtkIdType i = 0; i < num; i++)
  {
    fp->InsertPoint(i, this->GetPoint(ptIds->GetId(i)));
  }
}

// Determine (xmin,xmax, ymin,ymax, zmin,zmax) bounds of points.
void svtkPoints2D::ComputeBounds()
{
  if (this->GetMTime() > this->ComputeTime)
  {
    this->Bounds[0] = this->Bounds[2] = SVTK_DOUBLE_MAX;
    this->Bounds[1] = this->Bounds[3] = -SVTK_DOUBLE_MAX;
    for (svtkIdType i = 0; i < this->GetNumberOfPoints(); ++i)
    {
      double x[2];
      this->GetPoint(i, x);
      for (int j = 0; j < 2; ++j)
      {
        if (x[j] < this->Bounds[2 * j])
        {
          this->Bounds[2 * j] = x[j];
        }
        if (x[j] > this->Bounds[2 * j + 1])
        {
          this->Bounds[2 * j + 1] = x[j];
        }
      }
    }

    this->ComputeTime.Modified();
  }
}

// Return the bounds of the points.
double* svtkPoints2D::GetBounds()
{
  this->ComputeBounds();
  return this->Bounds;
}

// Return the bounds of the points.
void svtkPoints2D::GetBounds(double bounds[4])
{
  this->ComputeBounds();
  memcpy(bounds, this->Bounds, 4 * sizeof(double));
}

svtkTypeBool svtkPoints2D::Allocate(svtkIdType sz, svtkIdType ext)
{
  int numComp = this->Data->GetNumberOfComponents();
  return this->Data->Allocate(sz * numComp, ext * numComp);
}

void svtkPoints2D::Initialize()
{
  this->Data->Initialize();
  this->Modified();
}

int svtkPoints2D::GetDataType() const
{
  return this->Data->GetDataType();
}

// Specify the underlying data type of the object.
void svtkPoints2D::SetDataType(int dataType)
{
  if (dataType == this->Data->GetDataType())
  {
    return;
  }

  this->Data->Delete();
  this->Data = svtkDataArray::CreateDataArray(dataType);
  this->Data->SetNumberOfComponents(2);
  this->Data->SetName("Points2D");
  this->Modified();
}

// Set the data for this object. The tuple dimension must be consistent with
// the object.
void svtkPoints2D::SetData(svtkDataArray* data)
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
      this->Data->SetName("Points2D");
    }
    this->Modified();
  }
}

// Deep copy of data. Checks consistency to make sure this operation
// makes sense.
void svtkPoints2D::DeepCopy(svtkPoints2D* da)
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
void svtkPoints2D::ShallowCopy(svtkPoints2D* da)
{
  this->SetData(da->GetData());
}

unsigned long svtkPoints2D::GetActualMemorySize()
{
  return this->Data->GetActualMemorySize();
}

void svtkPoints2D::PrintSelf(ostream& os, svtkIndent indent)
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
}
