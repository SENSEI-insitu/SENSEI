/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPointSet.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkPointSet.h"

#include "svtkCell.h"
#include "svtkCellLocator.h"
#include "svtkClosestPointStrategy.h"
#include "svtkGarbageCollector.h"
#include "svtkGenericCell.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkPointLocator.h"
#include "svtkPointSetCellIterator.h"
#include "svtkStaticCellLocator.h"
#include "svtkStaticPointLocator.h"

#include "svtkSmartPointer.h"

#define SVTK_CREATE(type, name) svtkSmartPointer<type> name = svtkSmartPointer<type>::New()

svtkCxxSetObjectMacro(svtkPointSet, Points, svtkPoints);
svtkCxxSetObjectMacro(svtkPointSet, PointLocator, svtkAbstractPointLocator);
svtkCxxSetObjectMacro(svtkPointSet, CellLocator, svtkAbstractCellLocator);

//----------------------------------------------------------------------------
svtkPointSet::svtkPointSet()
{
  this->Editable = false;
  this->Points = nullptr;
  this->PointLocator = nullptr;
  this->CellLocator = nullptr;
}

//----------------------------------------------------------------------------
svtkPointSet::~svtkPointSet()
{
  this->Cleanup();

  if (this->PointLocator != nullptr)
  {
    cout << "DELETING LOCATOR: PointSet: " << this << " locator: " << this->PointLocator << "\n";
  }
  this->SetPointLocator(nullptr);
  this->SetCellLocator(nullptr);
}

//----------------------------------------------------------------------------
// Copy the geometric structure of an input point set object.
void svtkPointSet::CopyStructure(svtkDataSet* ds)
{
  svtkPointSet* ps = static_cast<svtkPointSet*>(ds);

  if (this->Points != ps->Points)
  {
    if (this->PointLocator)
    {
      this->PointLocator->Initialize();
    }
    this->SetPoints(ps->Points);

    if (this->CellLocator)
    {
      this->CellLocator->Initialize();
    }
  }
}

//----------------------------------------------------------------------------
void svtkPointSet::Cleanup()
{
  if (this->Points)
  {
    this->Points->UnRegister(this);
    this->Points = nullptr;
  }
}

//----------------------------------------------------------------------------
void svtkPointSet::Initialize()
{
  svtkDataSet::Initialize();

  this->Cleanup();

  if (this->PointLocator)
  {
    this->PointLocator->Initialize();
  }
  if (this->CellLocator)
  {
    this->CellLocator->Initialize();
  }
}

//----------------------------------------------------------------------------
void svtkPointSet::ComputeBounds()
{
  if (this->Points)
  {
    // only depends on tyhis->Points so only check this->Points mtime
    // The generic mtime check includes Field/Cell/PointData also
    // which has no impact on the bounds
    if (this->Points->GetMTime() >= this->ComputeTime)
    {
      const double* bounds = this->Points->GetBounds();
      for (int i = 0; i < 6; i++)
      {
        this->Bounds[i] = bounds[i];
      }
      this->ComputeTime.Modified();
    }
  }
}

//----------------------------------------------------------------------------
svtkMTimeType svtkPointSet::GetMTime()
{
  svtkMTimeType dsTime = svtkDataSet::GetMTime();

  if (this->Points)
  {
    if (this->Points->GetMTime() > dsTime)
    {
      dsTime = this->Points->GetMTime();
    }
  }

  // don't get locator's mtime because its an internal object that cannot be
  // modified directly from outside. Causes problems due to FindCell()
  // SetPoints() method.

  return dsTime;
}

//----------------------------------------------------------------------------
void svtkPointSet::BuildPointLocator()
{
  if (!this->Points)
  {
    return;
  }

  if (!this->PointLocator)
  {
    if (this->Editable || !this->Points->GetData()->HasStandardMemoryLayout())
    {
      this->PointLocator = svtkPointLocator::New();
    }
    else
    {
      this->PointLocator = svtkStaticPointLocator::New();
    }
    this->PointLocator->SetDataSet(this);
  }
  else if (this->Points->GetMTime() > this->PointLocator->GetMTime())
  {
    cout << "Building supplied point locator\n";
    this->PointLocator->SetDataSet(this);
  }

  this->PointLocator->BuildLocator();
}

//----------------------------------------------------------------------------
// Build the cell locator (if needed)
void svtkPointSet::BuildCellLocator()
{
  if (!this->Points)
  {
    return;
  }

  if (!this->CellLocator)
  {
    if (this->Editable || !this->Points->GetData()->HasStandardMemoryLayout())
    {
      this->CellLocator = svtkCellLocator::New();
    }
    else
    {
      this->CellLocator = svtkStaticCellLocator::New();
    }
    this->CellLocator->Register(this);
    this->CellLocator->Delete();
    this->CellLocator->SetDataSet(this);
  }
  else if (this->Points->GetMTime() > this->CellLocator->GetMTime())
  {
    this->CellLocator->SetDataSet(this);
  }

  this->CellLocator->BuildLocator();
}

//----------------------------------------------------------------------------
svtkIdType svtkPointSet::FindPoint(double x[3])
{
  if (!this->Points)
  {
    return -1;
  }

  if (!this->PointLocator)
  {
    this->BuildPointLocator();
  }

  return this->PointLocator->FindClosestPoint(x);
}

//----------------------------------------------------------------------------
// This FindCell() method is based on using a locator (either point or
// cell). In this application, point locators are typically faster to build
// and operate on than cell locator, yet do not always produce the correct
// result. The basic idea is that we find one or more close points to the
// query point, and we assume that one of the cells attached to one of the
// close points contains the query point. However this approach is not 100%
// reliable, in which case a slower cell locator must be used. The algorithm
// below (based on a point locator) uses progressively more complex (and
// expensive) approaches to identify close points near the query point (and
// connected cells). If a point locator approach proves unreliable, then a
// cell locator strategy should be used. Use subclasses of
// svtkFindCellStrategy to control the strategies.
svtkIdType svtkPointSet::FindCell(double x[3], svtkCell* cell, svtkGenericCell* gencell,
  svtkIdType cellId, double tol2, int& subId, double pcoords[3], double* weights)
{
  SVTK_CREATE(svtkClosestPointStrategy, strategy);
  strategy->Initialize(this);
  return strategy->FindCell(x, cell, gencell, cellId, tol2, subId, pcoords, weights);
}

//----------------------------------------------------------------------------
svtkIdType svtkPointSet::FindCell(double x[3], svtkCell* cell, svtkIdType cellId, double tol2,
  int& subId, double pcoords[3], double* weights)
{
  return this->FindCell(x, cell, nullptr, cellId, tol2, subId, pcoords, weights);
}

//----------------------------------------------------------------------------
svtkCellIterator* svtkPointSet::NewCellIterator()
{
  svtkPointSetCellIterator* iter = svtkPointSetCellIterator::New();
  iter->SetPointSet(this);
  return iter;
}

//----------------------------------------------------------------------------
void svtkPointSet::Squeeze()
{
  if (this->Points)
  {
    this->Points->Squeeze();
  }
  svtkDataSet::Squeeze();
}

//----------------------------------------------------------------------------
void svtkPointSet::ReportReferences(svtkGarbageCollector* collector)
{
  this->Superclass::ReportReferences(collector);
  svtkGarbageCollectorReport(collector, this->PointLocator, "PointLocator");
  svtkGarbageCollectorReport(collector, this->CellLocator, "CellLocator");
}

//----------------------------------------------------------------------------
unsigned long svtkPointSet::GetActualMemorySize()
{
  unsigned long size = this->svtkDataSet::GetActualMemorySize();
  if (this->Points)
  {
    size += this->Points->GetActualMemorySize();
  }
  return size;
}

//----------------------------------------------------------------------------
void svtkPointSet::ShallowCopy(svtkDataObject* dataObject)
{
  svtkPointSet* pointSet = svtkPointSet::SafeDownCast(dataObject);

  if (pointSet != nullptr)
  {
    this->SetEditable(pointSet->GetEditable());
    this->SetPoints(pointSet->GetPoints());
  }

  // Do superclass
  this->svtkDataSet::ShallowCopy(dataObject);
}

//----------------------------------------------------------------------------
void svtkPointSet::DeepCopy(svtkDataObject* dataObject)
{
  svtkPointSet* pointSet = svtkPointSet::SafeDownCast(dataObject);

  if (pointSet != nullptr)
  {
    this->SetEditable(pointSet->GetEditable());
    svtkPoints* newPoints;
    svtkPoints* pointsToCopy = pointSet->GetPoints();
    if (pointsToCopy)
    {
      newPoints = pointsToCopy->NewInstance();
      newPoints->SetDataType(pointsToCopy->GetDataType());
      newPoints->DeepCopy(pointsToCopy);
    }
    else
    {
      newPoints = svtkPoints::New();
    }
    this->SetPoints(newPoints);
    newPoints->Delete();
  }

  // Do superclass
  this->svtkDataSet::DeepCopy(dataObject);
}

//----------------------------------------------------------------------------
svtkPointSet* svtkPointSet::GetData(svtkInformation* info)
{
  return info ? svtkPointSet::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkPointSet* svtkPointSet::GetData(svtkInformationVector* v, int i)
{
  return svtkPointSet::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
void svtkPointSet::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Editable: " << (this->Editable ? "true\n" : "false\n");
  os << indent << "Number Of Points: " << this->GetNumberOfPoints() << "\n";
  os << indent << "Point Coordinates: " << this->Points << "\n";
  os << indent << "PointLocator: " << this->PointLocator << "\n";
  os << indent << "CellLocator: " << this->CellLocator << "\n";
}

//----------------------------------------------------------------------------
void svtkPointSet::Register(svtkObjectBase* o)
{
  this->RegisterInternal(o, 1);
}

//----------------------------------------------------------------------------
void svtkPointSet::UnRegister(svtkObjectBase* o)
{
  this->UnRegisterInternal(o, 1);
}
