/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPath.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkPath.h"

#include "svtkGenericCell.h"
#include "svtkIdList.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkIntArray.h"
#include "svtkNew.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"

#include <cassert>

//----------------------------------------------------------------------------
svtkStandardNewMacro(svtkPath);

//----------------------------------------------------------------------------
svtkPath::svtkPath()
{
  svtkNew<svtkPoints> points;
  this->SetPoints(points);

  svtkNew<svtkIntArray> controlPointCodes;
  controlPointCodes->SetNumberOfComponents(1);
  this->PointData->SetScalars(controlPointCodes);
}

//----------------------------------------------------------------------------
svtkPath::~svtkPath() = default;

//----------------------------------------------------------------------------
void svtkPath::Allocate(svtkIdType size, int extSize)
{
  this->Points->Allocate(size, extSize);
  this->PointData->Allocate(size, extSize);
}

//----------------------------------------------------------------------------
void svtkPath::GetCell(svtkIdType, svtkGenericCell* cell)
{
  cell->SetCellTypeToEmptyCell();
}

//----------------------------------------------------------------------------
void svtkPath::GetCellPoints(svtkIdType, svtkIdList* ptIds)
{
  ptIds->Reset();
}

//----------------------------------------------------------------------------
void svtkPath::GetPointCells(svtkIdType, svtkIdList* cellIds)
{
  cellIds->Reset();
}

//----------------------------------------------------------------------------
void svtkPath::Reset()
{
  this->Points->Reset();
  this->PointData->Reset();
}

//----------------------------------------------------------------------------
svtkPath* svtkPath::GetData(svtkInformation* info)
{
  return info ? svtkPath::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkPath* svtkPath::GetData(svtkInformationVector* v, int i)
{
  return svtkPath::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
void svtkPath::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
void svtkPath::InsertNextPoint(float pts[], int code)
{
  this->Points->InsertNextPoint(pts);

  svtkIntArray* codes = svtkArrayDownCast<svtkIntArray>(this->PointData->GetScalars());
  assert("control point code array is int type" && codes);
  codes->InsertNextValue(code);
}

//----------------------------------------------------------------------------
void svtkPath::InsertNextPoint(double pts[], int code)
{
  this->InsertNextPoint(pts[0], pts[1], pts[2], code);
}

//----------------------------------------------------------------------------
void svtkPath::InsertNextPoint(double x, double y, double z, int code)
{
  this->Points->InsertNextPoint(x, y, z);

  svtkIntArray* codes = svtkArrayDownCast<svtkIntArray>(this->PointData->GetScalars());
  assert("control point code array is int type" && codes);
  codes->InsertNextValue(code);
}

//----------------------------------------------------------------------------
void svtkPath::SetCodes(svtkIntArray* codes)
{
  this->PointData->SetScalars(codes);
}

//----------------------------------------------------------------------------
svtkIntArray* svtkPath::GetCodes()
{
  return svtkArrayDownCast<svtkIntArray>(this->PointData->GetScalars());
}
