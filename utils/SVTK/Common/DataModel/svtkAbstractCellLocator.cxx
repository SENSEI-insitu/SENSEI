/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAbstractCellLocator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkAbstractCellLocator.h"

#include "svtkCellArray.h"
#include "svtkDataSet.h"
#include "svtkGenericCell.h"
#include "svtkIdList.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPoints.h"
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
svtkAbstractCellLocator::svtkAbstractCellLocator()
{
  this->CacheCellBounds = 0;
  this->CellBounds = nullptr;
  this->MaxLevel = 8;
  this->Level = 0;
  this->RetainCellLists = 1;
  this->NumberOfCellsPerNode = 32;
  this->UseExistingSearchStructure = 0;
  this->LazyEvaluation = 0;
  this->GenericCell = svtkGenericCell::New();
}
//----------------------------------------------------------------------------
svtkAbstractCellLocator::~svtkAbstractCellLocator()
{
  this->GenericCell->Delete();
}
//----------------------------------------------------------------------------
bool svtkAbstractCellLocator::StoreCellBounds()
{
  if (this->CellBounds)
    return false;
  if (!this->DataSet)
    return false;
  // Allocate space for cell bounds storage, then fill
  svtkIdType numCells = this->DataSet->GetNumberOfCells();
  this->CellBounds = new double[numCells][6];
  for (svtkIdType j = 0; j < numCells; j++)
  {
    this->DataSet->GetCellBounds(j, CellBounds[j]);
  }
  return true;
}
//----------------------------------------------------------------------------
void svtkAbstractCellLocator::FreeCellBounds()
{
  delete[] this->CellBounds;
  this->CellBounds = nullptr;
}
//----------------------------------------------------------------------------
int svtkAbstractCellLocator::IntersectWithLine(const double p1[3], const double p2[3], double tol,
  double& t, double x[3], double pcoords[3], int& subId)
{
  svtkIdType cellId = -1;
  return this->IntersectWithLine(p1, p2, tol, t, x, pcoords, subId, cellId);
}
//----------------------------------------------------------------------------
int svtkAbstractCellLocator::IntersectWithLine(const double p1[3], const double p2[3], double tol,
  double& t, double x[3], double pcoords[3], int& subId, svtkIdType& cellId)
{
  int returnVal;
  returnVal = this->IntersectWithLine(p1, p2, tol, t, x, pcoords, subId, cellId, this->GenericCell);
  return returnVal;
}
//----------------------------------------------------------------------------
int svtkAbstractCellLocator::IntersectWithLine(const double svtkNotUsed(p1)[3],
  const double svtkNotUsed(p2)[3], double svtkNotUsed(tol), double& svtkNotUsed(t),
  double svtkNotUsed(x)[3], double svtkNotUsed(pcoords)[3], int& svtkNotUsed(subId),
  svtkIdType& svtkNotUsed(cellId), svtkGenericCell* svtkNotUsed(cell))
{
  svtkErrorMacro(<< "The locator class - " << this->GetClassName()
                << " does not yet support IntersectWithLine");
  return 0;
}
//----------------------------------------------------------------------------
int svtkAbstractCellLocator::IntersectWithLine(const double svtkNotUsed(p1)[3],
  const double svtkNotUsed(p2)[3], svtkPoints* svtkNotUsed(points), svtkIdList* svtkNotUsed(cellIds))
{
  svtkErrorMacro(<< "The locator class - " << this->GetClassName()
                << " does not yet support this IntersectWithLine interface");
  return 0;
}
//----------------------------------------------------------------------------
void svtkAbstractCellLocator::FindClosestPoint(
  const double x[3], double closestPoint[3], svtkIdType& cellId, int& subId, double& dist2)
{
  this->FindClosestPoint(x, closestPoint, this->GenericCell, cellId, subId, dist2);
}
//----------------------------------------------------------------------------
void svtkAbstractCellLocator::FindClosestPoint(const double svtkNotUsed(x)[3],
  double svtkNotUsed(closestPoint)[3], svtkGenericCell* svtkNotUsed(cell),
  svtkIdType& svtkNotUsed(cellId), int& svtkNotUsed(subId), double& svtkNotUsed(dist2))
{
  svtkErrorMacro(<< "The locator class - " << this->GetClassName()
                << " does not yet support FindClosestPoint");
}
//----------------------------------------------------------------------------
svtkIdType svtkAbstractCellLocator::FindClosestPointWithinRadius(double x[3], double radius,
  double closestPoint[3], svtkGenericCell* cell, svtkIdType& cellId, int& subId, double& dist2)
{
  int inside;
  return this->FindClosestPointWithinRadius(
    x, radius, closestPoint, cell, cellId, subId, dist2, inside);
}
//----------------------------------------------------------------------------
svtkIdType svtkAbstractCellLocator::FindClosestPointWithinRadius(
  double x[3], double radius, double closestPoint[3], svtkIdType& cellId, int& subId, double& dist2)
{
  int inside;
  return this->FindClosestPointWithinRadius(
    x, radius, closestPoint, this->GenericCell, cellId, subId, dist2, inside);
}
//----------------------------------------------------------------------------
svtkIdType svtkAbstractCellLocator::FindClosestPointWithinRadius(double svtkNotUsed(x)[3],
  double svtkNotUsed(radius), double svtkNotUsed(closestPoint)[3], svtkGenericCell* svtkNotUsed(cell),
  svtkIdType& svtkNotUsed(cellId), int& svtkNotUsed(subId), double& svtkNotUsed(dist2),
  int& svtkNotUsed(inside))
{
  svtkErrorMacro(<< "The locator class - " << this->GetClassName()
                << " does not yet support FindClosestPoint");
  return 0;
}
//----------------------------------------------------------------------------
void svtkAbstractCellLocator::FindCellsWithinBounds(
  double* svtkNotUsed(bbox), svtkIdList* svtkNotUsed(cells))
{
  svtkErrorMacro(<< "The locator class - " << this->GetClassName()
                << " does not yet support FindCellsWithinBounds");
}
//----------------------------------------------------------------------------
void svtkAbstractCellLocator::FindCellsAlongLine(const double svtkNotUsed(p1)[3],
  const double svtkNotUsed(p2)[3], double svtkNotUsed(tolerance), svtkIdList* svtkNotUsed(cells))
{
  svtkErrorMacro(<< "The locator " << this->GetClassName()
                << " does not yet support FindCellsAlongLine");
}
//---------------------------------------------------------------------------
svtkIdType svtkAbstractCellLocator::FindCell(double x[3])
{
  //
  double dist2 = 0, pcoords[3], weights[32];
  return this->FindCell(x, dist2, this->GenericCell, pcoords, weights);
}
//----------------------------------------------------------------------------
svtkIdType svtkAbstractCellLocator::FindCell(
  double x[3], double tol2, svtkGenericCell* GenCell, double pcoords[3], double* weights)
{
  svtkIdType returnVal = -1;
  int subId;
  //
  static bool warning_shown = false;
  if (!warning_shown)
  {
    svtkWarningMacro(<< this->GetClassName() << " Does not implement FindCell"
                    << " Reverting to slow DataSet implementation");
    warning_shown = true;
  }
  //
  if (this->DataSet)
  {
    returnVal = this->DataSet->FindCell(x, nullptr, GenCell, 0, tol2, subId, pcoords, weights);
  }
  return returnVal;
}
//----------------------------------------------------------------------------
bool svtkAbstractCellLocator::InsideCellBounds(double x[3], svtkIdType cell_ID)
{
  double cellBounds[6], delta[3] = { 0.0, 0.0, 0.0 };
  if (this->DataSet)
  {
    this->DataSet->GetCellBounds(cell_ID, cellBounds);
    return svtkMath::PointIsWithinBounds(x, cellBounds, delta) != 0;
  }
  return 0;
}
//----------------------------------------------------------------------------
void svtkAbstractCellLocator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Cache Cell Bounds: " << this->CacheCellBounds << "\n";
  os << indent << "Retain Cell Lists: " << (this->RetainCellLists ? "On\n" : "Off\n");
  os << indent << "Number of Cells Per Bucket: " << this->NumberOfCellsPerNode << "\n";
  os << indent << "UseExistingSearchStructure: " << this->UseExistingSearchStructure << "\n";
  os << indent << "LazyEvaluation: " << this->LazyEvaluation << "\n";
}
//----------------------------------------------------------------------------
