/*=========================================================================

  Program:   Visualization Toolkit
  Module:    otherEmptyCell.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkCellArray.h"
#include "svtkCellData.h"
#include "svtkEmptyCell.h"
#include "svtkFloatArray.h"
#include "svtkIdList.h"
#include "svtkPointData.h"
#include "svtkPointLocator.h"
#include "svtkPoints.h"

#include <sstream>

#include "svtkDebugLeaks.h"

void TestOEC(ostream& strm)
{
  svtkEmptyCell* cell = svtkEmptyCell::New();
  svtkCell* cell2 = cell->NewInstance();
  cell2->DeepCopy(cell);
  svtkIdList* ids = svtkIdList::New();
  svtkPoints* pts = svtkPoints::New();
  double v = 0.0;
  svtkFloatArray* cellScalars = svtkFloatArray::New();
  svtkPointLocator* locator = svtkPointLocator::New();
  svtkCellArray* verts = svtkCellArray::New();
  svtkCellArray* lines = svtkCellArray::New();
  svtkCellArray* polys = svtkCellArray::New();
  svtkPointData* inPd = svtkPointData::New();
  svtkPointData* outPd = svtkPointData::New();
  svtkCellData* inCd = svtkCellData::New();
  svtkCellData* outCd = svtkCellData::New();
  int cellId = 0;
  int inOut = 0;
  double t, tol = 0.0;
  double x[3];
  double c[3];
  double p[3];
  double d;
  double w[3];
  int s;

  strm << "Testing EmptyCell" << endl;
  strm << "Cell Type is: " << cell2->GetCellType() << endl;
  strm << "Cell Dimension is: " << cell2->GetCellDimension() << endl;
  strm << "Cell NumberOfEdges is: " << cell2->GetNumberOfEdges() << endl;
  strm << "Cell NumberOfFaces is: " << cell2->GetNumberOfFaces() << endl;
  strm << "Cell GetEdge(0) is: " << cell2->GetEdge(0) << endl;
  strm << "Cell GetFace(0) is: " << cell2->GetFace(0) << endl;
  strm << "Cell CellBoundary(0,p,ids) is: " << cell2->CellBoundary(0, p, ids) << endl;
  strm << "Cell EvaluatePosition(x, c, s, p, d, w)" << endl;
  cell2->EvaluatePosition(x, c, s, p, d, w);
  strm << "Cell EvaluateLocation(s, p, x, w)" << endl;
  cell2->EvaluateLocation(s, p, x, w);
  strm << "Cell Contour" << endl;
  cell2->Contour(v, cellScalars, locator, verts, lines, polys, inPd, outPd, inCd, cellId, outCd);
  strm << "Cell Clip" << endl;
  cell2->Clip(v, cellScalars, locator, verts, inPd, outPd, inCd, cellId, outCd, inOut);
  strm << "Cell IntersectWithLine" << endl;
  cell2->IntersectWithLine(x, x, tol, t, x, p, s);
  strm << "Cell Triangulate" << endl;
  cell2->Triangulate(s, ids, pts);
  strm << "Cell Derivatives" << endl;
  cell2->Derivatives(s, p, x, inOut, w);

  // clean up
  cell->Delete();
  cell2->Delete();
  ids->Delete();
  pts->Delete();
  cellScalars->Delete();
  locator->Delete();
  verts->Delete();
  lines->Delete();
  polys->Delete();
  inPd->Delete();
  outPd->Delete();
  inCd->Delete();
  outCd->Delete();
  strm << "Testing EmptyCell Complete" << endl;
}

int otherEmptyCell(int, char*[])
{
  std::ostringstream svtkmsg_with_warning_C4701;
  TestOEC(svtkmsg_with_warning_C4701);

  return 0;
}
