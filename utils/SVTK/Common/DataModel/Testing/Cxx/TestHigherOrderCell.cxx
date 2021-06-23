/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestHigherOrderCell.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkGenericCell.h"
#include "svtkPoints.h"

static const unsigned int depth = 5;
static unsigned char HigherOrderCell[][depth] = {
  { SVTK_LINE, SVTK_QUADRATIC_EDGE, SVTK_NUMBER_OF_CELL_TYPES, SVTK_NUMBER_OF_CELL_TYPES,
    SVTK_NUMBER_OF_CELL_TYPES },
  { SVTK_TRIANGLE, SVTK_QUADRATIC_TRIANGLE, SVTK_BIQUADRATIC_TRIANGLE, SVTK_NUMBER_OF_CELL_TYPES,
    SVTK_NUMBER_OF_CELL_TYPES },
  { SVTK_QUAD, SVTK_QUADRATIC_QUAD, SVTK_QUADRATIC_LINEAR_QUAD, SVTK_BIQUADRATIC_QUAD,
    SVTK_NUMBER_OF_CELL_TYPES },
  { SVTK_TETRA, SVTK_QUADRATIC_TETRA, SVTK_NUMBER_OF_CELL_TYPES, SVTK_NUMBER_OF_CELL_TYPES,
    SVTK_NUMBER_OF_CELL_TYPES },
  { SVTK_HEXAHEDRON, SVTK_QUADRATIC_HEXAHEDRON, SVTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON,
    SVTK_TRIQUADRATIC_HEXAHEDRON, SVTK_NUMBER_OF_CELL_TYPES },
  { SVTK_WEDGE, SVTK_QUADRATIC_WEDGE, SVTK_QUADRATIC_LINEAR_WEDGE, SVTK_BIQUADRATIC_QUADRATIC_WEDGE,
    SVTK_NUMBER_OF_CELL_TYPES },
  { SVTK_PYRAMID, SVTK_QUADRATIC_PYRAMID, SVTK_NUMBER_OF_CELL_TYPES, SVTK_NUMBER_OF_CELL_TYPES,
    SVTK_NUMBER_OF_CELL_TYPES }
};

//----------------------------------------------------------------------------
// Simply set the points to the pcoords coordinate
// and the point id to the natural order
void InitializeACell(svtkCell* cell)
{
  if (cell)
  {
    double* pcoords = cell->GetParametricCoords();
    int numPts = cell->GetNumberOfPoints();
    for (int i = 0; i < numPts; ++i)
    {
      double* point = pcoords + 3 * i;
      cell->GetPointIds()->SetId(i, i);
      // cerr << point[0] << "," << point[1] << "," << point[2] << endl;
      cell->GetPoints()->SetPoint(i, point);
    }
  }
}

//----------------------------------------------------------------------------
// c1 is the reference cell. In the test this is the linear cell
// and thus c2 is the higher order one. We need to check that result on c1
// are consistent with result on c2 (but we cannot say anything after that)
int CompareHigherOrderCell(svtkCell* c1, svtkCell* c2)
{
  int rval = 0;
  // c1->Print( cout );
  // c2->Print( cout );
  int c1numPts = c1->GetNumberOfPoints();
  int c2numPts = c2->GetNumberOfPoints();
  int numPts = c1numPts < c2numPts ? c1numPts : c2numPts;
  for (int p = 0; p < numPts; ++p)
  {
    svtkIdType pid1 = c1->GetPointId(p);
    svtkIdType pid2 = c2->GetPointId(p);
    if (pid1 != pid2)
    {
      cerr << "Problem with pid:" << pid1 << " != " << pid2 << " in cell #" << c1->GetCellType()
           << " and #" << c2->GetCellType() << endl;
      ++rval;
    }
    double* pt1 = c1->Points->GetPoint(p);
    double* pt2 = c2->Points->GetPoint(p);
    if (pt1[0] != pt2[0] || pt1[1] != pt2[1] || pt1[2] != pt2[2])
    {
      cerr << "Problem with points coord:" << pt1[0] << "," << pt1[1] << "," << pt1[2]
           << " != " << pt2[0] << "," << pt2[1] << "," << pt2[2] << " in cell #"
           << c1->GetCellType() << " and #" << c2->GetCellType() << endl;
      ++rval;
    }
  }
  return rval;
}

//----------------------------------------------------------------------------
int TestHigherOrderCell(int, char*[])
{
  int rval = 0;
  if (sizeof(HigherOrderCell[0]) != depth)
  {
    cerr << sizeof(HigherOrderCell[0]) << endl;
    cerr << "Problem in the test" << endl;
    return 1;
  }

  const unsigned char* orderCell;
  const unsigned int nCells = sizeof(HigherOrderCell) / depth;
  svtkCell* cellArray[depth];
  for (unsigned int i = 0; i < nCells; ++i)
  {
    orderCell = HigherOrderCell[i];
    // cerr << "Higher : " << (int)orderCell[0] << "," << (int)orderCell[1]
    // << "," << (int)orderCell[2] << "," << (int)orderCell[3] << ","
    // << (int)orderCell[4] << endl;
    for (unsigned int c = 0; c < depth; ++c)
    {
      const int cellType = orderCell[c];
      cellArray[c] = svtkGenericCell::InstantiateCell(cellType);
      InitializeACell(cellArray[c]);
    }
    svtkCell* linCell = cellArray[0];  // this is the reference linear cell
    svtkCell* quadCell = cellArray[1]; // this is the reference quadratic cell (serendipity)
    // const int numPts   = linCell->GetNumberOfPoints();
    const int numEdges = linCell->GetNumberOfEdges();
    const int numFaces = linCell->GetNumberOfFaces();
    const int dim = linCell->GetCellDimension();
    // First check consistency across cell of higher dimension:
    // Technically doing the loop from 1 to depth will be redundant when doing the
    // CompareHigherOrderCell on the quadratic cell since we will compare the exactly
    // same cell...
    for (unsigned int c = 1; c < depth; ++c)
    {
      svtkCell* cell = cellArray[c];
      if (cell)
      {
        if (cell->GetCellType() != (int)orderCell[c])
        {
          cerr << "Problem in the test" << endl;
          ++rval;
        }
        if (cell->GetCellDimension() != dim)
        {
          cerr << "Wrong dim for cellId #" << cell->GetCellType() << endl;
          ++rval;
        }
        if (cell->GetNumberOfEdges() != numEdges)
        {
          cerr << "Wrong numEdges for cellId #" << cell->GetCellType() << endl;
          ++rval;
        }
        if (cell->GetNumberOfFaces() != numFaces)
        {
          cerr << "Wrong numFace for cellId #" << cell->GetCellType() << endl;
          ++rval;
        }
        // Make sure that edge across all different cell are identical
        for (int e = 0; e < numEdges; ++e)
        {
          svtkCell* c1 = linCell->GetEdge(e);
          svtkCell* c2 = cell->GetEdge(e);
          cerr << "Doing Edge: #" << e << " comp:" << linCell->GetCellType() << " vs "
               << cell->GetCellType() << endl;
          rval += CompareHigherOrderCell(c1, c2);
          svtkCell* qc1 = quadCell->GetEdge(e);
          cerr << "Doing Edge: #" << e << " comp:" << quadCell->GetCellType() << " vs "
               << cell->GetCellType() << endl;
          if (cell->GetCellType() != SVTK_QUADRATIC_LINEAR_QUAD &&
            cell->GetCellType() != SVTK_QUADRATIC_LINEAR_WEDGE)
          {
            rval += CompareHigherOrderCell(qc1, c2);
          }
        }
        // Make sure that face across all different cell are identical
        for (int f = 0; f < numFaces; ++f)
        {
          svtkCell* f1 = linCell->GetFace(f);
          svtkCell* f2 = cell->GetFace(f);
          cerr << "Doing Face: #" << f << " comp:" << linCell->GetCellType() << " vs "
               << cell->GetCellType() << endl;
          if (cell->GetCellType() != SVTK_QUADRATIC_LINEAR_WEDGE)
          {
            rval += CompareHigherOrderCell(f1, f2);
          }
          svtkCell* qf1 = quadCell->GetFace(f);
          cerr << "Doing Face: #" << f << " comp:" << quadCell->GetCellType() << " vs "
               << cell->GetCellType() << endl;
          if (cell->GetCellType() != SVTK_QUADRATIC_LINEAR_QUAD &&
            cell->GetCellType() != SVTK_QUADRATIC_LINEAR_WEDGE)
          {
            rval += CompareHigherOrderCell(qf1, f2);
          }
        }
      }
    }
    // Cleanup
    for (unsigned int c = 0; c < depth; ++c)
    {
      svtkCell* cell = cellArray[c];
      if (cell)
      {
        cell->Delete();
      }
    }
  }

  return rval;
}
