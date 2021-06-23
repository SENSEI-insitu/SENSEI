/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCellIterator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkCellIterator.h"

#include "svtkGenericCell.h"
#include "svtkIdList.h"
#include "svtkNew.h"
#include "svtkPoints.h"

//------------------------------------------------------------------------------
void svtkCellIterator::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "CacheFlags: ";
  switch (this->CacheFlags)
  {
    case UninitializedFlag:
      os << "UninitializedFlag" << endl;
      break;
    default:
    {
      bool addSplit = false;

      if (this->CheckCache(CellTypeFlag))
      {
        os << "CellTypeFlag";
        addSplit = true;
      }

      if (this->CheckCache(PointIdsFlag))
      {
        os << (addSplit ? " | " : "") << "PointIdsFlag";
        addSplit = true;
      }

      if (this->CheckCache(PointsFlag))
      {
        os << (addSplit ? " | " : "") << "PointsFlag";
        addSplit = true;
      }

      if (this->CheckCache(FacesFlag))
      {
        os << (addSplit ? " | " : "") << "FacesFlag";
      }
      os << endl;
    }
  }

  os << indent << "CellType: " << this->CellType << endl;
  os << indent << "Points:" << endl;
  this->Points->PrintSelf(os, indent.GetNextIndent());
  os << indent << "PointIds:" << endl;
  this->PointIds->PrintSelf(os, indent.GetNextIndent());
  os << indent << "Faces:" << endl;
  this->Faces->PrintSelf(os, indent.GetNextIndent());
}

//------------------------------------------------------------------------------
int svtkCellIterator::GetCellDimension()
{
  // For the most common cell types, this is a fast call. If the cell type is
  // more exotic, then the cell must be grabbed and queried directly, which is
  // slow.

  int cellType = this->GetCellType();

  switch (cellType)
  {
    case SVTK_EMPTY_CELL:
    case SVTK_VERTEX:
    case SVTK_POLY_VERTEX:
      return 0;
    case SVTK_LINE:
    case SVTK_POLY_LINE:
    case SVTK_QUADRATIC_EDGE:
    case SVTK_CUBIC_LINE:
    case SVTK_LAGRANGE_CURVE:
    case SVTK_BEZIER_CURVE:
      return 1;
    case SVTK_TRIANGLE:
    case SVTK_QUAD:
    case SVTK_PIXEL:
    case SVTK_POLYGON:
    case SVTK_TRIANGLE_STRIP:
    case SVTK_QUADRATIC_TRIANGLE:
    case SVTK_QUADRATIC_QUAD:
    case SVTK_QUADRATIC_POLYGON:
    case SVTK_LAGRANGE_TRIANGLE:
    case SVTK_LAGRANGE_QUADRILATERAL:
    case SVTK_BEZIER_TRIANGLE:
    case SVTK_BEZIER_QUADRILATERAL:
      return 2;
    case SVTK_TETRA:
    case SVTK_VOXEL:
    case SVTK_HEXAHEDRON:
    case SVTK_WEDGE:
    case SVTK_PYRAMID:
    case SVTK_PENTAGONAL_PRISM:
    case SVTK_HEXAGONAL_PRISM:
    case SVTK_QUADRATIC_TETRA:
    case SVTK_QUADRATIC_HEXAHEDRON:
    case SVTK_QUADRATIC_WEDGE:
    case SVTK_QUADRATIC_PYRAMID:
    case SVTK_LAGRANGE_TETRAHEDRON:
    case SVTK_LAGRANGE_HEXAHEDRON:
    case SVTK_LAGRANGE_WEDGE:
    case SVTK_BEZIER_TETRAHEDRON:
    case SVTK_BEZIER_HEXAHEDRON:
    case SVTK_BEZIER_WEDGE:
      return 3;
    default:
      svtkNew<svtkGenericCell> cell;
      this->GetCell(cell);
      return cell->GetCellDimension();
  }
}

//------------------------------------------------------------------------------
void svtkCellIterator::GetCell(svtkGenericCell* cell)
{
  cell->SetCellType(this->GetCellType());
  cell->SetPointIds(this->GetPointIds());
  cell->SetPoints(this->GetPoints());

  cell->SetFaces(nullptr);
  if (cell->RequiresExplicitFaceRepresentation())
  {
    svtkIdList* faces = this->GetFaces();
    if (faces->GetNumberOfIds() != 0)
    {
      cell->SetFaces(faces->GetPointer(0));
    }
  }

  if (cell->RequiresInitialization())
  {
    cell->Initialize();
  }
}

//------------------------------------------------------------------------------
svtkCellIterator::svtkCellIterator()
  : CellType(SVTK_EMPTY_CELL)
  , CacheFlags(UninitializedFlag)
{
  this->Points = this->PointsContainer;
  this->PointIds = this->PointIdsContainer;
  this->Faces = this->FacesContainer;
}

//------------------------------------------------------------------------------
svtkCellIterator::~svtkCellIterator() = default;
