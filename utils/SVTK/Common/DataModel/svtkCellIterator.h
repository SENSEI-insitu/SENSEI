/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCellIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkCellIterator
 * @brief   Efficient cell iterator for svtkDataSet topologies.
 *
 *
 * svtkCellIterator provides a method for traversing cells in a data set. Call
 * the svtkDataSet::NewCellIterator() method to use this class.
 *
 * The cell is represented as a set of three pieces of information: The cell
 * type, the ids of the points constituting the cell, and the points themselves.
 * This iterator fetches these as needed. If only the cell type is used,
 * the type is not looked up until GetCellType is called, and the point
 * information is left uninitialized. This allows efficient screening of cells,
 * since expensive point lookups may be skipped depending on the cell type/etc.
 *
 * An example usage of this class:
 * ~~~
 * void myWorkerFunction(svtkDataSet *ds)
 * {
 *   svtkCellIterator *it = ds->NewCellIterator();
 *   for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextCell())
 *     {
 *     if (it->GetCellType() != SVTK_TETRA)
 *       {
 *       continue; // Skip non-tetrahedral cells
 *       }
 *
 *     svtkIdList *pointIds = it->GetPointIds();
 *     // Do screening on the point ids, maybe figure out scalar range and skip
 *        cells that do not lie in a certain range?
 *
 *     svtkPoints *points = it->GetPoints();
 *     // Do work using the cell points, or ...
 *
 *     svtkGenericCell *cell = ...;
 *     it->GetCell(cell);
 *     // ... do work with a svtkCell.
 *     }
 *   it->Delete();
 * }
 * ~~~
 *
 * The example above pulls in bits of information as needed to filter out cells
 * that aren't relevant. The least expensive lookups are performed first
 * (cell type, then point ids, then points/full cell) to prevent wasted cycles
 * fetching unnecessary data. Also note that at the end of the loop, the
 * iterator must be deleted as these iterators are svtkObject subclasses.
 */

#ifndef svtkCellIterator_h
#define svtkCellIterator_h

#include "svtkCellType.h"              // For SVTK_EMPTY_CELL
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkIdList.h"                // For inline methods
#include "svtkNew.h"                   // For svtkNew
#include "svtkObject.h"

class svtkGenericCell;
class svtkPoints;

class SVTKCOMMONDATAMODEL_EXPORT svtkCellIterator : public svtkObject
{
public:
  void PrintSelf(ostream& os, svtkIndent indent) override;
  svtkAbstractTypeMacro(svtkCellIterator, svtkObject);

  /**
   * Reset to the first cell.
   */
  void InitTraversal();

  /**
   * Increment to next cell. Always safe to call.
   */
  void GoToNextCell();

  /**
   * Returns false while the iterator is valid. Always safe to call.
   */
  virtual bool IsDoneWithTraversal() = 0;

  /**
   * Get the current cell type (e.g. SVTK_LINE, SVTK_VERTEX, SVTK_TETRA, etc).
   * This should only be called when IsDoneWithTraversal() returns false.
   */
  int GetCellType();

  /**
   * Get the current cell dimension (0, 1, 2, or 3). This should only be called
   * when IsDoneWithTraversal() returns false.
   */
  int GetCellDimension();

  /**
   * Get the id of the current cell.
   */
  virtual svtkIdType GetCellId() = 0;

  /**
   * Get the ids of the points in the current cell.
   * This should only be called when IsDoneWithTraversal() returns false.
   */
  svtkIdList* GetPointIds();

  /**
   * Get the points in the current cell.
   * This is usually a very expensive call, and should be avoided when possible.
   * This should only be called when IsDoneWithTraversal() returns false.
   */
  svtkPoints* GetPoints();

  /**
   * Get the faces for a polyhedral cell. This is only valid when CellType
   * is SVTK_POLYHEDRON.
   */
  svtkIdList* GetFaces();

  /**
   * Write the current full cell information into the argument.
   * This is usually a very expensive call, and should be avoided when possible.
   * This should only be called when IsDoneWithTraversal() returns false.
   */
  void GetCell(svtkGenericCell* cell);

  /**
   * Return the number of points in the current cell.
   * This should only be called when IsDoneWithTraversal() returns false.
   */
  svtkIdType GetNumberOfPoints();

  /**
   * Return the number of faces in the current cell.
   * This should only be called when IsDoneWithTraversal() returns false.
   */
  svtkIdType GetNumberOfFaces();

protected:
  svtkCellIterator();
  ~svtkCellIterator() override;

  /**
   * Update internal state to point to the first cell.
   */
  virtual void ResetToFirstCell() = 0;

  /**
   * Update internal state to point to the next cell.
   */
  virtual void IncrementToNextCell() = 0;

  /**
   * Lookup the cell type in the data set and store it in this->CellType.
   */
  virtual void FetchCellType() = 0;

  /**
   * Lookup the cell point ids in the data set and store them in this->PointIds.
   */
  virtual void FetchPointIds() = 0;

  /**
   * Lookup the cell points in the data set and store them in this->Points.
   */
  virtual void FetchPoints() = 0;

  /**
   * Lookup the cell faces in the data set and store them in this->Faces.
   * Few data sets support faces, so this method has a no-op default
   * implementation. See svtkUnstructuredGrid::GetFaceStream for
   * a description of the layout that Faces should have.
   */
  virtual void FetchFaces() {}

  int CellType;
  svtkPoints* Points;
  svtkIdList* PointIds;
  svtkIdList* Faces;

private:
  svtkCellIterator(const svtkCellIterator&) = delete;
  void operator=(const svtkCellIterator&) = delete;

  enum
  {
    UninitializedFlag = 0x0,
    CellTypeFlag = 0x1,
    PointIdsFlag = 0x2,
    PointsFlag = 0x4,
    FacesFlag = 0x8
  };

  void ResetCache()
  {
    this->CacheFlags = UninitializedFlag;
    this->CellType = SVTK_EMPTY_CELL;
  }

  void SetCache(unsigned char flags) { this->CacheFlags |= flags; }

  bool CheckCache(unsigned char flags) { return (this->CacheFlags & flags) == flags; }

  svtkNew<svtkPoints> PointsContainer;
  svtkNew<svtkIdList> PointIdsContainer;
  svtkNew<svtkIdList> FacesContainer;
  unsigned char CacheFlags;
};

//------------------------------------------------------------------------------
inline void svtkCellIterator::InitTraversal()
{
  this->ResetToFirstCell();
  this->ResetCache();
}

//------------------------------------------------------------------------------
inline void svtkCellIterator::GoToNextCell()
{
  this->IncrementToNextCell();
  this->ResetCache();
}

//------------------------------------------------------------------------------
inline int svtkCellIterator::GetCellType()
{
  if (!this->CheckCache(CellTypeFlag))
  {
    this->FetchCellType();
    this->SetCache(CellTypeFlag);
  }
  return this->CellType;
}

//------------------------------------------------------------------------------
inline svtkIdList* svtkCellIterator::GetPointIds()
{
  if (!this->CheckCache(PointIdsFlag))
  {
    this->FetchPointIds();
    this->SetCache(PointIdsFlag);
  }
  return this->PointIds;
}

//------------------------------------------------------------------------------
inline svtkPoints* svtkCellIterator::GetPoints()
{
  if (!this->CheckCache(PointsFlag))
  {
    this->FetchPoints();
    this->SetCache(PointsFlag);
  }
  return this->Points;
}

//------------------------------------------------------------------------------
inline svtkIdList* svtkCellIterator::GetFaces()
{
  if (!this->CheckCache(FacesFlag))
  {
    this->FetchFaces();
    this->SetCache(FacesFlag);
  }
  return this->Faces;
}

//------------------------------------------------------------------------------
inline svtkIdType svtkCellIterator::GetNumberOfPoints()
{
  if (!this->CheckCache(PointIdsFlag))
  {
    this->FetchPointIds();
    this->SetCache(PointIdsFlag);
  }
  return this->PointIds->GetNumberOfIds();
}

//------------------------------------------------------------------------------
inline svtkIdType svtkCellIterator::GetNumberOfFaces()
{
  switch (this->GetCellType())
  {
    case SVTK_EMPTY_CELL:
    case SVTK_VERTEX:
    case SVTK_POLY_VERTEX:
    case SVTK_LINE:
    case SVTK_POLY_LINE:
    case SVTK_TRIANGLE:
    case SVTK_TRIANGLE_STRIP:
    case SVTK_POLYGON:
    case SVTK_PIXEL:
    case SVTK_QUAD:
    case SVTK_QUADRATIC_EDGE:
    case SVTK_QUADRATIC_TRIANGLE:
    case SVTK_QUADRATIC_QUAD:
    case SVTK_QUADRATIC_POLYGON:
    case SVTK_BIQUADRATIC_QUAD:
    case SVTK_QUADRATIC_LINEAR_QUAD:
    case SVTK_BIQUADRATIC_TRIANGLE:
    case SVTK_CUBIC_LINE:
    case SVTK_CONVEX_POINT_SET:
    case SVTK_PARAMETRIC_CURVE:
    case SVTK_PARAMETRIC_SURFACE:
    case SVTK_PARAMETRIC_TRI_SURFACE:
    case SVTK_PARAMETRIC_QUAD_SURFACE:
    case SVTK_HIGHER_ORDER_EDGE:
    case SVTK_HIGHER_ORDER_TRIANGLE:
    case SVTK_HIGHER_ORDER_QUAD:
    case SVTK_HIGHER_ORDER_POLYGON:
    case SVTK_LAGRANGE_CURVE:
    case SVTK_LAGRANGE_TRIANGLE:
    case SVTK_LAGRANGE_QUADRILATERAL:
    case SVTK_BEZIER_CURVE:
    case SVTK_BEZIER_TRIANGLE:
    case SVTK_BEZIER_QUADRILATERAL:
      return 0;

    case SVTK_TETRA:
    case SVTK_QUADRATIC_TETRA:
    case SVTK_PARAMETRIC_TETRA_REGION:
    case SVTK_HIGHER_ORDER_TETRAHEDRON:
    case SVTK_LAGRANGE_TETRAHEDRON:
    case SVTK_BEZIER_TETRAHEDRON:
      return 4;

    case SVTK_PYRAMID:
    case SVTK_QUADRATIC_PYRAMID:
    case SVTK_HIGHER_ORDER_PYRAMID:
    case SVTK_WEDGE:
    case SVTK_QUADRATIC_WEDGE:
    case SVTK_QUADRATIC_LINEAR_WEDGE:
    case SVTK_BIQUADRATIC_QUADRATIC_WEDGE:
    case SVTK_HIGHER_ORDER_WEDGE:
    case SVTK_LAGRANGE_WEDGE:
    case SVTK_BEZIER_WEDGE:
      return 5;

    case SVTK_VOXEL:
    case SVTK_HEXAHEDRON:
    case SVTK_QUADRATIC_HEXAHEDRON:
    case SVTK_TRIQUADRATIC_HEXAHEDRON:
    case SVTK_HIGHER_ORDER_HEXAHEDRON:
    case SVTK_PARAMETRIC_HEX_REGION:
    case SVTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON:
    case SVTK_LAGRANGE_HEXAHEDRON:
    case SVTK_BEZIER_HEXAHEDRON:
      return 6;

    case SVTK_PENTAGONAL_PRISM:
      return 7;

    case SVTK_HEXAGONAL_PRISM:
      return 8;

    case SVTK_POLYHEDRON: // Need to look these up
      if (!this->CheckCache(FacesFlag))
      {
        this->FetchFaces();
        this->SetCache(FacesFlag);
      }
      return this->Faces->GetNumberOfIds() != 0 ? this->Faces->GetId(0) : 0;

    default:
      svtkGenericWarningMacro("Unknown cell type: " << this->CellType);
      break;
  }

  return 0;
}

#endif // svtkCellIterator_h
