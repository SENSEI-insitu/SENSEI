/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestMappedGridDeepCopy

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*----------------------------------------------------------------------------
  This test was written by Menno Deij - van Rijswijk (MARIN).
----------------------------------------------------------------------------*/

#include "svtkCell.h" // for cell types
#include "svtkCellIterator.h"
#include "svtkDataArray.h"
#include "svtkDebugLeaks.h"
#include "svtkIdList.h"
#include "svtkIdTypeArray.h"
#include "svtkInformation.h"
#include "svtkMappedUnstructuredGrid.h"
#include "svtkNew.h"
#include "svtkPoints.h"
#include "svtkTestUtilities.h"
#include "svtkUnstructuredGrid.h"
#include "svtkXMLUnstructuredGridReader.h"
#include "svtkXMLUnstructuredGridWriter.h"

#include <algorithm>
#include <fstream>
#include <string>

namespace
{ // this namespace contains the supporting mapped grid definition used in the test

template <class I>
class MappedCellIterator : public svtkCellIterator
{
public:
  svtkTemplateTypeMacro(MappedCellIterator<I>, svtkCellIterator);
  typedef MappedCellIterator<I> ThisType;

  static MappedCellIterator<I>* New();

  void SetMappedUnstructuredGrid(svtkMappedUnstructuredGrid<I, ThisType>* grid);

  void PrintSelf(std::ostream& os, svtkIndent id) override;

  bool IsDoneWithTraversal() override;
  svtkIdType GetCellId() override;

protected:
  MappedCellIterator();
  ~MappedCellIterator() override;
  void ResetToFirstCell() override { this->CellId = 0; }
  void IncrementToNextCell() override { this->CellId++; }
  void FetchCellType() override;
  void FetchPointIds() override;
  void FetchPoints() override;
  void FetchFaces() override;

private:
  MappedCellIterator(const MappedCellIterator&) = delete;
  void operator=(const MappedCellIterator&) = delete;

  svtkIdType CellId;
  svtkIdType NumberOfCells;
  svtkSmartPointer<I> Impl;
  svtkSmartPointer<svtkPoints> GridPoints;
};

template <class I>
MappedCellIterator<I>* MappedCellIterator<I>::New()
{
  SVTK_STANDARD_NEW_BODY(ThisType);
}

template <class I>
MappedCellIterator<I>::MappedCellIterator()
  : CellId(0)
  , NumberOfCells(0)
  , Impl(nullptr)
  , GridPoints(nullptr)
{
}

template <class I>
void MappedCellIterator<I>::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "Mapped Internal Block" << endl;
}

template <class I>
MappedCellIterator<I>::~MappedCellIterator() = default;

template <class I>
void MappedCellIterator<I>::SetMappedUnstructuredGrid(svtkMappedUnstructuredGrid<I, ThisType>* grid)
{
  this->Impl = grid->GetImplementation();
  this->CellId = 0;
  this->GridPoints = grid->GetPoints();
  this->NumberOfCells = grid->GetNumberOfCells();
}

template <class I>
bool MappedCellIterator<I>::IsDoneWithTraversal()
{
  if (!this->Impl)
    return true;
  return CellId >= this->NumberOfCells;
}

template <class I>
svtkIdType MappedCellIterator<I>::GetCellId()
{
  return this->CellId;
}

template <class I>
void MappedCellIterator<I>::FetchCellType()
{
  this->CellType = Impl->GetCellType(this->CellId);
}

template <class I>
void MappedCellIterator<I>::FetchPointIds()
{
  this->Impl->GetCellPoints(this->CellId, this->PointIds);
}

template <class I>
void MappedCellIterator<I>::FetchPoints()
{
  this->GridPoints->GetPoints(this->GetPointIds(), this->Points);
}

template <class I>
void MappedCellIterator<I>::FetchFaces()
{
  this->Impl->GetFaceStream(this->CellId, this->Faces);
}

class MappedGrid;
class MappedGridImpl : public svtkObject
{
public:
  static MappedGridImpl* New();

  void Initialize(svtkUnstructuredGrid* ug)
  {
    ug->Register(this);
    _grid = ug;
  }

  void PrintSelf(std::ostream& os, svtkIndent id) override;

  // API for svtkMappedUnstructuredGrid implementation
  virtual int GetCellType(svtkIdType cellId);
  virtual void GetCellPoints(svtkIdType cellId, svtkIdList* ptIds);
  virtual void GetFaceStream(svtkIdType cellId, svtkIdList* ptIds);
  virtual void GetPointCells(svtkIdType ptId, svtkIdList* cellIds);
  virtual int GetMaxCellSize();
  virtual void GetIdsOfCellsOfType(int type, svtkIdTypeArray* array);
  virtual int IsHomogeneous();

  // This container is read only -- these methods do nothing but print a warning.
  void Allocate(svtkIdType numCells, int extSize = 1000);
  svtkIdType InsertNextCell(int type, svtkIdList* ptIds);
  svtkIdType InsertNextCell(int type, svtkIdType npts, const svtkIdType ptIds[])
    SVTK_SIZEHINT(ptIds, npts);
  svtkIdType InsertNextCell(int type, svtkIdType npts, const svtkIdType ptIds[], svtkIdType nfaces,
    const svtkIdType faces[]) SVTK_SIZEHINT(ptIds, npts) SVTK_SIZEHINT(faces, nfaces);
  void ReplaceCell(svtkIdType cellId, int npts, const svtkIdType pts[]) SVTK_SIZEHINT(pts, npts);

  svtkIdType GetNumberOfCells();
  void SetOwner(MappedGrid* owner) { this->Owner = owner; }

  svtkPoints* GetPoints() { return _grid->GetPoints(); }

protected:
  MappedGridImpl() = default;
  ~MappedGridImpl() override { _grid->UnRegister(this); }

private:
  svtkUnstructuredGrid* _grid;
  MappedGrid* Owner;
};

svtkStandardNewMacro(MappedGridImpl);

void MappedGridImpl::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "Mapped Grid Implementation" << endl;
}

int MappedGridImpl::GetCellType(svtkIdType cellId)
{
  return _grid->GetCellType(cellId);
}

int MappedGridImpl::GetMaxCellSize()
{
  return _grid->GetMaxCellSize();
}

void MappedGridImpl::GetCellPoints(svtkIdType cellId, svtkIdList* ptIds)
{
  _grid->GetCellPoints(cellId, ptIds);
}

void MappedGridImpl::GetFaceStream(svtkIdType cellId, svtkIdList* ptIds)
{
  _grid->GetFaceStream(cellId, ptIds);
}

void MappedGridImpl::GetPointCells(svtkIdType ptId, svtkIdList* cellIds)
{
  _grid->GetPointCells(ptId, cellIds);
}

int MappedGridImpl::IsHomogeneous()
{
  return _grid->IsHomogeneous();
}

svtkIdType MappedGridImpl::GetNumberOfCells()
{
  return _grid->GetNumberOfCells();
}

void MappedGridImpl::GetIdsOfCellsOfType(int type, svtkIdTypeArray* array)
{
  _grid->GetIdsOfCellsOfType(type, array);
}

void MappedGridImpl::Allocate(svtkIdType svtkNotUsed(numCells), int svtkNotUsed(extSize))
{
  svtkWarningMacro(<< "Read only block\n");
}

svtkIdType MappedGridImpl::InsertNextCell(int svtkNotUsed(type), svtkIdList* svtkNotUsed(ptIds))
{
  svtkWarningMacro(<< "Read only block\n");
  return -1;
}

svtkIdType MappedGridImpl::InsertNextCell(
  int svtkNotUsed(type), svtkIdType svtkNotUsed(npts), const svtkIdType svtkNotUsed(ptIds)[])
{
  svtkWarningMacro(<< "Read only block\n");
  return -1;
}

svtkIdType MappedGridImpl::InsertNextCell(int svtkNotUsed(type), svtkIdType svtkNotUsed(npts),
  const svtkIdType svtkNotUsed(ptIds)[], svtkIdType svtkNotUsed(nfaces),
  const svtkIdType svtkNotUsed(faces)[])
{
  svtkWarningMacro(<< "Read only block\n");
  return -1;
}

void MappedGridImpl::ReplaceCell(
  svtkIdType svtkNotUsed(cellId), int svtkNotUsed(npts), const svtkIdType svtkNotUsed(pts)[])
{
  svtkWarningMacro(<< "Read only block\n");
}

class MappedGrid
  : public svtkMappedUnstructuredGrid<MappedGridImpl, MappedCellIterator<MappedGridImpl> >
{
public:
  typedef svtkMappedUnstructuredGrid<MappedGridImpl, MappedCellIterator<MappedGridImpl> > _myBase;

  int GetDataObjectType() override { return SVTK_UNSTRUCTURED_GRID_BASE; }

  static MappedGrid* New();

  svtkPoints* GetPoints() override { return this->GetImplementation()->GetPoints(); }

  svtkIdType GetNumberOfPoints() override
  {
    return this->GetImplementation()->GetPoints()->GetNumberOfPoints();
  }

protected:
  MappedGrid()
  {
    MappedGridImpl* ig = MappedGridImpl::New();
    ig->SetOwner(this);
    this->SetImplementation(ig);
    ig->Delete();
  }
  ~MappedGrid() override = default;

private:
  MappedGrid(const MappedGrid&) = delete;
  void operator=(const MappedGrid&) = delete;
};

svtkStandardNewMacro(MappedGrid);

} // end anonymous namespace

using namespace std;

int TestMappedGridDeepCopy(int svtkNotUsed(argc), char*[] svtkNotUsed(argv))
{
  svtkNew<svtkPoints> points;

  points->InsertNextPoint(0, 0, 0);
  points->InsertNextPoint(1, 0, 0);
  points->InsertNextPoint(1, 1, 0);
  points->InsertNextPoint(0, 1, 0);

  points->InsertNextPoint(0, 0, 1);
  points->InsertNextPoint(1, 0, 1);
  points->InsertNextPoint(1, 1, 1);
  points->InsertNextPoint(0, 1, 1);

  points->InsertNextPoint(.5, .5, 2);
  points->InsertNextPoint(.5, .5, -1);

  svtkNew<svtkUnstructuredGrid> original;
  original->SetPoints(points);

  original->Allocate(3); // allocate for 3 cells

  svtkNew<svtkIdList> ids;

  // add a hexahedron of the first 8 points (i.e. a cube)
  ids->InsertNextId(0);
  ids->InsertNextId(1);
  ids->InsertNextId(2);
  ids->InsertNextId(3);
  ids->InsertNextId(4);
  ids->InsertNextId(5);
  ids->InsertNextId(6);
  ids->InsertNextId(7);
  original->InsertNextCell(SVTK_HEXAHEDRON, ids.GetPointer());
  ids->Reset();

  // add a polyhedron comprise of the top hexahedron face
  // and four triangles to the 9th point
  ids->InsertNextId(4);
  ids->InsertNextId(5);
  ids->InsertNextId(6);
  ids->InsertNextId(7);
  ids->InsertNextId(8);

  // this list of faces does NOT include nfaces as the first entry
  svtkNew<svtkIdList> faces;
  // top face of four points
  faces->InsertNextId(4);

  faces->InsertNextId(4);
  faces->InsertNextId(5);
  faces->InsertNextId(6);
  faces->InsertNextId(7);

  // four triangle side faces, each of three points
  faces->InsertNextId(3);
  faces->InsertNextId(4);
  faces->InsertNextId(5);
  faces->InsertNextId(8);

  faces->InsertNextId(3);
  faces->InsertNextId(5);
  faces->InsertNextId(6);
  faces->InsertNextId(8);

  faces->InsertNextId(3);
  faces->InsertNextId(6);
  faces->InsertNextId(7);
  faces->InsertNextId(8);

  faces->InsertNextId(3);
  faces->InsertNextId(7);
  faces->InsertNextId(4);
  faces->InsertNextId(8);

  // insert the polyhedron cell
  original->InsertNextCell(
    SVTK_POLYHEDRON, 5, ids.GetPointer()->GetPointer(0), 5, faces.GetPointer()->GetPointer(0));

  svtkNew<svtkGenericCell> aCell;
  original->GetCell(1, aCell.GetPointer());

  // this is the full faces list, *including* the leading nfaces
  svtkIdType* cellFaces = aCell->GetFaces();
  if (cellFaces[0] != 5)
  {
    cerr << " expected 5 faces, got " << cellFaces[0] << endl;
    return EXIT_FAILURE;
  }
  for (int i = 0; i < faces->GetNumberOfIds(); ++i)
  {
    if (faces->GetId(i) != cellFaces[i + 1])
    {
      cerr << "faces array not identical at position " << i << endl;
      return EXIT_FAILURE;
    }
  }

  // put another pyramid on the bottom towards the 10th point
  faces->Reset();
  ids->Reset();

  // the list of points that the pyramid references
  ids->InsertNextId(0);
  ids->InsertNextId(1);
  ids->InsertNextId(2);
  ids->InsertNextId(3);
  ids->InsertNextId(9);

  // bottom face of four points
  faces->InsertNextId(4);

  faces->InsertNextId(0);
  faces->InsertNextId(1);
  faces->InsertNextId(2);
  faces->InsertNextId(3);

  // four side faces, each of three points
  faces->InsertNextId(3);
  faces->InsertNextId(0);
  faces->InsertNextId(1);
  faces->InsertNextId(9);

  faces->InsertNextId(3);
  faces->InsertNextId(1);
  faces->InsertNextId(2);
  faces->InsertNextId(9);

  faces->InsertNextId(3);
  faces->InsertNextId(2);
  faces->InsertNextId(3);
  faces->InsertNextId(9);

  faces->InsertNextId(3);
  faces->InsertNextId(3);
  faces->InsertNextId(0);
  faces->InsertNextId(9);

  // insert the cell. We now have two pyramids with a cube in between
  original->InsertNextCell(
    SVTK_POLYHEDRON, 5, ids.GetPointer()->GetPointer(0), 5, faces.GetPointer()->GetPointer(0));

  // create a mapped grid which basically takes the original grid
  // and uses it to map to.
  svtkNew<MappedGrid> mg;
  mg->GetImplementation()->Initialize(original);

  // copy the mapped grid into a normal unstructured grid.
  // copying will proceed via the super class
  // svtkUnstructuredGridBase::DeepCopy function
  // implementation that uses a cell iterator. This will
  // invoke to InsertNextCell function with face list
  // for *all* cells (even if they are not SVTK_POLYHEDRON).
  // In the old implementation this gave copy errors. The fix
  // proposed together with this test addresses that issue.
  svtkNew<svtkUnstructuredGrid> copy;
  copy->Allocate(mg->GetNumberOfCells());
  copy->DeepCopy(mg.GetPointer());

  svtkCellIterator* oIt = original->NewCellIterator();
  svtkCellIterator* cIt = copy->NewCellIterator();

  svtkNew<svtkGenericCell> orig, copied;
  for (cIt->InitTraversal(), oIt->InitTraversal();
       !cIt->IsDoneWithTraversal() && !oIt->IsDoneWithTraversal();
       cIt->GoToNextCell(), oIt->GoToNextCell())
  {
    oIt->GetCell(orig.GetPointer());
    cIt->GetCell(copied.GetPointer());

    if (cIt->GetCellType() != oIt->GetCellType())
    {
      cerr << "Cell types do not match" << endl;
      return EXIT_FAILURE;
    }

    if (cIt->GetCellType() == SVTK_POLYHEDRON)
    {
      svtkIdList* oFaces = oIt->GetFaces();
      svtkIdList* cFaces = cIt->GetFaces();

      if (cFaces->GetNumberOfIds() != oFaces->GetNumberOfIds())
      {
        cerr << "Face id list length does not match" << endl;
        cerr << "Original: ";
        for (svtkIdType i = 0; i < oFaces->GetNumberOfIds(); ++i)
        {
          cerr << oFaces->GetId(i) << " ";
        }
        cerr << endl;

        cerr << "Copied:   ";
        for (svtkIdType i = 0; i < cFaces->GetNumberOfIds(); ++i)
          cerr << cFaces->GetId(i) << " ";
        cerr << endl;

        return EXIT_FAILURE;
      }

      for (svtkIdType i = 0; i < cFaces->GetNumberOfIds(); ++i)
      {
        svtkIdType c = cFaces->GetId(i);
        svtkIdType o = oFaces->GetId(i);

        if (c != o)
        {
          cerr << "Face id list content does not match at" << i << endl;
          return EXIT_FAILURE;
        }
      }
    }
  }
  oIt->Delete();
  cIt->Delete();
  return EXIT_SUCCESS;
}
