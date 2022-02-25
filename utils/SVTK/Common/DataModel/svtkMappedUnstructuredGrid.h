/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMappedUnstructuredGrid.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkMappedUnstructuredGrid
 * @brief   Allows datasets with arbitrary storage
 * layouts to be used with SVTK.
 *
 *
 * This class fulfills the svtkUnstructuredGridBase API while delegating to a
 * arbitrary implementation of the dataset topology. The purpose of
 * svtkMappedUnstructuredGrid is to allow external data structures to be used
 * directly in a SVTK pipeline, e.g. for in-situ analysis of a running
 * simulation.
 *
 * When introducing an external data structure into SVTK, there are 3 principle
 * components of the dataset to consider:
 * - Points
 * - Cells (topology)
 * - Point/Cell attributes
 *
 * Points and attributes can be handled by subclassing svtkMappedDataArray and
 * implementing that interface to adapt the external data structures into SVTK.
 * The svtkMappedDataArray subclasses can then be used as the svtkPoints's Data
 * member (for points/nodes) or added directly to svtkPointData, svtkCellData, or
 * svtkFieldData for attribute information. Filters used in the pipeline will
 * need to be modified to remove calls to svtkDataArray::GetVoidPointer and use
 * a suitable svtkArrayDispatch instead.
 *
 * Introducing an arbitrary topology implementation into SVTK requires the use of
 * the svtkMappedUnstructuredGrid class. Unlike the data array counterpart, the
 * mapped unstructured grid is not subclassed, but rather takes an adaptor
 * class as a template argument. This is to allow cheap shallow copies of the
 * data by passing the reference-counted implementation object to new instances
 * of svtkMappedUnstructuredGrid.
 *
 * The implementation class should derive from svtkObject (for reference
 * counting) and implement the usual svtkObject API requirements, such as a
 * static New() method and PrintSelf function. The following methods must also
 * be implemented:
 * - svtkIdType GetNumberOfCells()
 * - int GetCellType(svtkIdType cellId)
 * - void GetCellPoints(svtkIdType cellId, svtkIdList *ptIds)
 * - void GetPointCells(svtkIdType ptId, svtkIdList *cellIds)
 * - int GetMaxCellSize()
 * - void GetIdsOfCellsOfType(int type, svtkIdTypeArray *array)
 * - int IsHomogeneous()
 * - void Allocate(svtkIdType numCells, int extSize = 1000)
 * - svtkIdType InsertNextCell(int type, svtkIdList *ptIds)
 * - svtkIdType InsertNextCell(int type, svtkIdType npts, const svtkIdType ptIds[])
 * - svtkIdType InsertNextCell(int type, svtkIdType npts, const svtkIdType ptIds[],
 *                            svtkIdType nfaces, const int faces[])
 * - void ReplaceCell(svtkIdType cellId, int npts, const svtkIdType pts[])
 *
 * These methods should provide the same functionality as defined in
 * svtkUnstructuredGrid. See that class's documentation for more information.
 *
 * Note that since the implementation class is used as a compile-time template
 * parameter in svtkMappedUnstructuredGrid, the above methods do not need be
 * virtuals. The compiler will statically bind the calls, making dynamic vtable
 * lookups unnecessary and giving a slight performance boost.
 *
 * Adapting a filter or algorithm to safely traverse the
 * svtkMappedUnstructuredGrid's topology requires removing calls the following
 * implementation-dependent svtkUnstructuredGrid methods:
 * - svtkUnstructuredGrid::GetCellTypesArray()
 * - svtkUnstructuredGrid::GetCellLocationsArray()
 * - svtkUnstructuredGrid::GetCellLinks()
 * - svtkUnstructuredGrid::GetCells()
 * Access to the values returned by these methods should be replaced by the
 * equivalent random-access lookup methods in the svtkUnstructuredGridBase API,
 * or use svtkCellIterator (see svtkDataSet::NewCellIterator) for sequential
 * access.
 *
 * A custom svtkCellIterator implementation may be specified for a particular
 * svtkMappedUnstructuredGrid as the second template parameter. By default,
 * svtkMappedUnstructuredGridCellIterator will be used, which increments an
 * internal cell id counter and performs random-access lookup as-needed. More
 * efficient implementations may be used with data structures better suited for
 * sequential access, see svtkUnstructuredGridCellIterator for an example.
 *
 * A set of four macros are provided to generate a concrete subclass of
 * svtkMappedUnstructuredGrid with a specified implementation, cell iterator,
 * and export declaration. They are:
 * - svtkMakeMappedUnstructuredGrid(_className, _impl)
 *   - Create a subclass of svtkMappedUnstructuredGrid using _impl implementation
 *     that is named _className.
 * - svtkMakeMappedUnstructuredGridWithIter(_className, _impl, _cIter)
 *   - Create a subclass of svtkMappedUnstructuredGrid using _impl implementation
 *     and _cIter svtkCellIterator that is named _className.
 * - svtkMakeExportedMappedUnstructuredGrid(_className, _impl, _exportDecl)
 *   - Create a subclass of svtkMappedUnstructuredGrid using _impl implementation
 *     that is named _className. _exportDecl is used to decorate the class
 *     declaration.
 * - svtkMakeExportedMappedUnstructuredGridWithIter(_className, _impl, _cIter, _exportDecl)
 *   - Create a subclass of svtkMappedUnstructuredGrid using _impl implementation
 *     and _cIter svtkCellIterator that is named _className. _exportDecl is used
 *     to decorate the class declaration.
 *
 * To instantiate a svtkMappedUnstructuredGrid subclass created by the above
 * macro, the follow pattern is encouraged:
 *
 * @code
 * MyGrid.h:
 * ----------------------------------------------------------------------
 * class MyGridImplementation : public svtkObject
 * {
 * public:
 *   ... (svtkObject required API) ...
 *   ... (svtkMappedUnstructuredGrid Implementation required API) ...
 *   void SetImplementationDetails(...raw data from external source...);
 * };
 *
 * svtkMakeMappedUnstructuredGrid(MyGrid, MyGridImplementation)
 *
 * SomeSource.cxx
 * ----------------------------------------------------------------------
 * svtkNew<MyGrid> grid;
 * grid->GetImplementation()->SetImplementationDetails(...);
 * // grid is now ready to use.
 * @endcode
 *
 * The svtkCPExodusIIElementBlock class provides an example of
 * svtkMappedUnstructuredGrid usage, adapting the Exodus II data structures for
 * the SVTK pipeline.
 */

#ifndef svtkMappedUnstructuredGrid_h
#define svtkMappedUnstructuredGrid_h

#include "svtkUnstructuredGridBase.h"

#include "svtkMappedUnstructuredGridCellIterator.h" // For default cell iterator
#include "svtkNew.h"                                // For svtkNew
#include "svtkSmartPointer.h"                       // For svtkSmartPointer

template <class Implementation,
  class CellIterator = svtkMappedUnstructuredGridCellIterator<Implementation> >
class svtkMappedUnstructuredGrid : public svtkUnstructuredGridBase
{
  typedef svtkMappedUnstructuredGrid<Implementation, CellIterator> SelfType;

public:
  svtkTemplateTypeMacro(SelfType, svtkUnstructuredGridBase);
  typedef Implementation ImplementationType;
  typedef CellIterator CellIteratorType;

  // Virtuals from various base classes:
  void PrintSelf(ostream& os, svtkIndent indent) override;
  void CopyStructure(svtkDataSet* pd) override;
  void ShallowCopy(svtkDataObject* src) override;
  svtkIdType GetNumberOfCells() override;
  using svtkDataSet::GetCell;
  svtkCell* GetCell(svtkIdType cellId) override;
  void GetCell(svtkIdType cellId, svtkGenericCell* cell) override;
  int GetCellType(svtkIdType cellId) override;
  void GetCellPoints(svtkIdType cellId, svtkIdList* ptIds) override;
  svtkCellIterator* NewCellIterator() override;
  void GetPointCells(svtkIdType ptId, svtkIdList* cellIds) override;
  int GetMaxCellSize() override;
  void GetIdsOfCellsOfType(int type, svtkIdTypeArray* array) override;
  int IsHomogeneous() override;
  void Allocate(svtkIdType numCells, int extSize = 1000) override;
  svtkMTimeType GetMTime() override;

  void SetImplementation(ImplementationType* impl);
  ImplementationType* GetImplementation();

protected:
  svtkMappedUnstructuredGrid();
  ~svtkMappedUnstructuredGrid() override;

  // For convenience...
  typedef svtkMappedUnstructuredGrid<Implementation, CellIterator> ThisType;

  svtkSmartPointer<ImplementationType> Impl;

  svtkIdType InternalInsertNextCell(int type, svtkIdType npts, const svtkIdType ptIds[]) override;
  svtkIdType InternalInsertNextCell(int type, svtkIdList* ptIds) override;
  svtkIdType InternalInsertNextCell(int type, svtkIdType npts, const svtkIdType ptIds[],
    svtkIdType nfaces, const svtkIdType faces[]) override;
  void InternalReplaceCell(svtkIdType cellId, int npts, const svtkIdType pts[]) override;

private:
  svtkMappedUnstructuredGrid(const svtkMappedUnstructuredGrid&) = delete;
  void operator=(const svtkMappedUnstructuredGrid&) = delete;

  svtkNew<svtkGenericCell> TempCell;
};

#include "svtkMappedUnstructuredGrid.txx"

// We need to fake the superclass for the wrappers, otherwise they will choke on
// the template:
#ifndef __SVTK_WRAP__

#define svtkMakeExportedMappedUnstructuredGrid(_className, _impl, _exportDecl)                      \
  class _exportDecl _className : public svtkMappedUnstructuredGrid<_impl>                           \
  {                                                                                                \
  public:                                                                                          \
    svtkTypeMacro(_className, svtkMappedUnstructuredGrid<_impl>);                                    \
    static _className* New();                                                                      \
                                                                                                   \
  protected:                                                                                       \
    _className()                                                                                   \
    {                                                                                              \
      _impl* i = _impl::New();                                                                     \
      this->SetImplementation(i);                                                                  \
      i->Delete();                                                                                 \
    }                                                                                              \
    ~_className() override {}                                                                      \
                                                                                                   \
  private:                                                                                         \
    _className(const _className&);                                                                 \
    void operator=(const _className&);                                                             \
  }

#define svtkMakeExportedMappedUnstructuredGridWithIter(_className, _impl, _cIter, _exportDecl)      \
  class _exportDecl _className : public svtkMappedUnstructuredGrid<_impl, _cIter>                   \
  {                                                                                                \
  public:                                                                                          \
    svtkTypeMacro(_className, svtkMappedUnstructuredGrid<_impl, _cIter>);                            \
    static _className* New();                                                                      \
                                                                                                   \
  protected:                                                                                       \
    _className()                                                                                   \
    {                                                                                              \
      _impl* i = _impl::New();                                                                     \
      this->SetImplementation(i);                                                                  \
      i->Delete();                                                                                 \
    }                                                                                              \
    ~_className() override {}                                                                      \
                                                                                                   \
  private:                                                                                         \
    _className(const _className&);                                                                 \
    void operator=(const _className&);                                                             \
  }

#else // __SVTK_WRAP__

#define svtkMakeExportedMappedUnstructuredGrid(_className, _impl, _exportDecl)                      \
  class _exportDecl _className : public svtkUnstructuredGridBase                                    \
  {                                                                                                \
  public:                                                                                          \
    svtkTypeMacro(_className, svtkUnstructuredGridBase);                                             \
    static _className* New();                                                                      \
                                                                                                   \
  protected:                                                                                       \
    _className() {}                                                                                \
    ~_className() override {}                                                                      \
                                                                                                   \
  private:                                                                                         \
    _className(const _className&);                                                                 \
    void operator=(const _className&);                                                             \
  }

#define svtkMakeExportedMappedUnstructuredGridWithIter(_className, _impl, _cIter, _exportDecl)      \
  class _exportDecl _className : public svtkUnstructuredGridBase                                    \
  {                                                                                                \
  public:                                                                                          \
    svtkTypeMacro(_className, svtkUnstructuredGridBase);                                             \
    static _className* New();                                                                      \
                                                                                                   \
  protected:                                                                                       \
    _className() {}                                                                                \
    ~_className() override {}                                                                      \
                                                                                                   \
  private:                                                                                         \
    _className(const _className&);                                                                 \
    void operator=(const _className&);                                                             \
  }

#endif // __SVTK_WRAP__

#define svtkMakeMappedUnstructuredGrid(_className, _impl)                                           \
  svtkMakeExportedMappedUnstructuredGrid(_className, _impl, )

#define svtkMakeMappedUnstructuredGridWithIter(_className, _impl, _cIter, _exportDecl)              \
  svtkMakeExportedMappedUnstructuredGridWithIter(_className, _impl, _cIter, )

#endif // svtkMappedUnstructuredGrid_h

// SVTK-HeaderTest-Exclude: svtkMappedUnstructuredGrid.h
