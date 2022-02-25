/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnstructuredGridBase.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkUnstructuredGridBase
 * @brief   dataset represents arbitrary combinations
 * of all possible cell types. May be mapped onto a non-standard memory layout.
 *
 *
 * svtkUnstructuredGridBase defines the core svtkUnstructuredGrid API, omitting
 * functions that are implementation dependent.
 *
 * @sa
 * svtkMappedDataArray svtkUnstructuredGrid
 */

#ifndef svtkUnstructuredGridBase_h
#define svtkUnstructuredGridBase_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkPointSet.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkUnstructuredGridBase : public svtkPointSet
{
public:
  svtkAbstractTypeMacro(svtkUnstructuredGridBase, svtkPointSet);
  void PrintSelf(ostream& os, svtkIndent indent) override
  {
    this->Superclass::PrintSelf(os, indent);
  }

  int GetDataObjectType() override { return SVTK_UNSTRUCTURED_GRID_BASE; }

  /**
   * Allocate memory for the number of cells indicated. extSize is not used.
   */
  virtual void Allocate(svtkIdType numCells = 1000, int extSize = 1000) = 0;

  /**
   * Shallow and Deep copy.
   */
  void DeepCopy(svtkDataObject* src) override;

  /**
   * Insert/create cell in object by type and list of point ids defining
   * cell topology. Most cells require just a type which implicitly defines
   * a set of points and their ordering. For non-polyhedron cell type, npts
   * is the number of unique points in the cell. pts are the list of global
   * point Ids. For polyhedron cell, a special input format is required.
   * npts is the number of faces in the cell. ptIds is the list of face stream:
   * (numFace0Pts, id1, id2, id3, numFace1Pts,id1, id2, id3, ...)
   * Make sure you have called Allocate() before calling this method
   */
  svtkIdType InsertNextCell(int type, svtkIdType npts, const svtkIdType ptIds[])
    SVTK_SIZEHINT(ptIds, npts);

  /**
   * Insert/create cell in object by a list of point ids defining
   * cell topology. Most cells require just a type which implicitly defines
   * a set of points and their ordering. For non-polyhedron cell type, ptIds
   * is the list of global Ids of unique cell points. For polyhedron cell,
   * a special ptIds input format is required:
   * (numCellFaces, numFace0Pts, id1, id2, id3, numFace1Pts,id1, id2, id3, ...)
   * Make sure you have called Allocate() before calling this method
   */
  svtkIdType InsertNextCell(int type, svtkIdList* ptIds);

  // Description:
  // Insert/create a polyhedron cell. npts is the number of unique points in
  // the cell. pts is the list of the unique cell point Ids. nfaces is the
  // number of faces in the cell. faces is the face-stream
  // [numFace0Pts, id1, id2, id3, numFace1Pts,id1, id2, id3, ...].
  // All point Ids are global.
  // Make sure you have called Allocate() before calling this method
  svtkIdType InsertNextCell(int type, svtkIdType npts, const svtkIdType ptIds[], svtkIdType nfaces,
    const svtkIdType faces[]) SVTK_SIZEHINT(ptIds, npts) SVTK_SIZEHINT(faces, nfaces);

  /**
   * Replace the points defining cell "cellId" with a new set of points. This
   * operator is (typically) used when links from points to cells have not been
   * built (i.e., BuildLinks() has not been executed). Use the operator
   * ReplaceLinkedCell() to replace a cell when cell structure has been built.
   */
  void ReplaceCell(svtkIdType cellId, int npts, const svtkIdType pts[]) SVTK_SIZEHINT(pts, npts);

  /**
   * Fill svtkIdTypeArray container with list of cell Ids.  This
   * method traverses all cells and, for a particular cell type,
   * inserts the cell Id into the container.
   */
  virtual void GetIdsOfCellsOfType(int type, svtkIdTypeArray* array) = 0;

  /**
   * Traverse cells and determine if cells are all of the same type.
   */
  virtual int IsHomogeneous() = 0;

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkUnstructuredGridBase* GetData(svtkInformation* info);
  static svtkUnstructuredGridBase* GetData(svtkInformationVector* v, int i = 0);
  //@}

protected:
  svtkUnstructuredGridBase();
  ~svtkUnstructuredGridBase() override;

  virtual svtkIdType InternalInsertNextCell(int type, svtkIdList* ptIds) = 0;
  virtual svtkIdType InternalInsertNextCell(int type, svtkIdType npts, const svtkIdType ptIds[]) = 0;
  virtual svtkIdType InternalInsertNextCell(int type, svtkIdType npts, const svtkIdType ptIds[],
    svtkIdType nfaces, const svtkIdType faces[]) = 0;
  virtual void InternalReplaceCell(svtkIdType cellId, int npts, const svtkIdType pts[]) = 0;

private:
  svtkUnstructuredGridBase(const svtkUnstructuredGridBase&) = delete;
  void operator=(const svtkUnstructuredGridBase&) = delete;
};

#endif
