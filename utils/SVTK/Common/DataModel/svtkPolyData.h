/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPolyData.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPolyData
 * @brief   concrete dataset represents vertices, lines, polygons, and triangle strips
 *
 * svtkPolyData is a data object that is a concrete implementation of
 * svtkDataSet. svtkPolyData represents a geometric structure consisting of
 * vertices, lines, polygons, and/or triangle strips. Point and cell
 * attribute values (e.g., scalars, vectors, etc.) also are represented.
 *
 * The actual cell types (svtkCellType.h) supported by svtkPolyData are:
 * svtkVertex, svtkPolyVertex, svtkLine, svtkPolyLine, svtkTriangle, svtkQuad,
 * svtkPolygon, and svtkTriangleStrip.
 *
 * One important feature of svtkPolyData objects is that special traversal and
 * data manipulation methods are available to process data. These methods are
 * generally more efficient than svtkDataSet methods and should be used
 * whenever possible. For example, traversing the cells in a dataset we would
 * use GetCell(). To traverse cells with svtkPolyData we would retrieve the
 * cell array object representing polygons (for example using GetPolys()) and
 * then use svtkCellArray's InitTraversal() and GetNextCell() methods.
 *
 * @warning
 * Because svtkPolyData is implemented with four separate instances of
 * svtkCellArray to represent 0D vertices, 1D lines, 2D polygons, and 2D
 * triangle strips, it is possible to create svtkPolyData instances that
 * consist of a mixture of cell types. Because of the design of the class,
 * there are certain limitations on how mixed cell types are inserted into
 * the svtkPolyData, and in turn the order in which they are processed and
 * rendered. To preserve the consistency of cell ids, and to insure that
 * cells with cell data are rendered properly, users must insert mixed cells
 * in the order of vertices (svtkVertex and svtkPolyVertex), lines (svtkLine and
 * svtkPolyLine), polygons (svtkTriangle, svtkQuad, svtkPolygon), and triangle
 * strips (svtkTriangleStrip).
 *
 * @warning
 * Some filters when processing svtkPolyData with mixed cell types may process
 * the cells in differing ways. Some will convert one type into another
 * (e.g., svtkTriangleStrip into svtkTriangles) or expect a certain type
 * (svtkDecimatePro expects triangles or triangle strips; svtkTubeFilter
 * expects lines). Read the documentation for each filter carefully to
 * understand how each part of svtkPolyData is processed.
 *
 * @warning
 * Some of the methods specified here function properly only when the dataset
 * has been specified as "Editable". They are documented as such.
 */

#ifndef svtkPolyData_h
#define svtkPolyData_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkPointSet.h"

#include "svtkCellArray.h"         // Needed for inline methods
#include "svtkCellLinks.h"         // Needed for inline methods
#include "svtkPolyDataInternals.h" // Needed for inline methods

class svtkVertex;
class svtkPolyVertex;
class svtkLine;
class svtkPolyLine;
class svtkTriangle;
class svtkQuad;
class svtkPolygon;
class svtkTriangleStrip;
class svtkEmptyCell;
struct svtkPolyDataDummyContainter;
class svtkIncrementalPointLocator;

class SVTKCOMMONDATAMODEL_EXPORT svtkPolyData : public svtkPointSet
{
public:
  static svtkPolyData* New();

  svtkTypeMacro(svtkPolyData, svtkPointSet);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return what type of dataset this is.
   */
  int GetDataObjectType() override { return SVTK_POLY_DATA; }

  /**
   * Copy the geometric and topological structure of an input poly data object.
   */
  void CopyStructure(svtkDataSet* ds) override;

  //@{
  /**
   * Standard svtkDataSet interface.
   */
  svtkIdType GetNumberOfCells() override;
  using svtkDataSet::GetCell;
  svtkCell* GetCell(svtkIdType cellId) override;
  void GetCell(svtkIdType cellId, svtkGenericCell* cell) override;
  int GetCellType(svtkIdType cellId) override;
  void GetCellBounds(svtkIdType cellId, double bounds[6]) override;
  void GetCellNeighbors(svtkIdType cellId, svtkIdList* ptIds, svtkIdList* cellIds) override;
  //@}

  /**
   * Copy cells listed in idList from pd, including points, point data,
   * and cell data.  This method assumes that point and cell data have
   * been allocated.  If you pass in a point locator, then the points
   * won't be duplicated in the output. This requires the use of an
   * incremental point locator.
   */
  void CopyCells(svtkPolyData* pd, svtkIdList* idList, svtkIncrementalPointLocator* locator = nullptr);

  /**
   * Copy a cells point ids into list provided. (Less efficient.)
   */
  void GetCellPoints(svtkIdType cellId, svtkIdList* ptIds) override;

  /**
   * Efficient method to obtain cells using a particular point. Make sure that
   * routine BuildLinks() has been called.
   */
  void GetPointCells(svtkIdType ptId, svtkIdList* cellIds) override;

  /**
   * Compute the (X, Y, Z)  bounds of the data. Note that the method only considers
   * points that are used by cells (unless there are no cells, in which case all
   * points are considered). This is done for usability and historical reasons.
   */
  void ComputeBounds() override;

  /**
   * Recover extra allocated memory when creating data whose initial size
   * is unknown. Examples include using the InsertNextCell() method, or
   * when using the CellArray::EstimateSize() method to create vertices,
   * lines, polygons, or triangle strips.
   */
  void Squeeze() override;

  /**
   * Return the maximum cell size in this poly data.
   */
  int GetMaxCellSize() override;

  /**
   * Set the cell array defining vertices.
   */
  void SetVerts(svtkCellArray* v);

  /**
   * Get the cell array defining vertices. If there are no vertices, an
   * empty array will be returned (convenience to simplify traversal).
   */
  svtkCellArray* GetVerts();

  /**
   * Set the cell array defining lines.
   */
  void SetLines(svtkCellArray* l);

  /**
   * Get the cell array defining lines. If there are no lines, an
   * empty array will be returned (convenience to simplify traversal).
   */
  svtkCellArray* GetLines();

  /**
   * Set the cell array defining polygons.
   */
  void SetPolys(svtkCellArray* p);

  /**
   * Get the cell array defining polygons. If there are no polygons, an
   * empty array will be returned (convenience to simplify traversal).
   */
  svtkCellArray* GetPolys();

  /**
   * Set the cell array defining triangle strips.
   */
  void SetStrips(svtkCellArray* s);

  /**
   * Get the cell array defining triangle strips. If there are no
   * triangle strips, an empty array will be returned (convenience to
   * simplify traversal).
   */
  svtkCellArray* GetStrips();

  //@{
  /**
   * Return the number of primitives of a particular type held.
   */
  svtkIdType GetNumberOfVerts() { return (this->Verts ? this->Verts->GetNumberOfCells() : 0); }
  svtkIdType GetNumberOfLines() { return (this->Lines ? this->Lines->GetNumberOfCells() : 0); }
  svtkIdType GetNumberOfPolys() { return (this->Polys ? this->Polys->GetNumberOfCells() : 0); }
  svtkIdType GetNumberOfStrips() { return (this->Strips ? this->Strips->GetNumberOfCells() : 0); }
  //@}

  /**
   * Preallocate memory for the internal cell arrays. Each of the internal
   * cell arrays (verts, lines, polys, and strips) will be resized to hold
   * @a numCells cells of size @a maxCellSize.
   *
   * Existing data is not preserved and the number of cells is set to zero.
   *
   * @return True if allocation succeeds.
   */
  bool AllocateEstimate(svtkIdType numCells, svtkIdType maxCellSize);

  /**
   * Preallocate memory for the internal cell arrays. Each of the internal
   * cell arrays (verts, lines, polys, and strips) will be resized to hold
   * the indicated number of cells of the specified cell size.
   *
   * Existing data is not preserved and the number of cells is set to zero.
   *
   * @return True if allocation succeeds.
   */
  bool AllocateEstimate(svtkIdType numVerts, svtkIdType maxVertSize, svtkIdType numLines,
    svtkIdType maxLineSize, svtkIdType numPolys, svtkIdType maxPolySize, svtkIdType numStrips,
    svtkIdType maxStripSize);

  /**
   * Preallocate memory for the internal cell arrays. Each of the internal
   * cell arrays (verts, lines, polys, and strips) will be resized to hold
   * @a numCells cells and @a connectivitySize pointIds.
   *
   * Existing data is not preserved and the number of cells is set to zero.
   *
   * @return True if allocation succeeds.
   */
  bool AllocateExact(svtkIdType numCells, svtkIdType connectivitySize);

  /**
   * Preallocate memory for the internal cell arrays. Each of the internal
   * cell arrays (verts, lines, polys, and strips) will be resized to hold
   * the indicated number of cells and the specified number of point ids
   * (ConnSize).
   *
   * Existing data is not preserved and the number of cells is set to zero.
   *
   * @return True if allocation succeeds.
   */
  bool AllocateExact(svtkIdType numVerts, svtkIdType vertConnSize, svtkIdType numLines,
    svtkIdType lineConnSize, svtkIdType numPolys, svtkIdType polyConnSize, svtkIdType numStrips,
    svtkIdType stripConnSize);

  /**
   * Preallocate memory for the internal cell arrays such that they are the
   * same size as those in @a pd.
   *
   * Existing data is not preserved and the number of cells is set to zero.
   *
   * @return True if allocation succeeds.
   */
  bool AllocateCopy(svtkPolyData* pd);

  /**
   * Preallocate memory for the internal cell arrays such that they are
   * proportional to those in @a pd by a factor of @a ratio (for instance,
   * @a ratio = 2 allocates twice as many cells).
   *
   * Existing data is not preserved and the number of cells is set to zero.
   *
   * @return True if allocation succeeds.
   */
  bool AllocateProportional(svtkPolyData* pd, double ratio);

  /**
   * Method allocates initial storage for vertex, line, polygon, and
   * triangle strip arrays. Use this method before the method
   * PolyData::InsertNextCell(). (Or, provide vertex, line, polygon, and
   * triangle strip cell arrays). @a extSize is no longer used.
   */
  void Allocate(svtkIdType numCells = 1000, int svtkNotUsed(extSize) = 1000)
  {
    this->AllocateExact(numCells, numCells);
  }

  /**
   * Similar to the method above, this method allocates initial storage for
   * vertex, line, polygon, and triangle strip arrays. It does this more
   * intelligently, examining the supplied inPolyData to determine whether to
   * allocate the verts, lines, polys, and strips arrays.  (These arrays are
   * allocated only if there is data in the corresponding arrays in the
   * inPolyData.)  Caution: if the inPolyData has no verts, and after
   * allocating with this method an PolyData::InsertNextCell() is invoked
   * where a vertex is inserted, bad things will happen.
   */
  void Allocate(svtkPolyData* inPolyData, svtkIdType numCells = 1000, int svtkNotUsed(extSize) = 1000)
  {
    this->AllocateProportional(
      inPolyData, static_cast<double>(numCells) / inPolyData->GetNumberOfCells());
  }

  /**
   * Insert a cell of type SVTK_VERTEX, SVTK_POLY_VERTEX, SVTK_LINE, SVTK_POLY_LINE,
   * SVTK_TRIANGLE, SVTK_QUAD, SVTK_POLYGON, or SVTK_TRIANGLE_STRIP.  Make sure that
   * the PolyData::Allocate() function has been called first or that vertex,
   * line, polygon, and triangle strip arrays have been supplied.
   * Note: will also insert SVTK_PIXEL, but converts it to SVTK_QUAD.
   */
  svtkIdType InsertNextCell(int type, int npts, const svtkIdType pts[]) SVTK_SIZEHINT(pts, npts);

  /**
   * Insert a cell of type SVTK_VERTEX, SVTK_POLY_VERTEX, SVTK_LINE, SVTK_POLY_LINE,
   * SVTK_TRIANGLE, SVTK_QUAD, SVTK_POLYGON, or SVTK_TRIANGLE_STRIP.  Make sure that
   * the PolyData::Allocate() function has been called first or that vertex,
   * line, polygon, and triangle strip arrays have been supplied.
   * Note: will also insert SVTK_PIXEL, but converts it to SVTK_QUAD.
   */
  svtkIdType InsertNextCell(int type, svtkIdList* pts);

  /**
   * Begin inserting data all over again. Memory is not freed but otherwise
   * objects are returned to their initial state.
   */
  void Reset();

  /**
   * Create data structure that allows random access of cells. BuildCells is
   * expensive but necessary to make use of the faster non-virtual implementations
   * of GetCell/GetCellPoints. One may check if cells need to be built via
   * NeedToBuilds before invoking. Cells always need to be built/re-built after
   * low level direct modifications to verts, lines, polys or strips cell arrays.
   */
  void BuildCells();

  /**
   * Check if BuildCells is needed.
   */
  bool NeedToBuildCells() { return this->Cells == nullptr; }

  /**
   * Create upward links from points to cells that use each point. Enables
   * topologically complex queries. Normally the links array is allocated
   * based on the number of points in the svtkPolyData. The optional
   * initialSize parameter can be used to allocate a larger size initially.
   */
  void BuildLinks(int initialSize = 0);

  /**
   * Release data structure that allows random access of the cells. This must
   * be done before a 2nd call to BuildLinks(). DeleteCells implicitly deletes
   * the links as well since they are no longer valid.
   */
  void DeleteCells();

  /**
   * Release the upward links from point to cells that use each point.
   */
  void DeleteLinks();

  //@{
  /**
   * Special (efficient) operations on poly data. Use carefully (i.e., make
   * sure that BuildLinks() has been called).
   */
  void GetPointCells(svtkIdType ptId, svtkIdType& ncells, svtkIdType*& cells)
    SVTK_SIZEHINT(cells, ncells);
#ifndef SVTK_LEGACY_REMOVE
  SVTK_LEGACY(void GetPointCells(svtkIdType ptId, unsigned short& ncells, svtkIdType*& cells))
  SVTK_SIZEHINT(cells, ncells);
#endif
  //@}

  /**
   * Get the neighbors at an edge. More efficient than the general
   * GetCellNeighbors(). Assumes links have been built (with BuildLinks()),
   * and looks specifically for edge neighbors.
   */
  void GetCellEdgeNeighbors(svtkIdType cellId, svtkIdType p1, svtkIdType p2, svtkIdList* cellIds);

  /**
   * Get a list of point ids that define a cell. The cell type is
   * returned. Requires the the cells have been built with BuildCells.
   *
   * @warning Subsequent calls to this method may invalidate previous call
   * results.
   *
   * The @a pts pointer must not be modified.
   */
  unsigned char GetCellPoints(svtkIdType cellId, svtkIdType& npts, svtkIdType const*& pts)
    SVTK_SIZEHINT(pts, npts);

  /**
   * Given three vertices, determine whether it's a triangle. Make sure
   * BuildLinks() has been called first.
   */
  int IsTriangle(int v1, int v2, int v3);

  /**
   * Determine whether two points form an edge. If they do, return non-zero.
   * By definition PolyVertex and PolyLine have no edges since 1-dimensional
   * edges are only found on cells 2D and higher.
   * Edges are defined as 1-D boundary entities to cells.
   * Make sure BuildLinks() has been called first.
   */
  int IsEdge(svtkIdType p1, svtkIdType p2);

  /**
   * Determine whether a point is used by a particular cell. If it is, return
   * non-zero. Make sure BuildCells() has been called first.
   */
  int IsPointUsedByCell(svtkIdType ptId, svtkIdType cellId);

  /**
   * Replace the points defining cell "cellId" with a new set of points. This
   * operator is (typically) used when links from points to cells have not been
   * built (i.e., BuildLinks() has not been executed). Use the operator
   * ReplaceLinkedCell() to replace a cell when cell structure has been built. Use this
   * method only when the dataset is set as Editable.
   * @{
   */
  void ReplaceCell(svtkIdType cellId, svtkIdList* ids);
  void ReplaceCell(svtkIdType cellId, int npts, const svtkIdType pts[]) SVTK_SIZEHINT(pts, npts);
  /**@}*/

  /**
   * Replace a point in the cell connectivity list with a different point. Use this
   * method only when the dataset is set as Editable.
   */
  void ReplaceCellPoint(svtkIdType cellId, svtkIdType oldPtId, svtkIdType newPtId);

  /**
   * Reverse the order of point ids defining the cell. Use this
   * method only when the dataset is set as Editable.
   */
  void ReverseCell(svtkIdType cellId);

  //@{
  /**
   * Mark a point/cell as deleted from this svtkPolyData. Use this
   * method only when the dataset is set as Editable.
   */
  void DeletePoint(svtkIdType ptId);
  void DeleteCell(svtkIdType cellId);
  //@}

  /**
   * The cells marked by calls to DeleteCell are stored in the Cell Array
   * SVTK_EMPTY_CELL, but they still exist in the cell arrays.  Calling
   * RemoveDeletedCells will traverse the cell arrays and remove/compact the
   * cell arrays as well as any cell data thus truly removing the cells from
   * the polydata object. Use this method only when the dataset is set as
   * Editable.
   */
  void RemoveDeletedCells();

  //@{
  /**
   * Add a point to the cell data structure (after cell pointers have been
   * built). This method adds the point and then allocates memory for the
   * links to the cells.  (To use this method, make sure points are available
   * and BuildLinks() has been invoked.) Of the two methods below, one inserts
   * a point coordinate and the other just makes room for cell links. Use this
   * method only when the dataset is set as Editable.
   */
  svtkIdType InsertNextLinkedPoint(int numLinks);
  svtkIdType InsertNextLinkedPoint(double x[3], int numLinks);
  //@}

  /**
   * Add a new cell to the cell data structure (after cell pointers have been
   * built). This method adds the cell and then updates the links from the
   * points to the cells. (Memory is allocated as necessary.) Use this method
   * only when the dataset is set as Editable.
   */
  svtkIdType InsertNextLinkedCell(int type, int npts, const svtkIdType pts[]) SVTK_SIZEHINT(pts, npts);

  /**
   * Replace one cell with another in cell structure. This operator updates
   * the connectivity list and the point's link list. It does not delete
   * references to the old cell in the point's link list. Use the operator
   * RemoveCellReference() to delete all references from points to (old)
   * cell.  You may also want to consider using the operator ResizeCellList()
   * if the link list is changing size. Use this method only when the dataset
   * is set as Editable.
   */
  void ReplaceLinkedCell(svtkIdType cellId, int npts, const svtkIdType pts[]) SVTK_SIZEHINT(pts, npts);

  /**
   * Remove all references to cell in cell structure. This means the links
   * from the cell's points to the cell are deleted. Memory is not
   * reclaimed. Use the method ResizeCellList() to resize the link list from
   * a point to its using cells. (This operator assumes BuildLinks() has been
   * called.) Use this method only when the dataset is set as Editable.
   */
  void RemoveCellReference(svtkIdType cellId);

  /**
   * Add references to cell in cell structure. This means the links from
   * the cell's points to the cell are modified. Memory is not extended. Use the
   * method ResizeCellList() to resize the link list from a point to its using
   * cells. (This operator assumes BuildLinks() has been called.) Use this
   * method only when the dataset is set as Editable.
   */
  void AddCellReference(svtkIdType cellId);

  /**
   * Remove a reference to a cell in a particular point's link list. You may
   * also consider using RemoveCellReference() to remove the references from
   * all the cell's points to the cell. This operator does not reallocate
   * memory; use the operator ResizeCellList() to do this if necessary. Use
   * this method only when the dataset is set as Editable.
   */
  void RemoveReferenceToCell(svtkIdType ptId, svtkIdType cellId);

  /**
   * Add a reference to a cell in a particular point's link list. (You may also
   * consider using AddCellReference() to add the references from all the
   * cell's points to the cell.) This operator does not realloc memory; use the
   * operator ResizeCellList() to do this if necessary. Use this
   * method only when the dataset is set as Editable.
   */
  void AddReferenceToCell(svtkIdType ptId, svtkIdType cellId);

  /**
   * Resize the list of cells using a particular point. (This operator
   * assumes that BuildLinks() has been called.) Use this method only when
   * the dataset is set as Editable.
   */
  void ResizeCellList(svtkIdType ptId, int size);

  /**
   * Restore object to initial state. Release memory back to system.
   */
  void Initialize() override;

  //@{
  /**
   * Get the piece and the number of pieces. Similar to extent in 3D.
   */
  virtual int GetPiece();
  virtual int GetNumberOfPieces();
  //@}

  /**
   * Get the ghost level.
   */
  virtual int GetGhostLevel();

  /**
   * Return the actual size of the data in kibibytes (1024 bytes). This number
   * is valid only after the pipeline has updated. The memory size
   * returned is guaranteed to be greater than or equal to the
   * memory required to represent the data (e.g., extra space in
   * arrays, etc. are not included in the return value). THIS METHOD
   * IS THREAD SAFE.
   */
  unsigned long GetActualMemorySize() override;

  //@{
  /**
   * Shallow and Deep copy.
   */
  void ShallowCopy(svtkDataObject* src) override;
  void DeepCopy(svtkDataObject* src) override;
  //@}

  /**
   * This method will remove any cell that is marked as ghost
   * (has the svtkDataSetAttributes::DUPLICATECELL bit set).
   * It does not remove unused points.
   */
  void RemoveGhostCells();

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkPolyData* GetData(svtkInformation* info);
  static svtkPolyData* GetData(svtkInformationVector* v, int i = 0);
  //@}

  /**
   * Scalar field critical point classification (for manifold 2D meshes).
   * Reference: J. Milnor "Morse Theory", Princeton University Press, 1963.

   * Given a pointId and an attribute representing a scalar field, this member
   * returns the index of the critical point:
   * svtkPolyData::MINIMUM (index 0): local minimum;
   * svtkPolyData::SADDLE  (index 1): local saddle;
   * svtkPolyData::MAXIMUM (index 2): local maximum.

   * Other returned values are:
   * svtkPolyData::REGULAR_POINT: regular point (the gradient does not vanish);
   * svtkPolyData::ERR_NON_MANIFOLD_STAR: the star of the considered vertex is
   * not manifold (could not evaluate the index)
   * svtkPolyData::ERR_INCORRECT_FIELD: the number of entries in the scalar field
   * array is different form the number of vertices in the mesh.
   * svtkPolyData::ERR_NO_SUCH_FIELD: the specified scalar field does not exist.
   */
  enum
  {
    ERR_NO_SUCH_FIELD = -4,
    ERR_INCORRECT_FIELD = -3,
    ERR_NON_MANIFOLD_STAR = -2,
    REGULAR_POINT = -1,
    MINIMUM = 0,
    SADDLE = 1,
    MAXIMUM = 2
  };

  int GetScalarFieldCriticalIndex(svtkIdType pointId, svtkDataArray* scalarField);
  int GetScalarFieldCriticalIndex(svtkIdType pointId, int fieldId);
  int GetScalarFieldCriticalIndex(svtkIdType pointId, const char* fieldName);

  /**
   * Return the mesh (geometry/topology) modification time.
   * This time is different from the usual MTime which also takes into
   * account the modification of data arrays. This function can be used to
   * track the changes on the mesh separately from the data arrays
   * (eg. static mesh over time with transient data).
   */
  virtual svtkMTimeType GetMeshMTime();

  /**
   * Get MTime which also considers its cell array MTime.
   */
  svtkMTimeType GetMTime() override;

  /**
   * Get a pointer to the cell, ie [npts pid1 .. pidn]. The cell type is
   * returned. Requires the the cells have been built with BuildCells.
   * The @a pts pointer must not be modified.
   *
   * @warning Internal cell storage has changed, and cell size is no longer
   * stored with the cell point ids. The `pts` array returned here no longer
   * exists in memory.
   */
  unsigned char GetCell(svtkIdType cellId, const svtkIdType*& pts);

protected:
  svtkPolyData();
  ~svtkPolyData() override;

  using TaggedCellId = svtkPolyData_detail::TaggedCellId;
  using CellMap = svtkPolyData_detail::CellMap;

  svtkCellArray* GetCellArrayInternal(TaggedCellId tag);

  // constant cell objects returned by GetCell called.
  svtkSmartPointer<svtkVertex> Vertex;
  svtkSmartPointer<svtkPolyVertex> PolyVertex;
  svtkSmartPointer<svtkLine> Line;
  svtkSmartPointer<svtkPolyLine> PolyLine;
  svtkSmartPointer<svtkTriangle> Triangle;
  svtkSmartPointer<svtkQuad> Quad;
  svtkSmartPointer<svtkPolygon> Polygon;
  svtkSmartPointer<svtkTriangleStrip> TriangleStrip;
  svtkSmartPointer<svtkEmptyCell> EmptyCell;

  // points inherited
  // point data (i.e., scalars, vectors, normals, tcoords) inherited
  svtkSmartPointer<svtkCellArray> Verts;
  svtkSmartPointer<svtkCellArray> Lines;
  svtkSmartPointer<svtkCellArray> Polys;
  svtkSmartPointer<svtkCellArray> Strips;

  // supporting structures for more complex topological operations
  // built only when necessary
  svtkSmartPointer<CellMap> Cells;
  svtkSmartPointer<svtkCellLinks> Links;

  svtkNew<svtkIdList> LegacyBuffer;

  // dummy static member below used as a trick to simplify traversal
  static svtkPolyDataDummyContainter DummyContainer;

private:
  // Hide these from the user and the compiler.

  /**
   * For legacy compatibility. Do not use.
   */
  void GetCellNeighbors(svtkIdType cellId, svtkIdList& ptIds, svtkIdList& cellIds)
  {
    this->GetCellNeighbors(cellId, &ptIds, &cellIds);
  }

  void Cleanup();

private:
  svtkPolyData(const svtkPolyData&) = delete;
  void operator=(const svtkPolyData&) = delete;
};

//------------------------------------------------------------------------------
inline void svtkPolyData::GetPointCells(svtkIdType ptId, svtkIdType& ncells, svtkIdType*& cells)
{
  ncells = this->Links->GetNcells(ptId);
  cells = this->Links->GetCells(ptId);
}

#ifndef SVTK_LEGACY_REMOVE
inline void svtkPolyData::GetPointCells(svtkIdType ptId, unsigned short& ncells, svtkIdType*& cells)
{
  SVTK_LEGACY_BODY(svtkPolyData::GetPointCells, "SVTK 9.0");
  ncells = static_cast<unsigned short>(this->Links->GetNcells(ptId));
  cells = this->Links->GetCells(ptId);
}
#endif

//------------------------------------------------------------------------------
inline svtkIdType svtkPolyData::GetNumberOfCells()
{
  return (this->GetNumberOfVerts() + this->GetNumberOfLines() + this->GetNumberOfPolys() +
    this->GetNumberOfStrips());
}

//------------------------------------------------------------------------------
inline int svtkPolyData::GetCellType(svtkIdType cellId)
{
  if (!this->Cells)
  {
    this->BuildCells();
  }
  return static_cast<int>(this->Cells->GetTag(cellId).GetCellType());
}

//------------------------------------------------------------------------------
inline int svtkPolyData::IsTriangle(int v1, int v2, int v3)
{
  svtkIdType n1;
  int i, j, tVerts[3];
  svtkIdType* cells;
  const svtkIdType* tVerts2;
  svtkIdType n2;

  tVerts[0] = v1;
  tVerts[1] = v2;
  tVerts[2] = v3;

  for (i = 0; i < 3; i++)
  {
    this->GetPointCells(tVerts[i], n1, cells);
    for (j = 0; j < n1; j++)
    {
      this->GetCellPoints(cells[j], n2, tVerts2);
      if ((tVerts[0] == tVerts2[0] || tVerts[0] == tVerts2[1] || tVerts[0] == tVerts2[2]) &&
        (tVerts[1] == tVerts2[0] || tVerts[1] == tVerts2[1] || tVerts[1] == tVerts2[2]) &&
        (tVerts[2] == tVerts2[0] || tVerts[2] == tVerts2[1] || tVerts[2] == tVerts2[2]))
      {
        return 1;
      }
    }
  }
  return 0;
}

//------------------------------------------------------------------------------
inline int svtkPolyData::IsPointUsedByCell(svtkIdType ptId, svtkIdType cellId)
{
  svtkIdType npts;
  const svtkIdType* pts;

  this->GetCellPoints(cellId, npts, pts);
  for (svtkIdType i = 0; i < npts; i++)
  {
    if (pts[i] == ptId)
    {
      return 1;
    }
  }

  return 0;
}

//------------------------------------------------------------------------------
inline void svtkPolyData::DeletePoint(svtkIdType ptId)
{
  this->Links->DeletePoint(ptId);
}

//------------------------------------------------------------------------------
inline void svtkPolyData::DeleteCell(svtkIdType cellId)
{
  this->Cells->GetTag(cellId).MarkDeleted();
}

//------------------------------------------------------------------------------
inline void svtkPolyData::RemoveCellReference(svtkIdType cellId)
{
  const svtkIdType* pts;
  svtkIdType npts;

  this->GetCellPoints(cellId, npts, pts);
  for (svtkIdType i = 0; i < npts; i++)
  {
    this->Links->RemoveCellReference(cellId, pts[i]);
  }
}

//------------------------------------------------------------------------------
inline void svtkPolyData::AddCellReference(svtkIdType cellId)
{
  const svtkIdType* pts;
  svtkIdType npts;

  this->GetCellPoints(cellId, npts, pts);
  for (svtkIdType i = 0; i < npts; i++)
  {
    this->Links->AddCellReference(cellId, pts[i]);
  }
}

//------------------------------------------------------------------------------
inline void svtkPolyData::ResizeCellList(svtkIdType ptId, int size)
{
  this->Links->ResizeCellList(ptId, size);
}

//------------------------------------------------------------------------------
inline svtkCellArray* svtkPolyData::GetCellArrayInternal(svtkPolyData::TaggedCellId tag)
{
  switch (tag.GetTarget())
  {
    case svtkPolyData_detail::Target::Verts:
      return this->Verts;
    case svtkPolyData_detail::Target::Lines:
      return this->Lines;
    case svtkPolyData_detail::Target::Polys:
      return this->Polys;
    case svtkPolyData_detail::Target::Strips:
      return this->Strips;
  }
  return nullptr; // unreachable
}

//------------------------------------------------------------------------------
inline void svtkPolyData::ReplaceCellPoint(svtkIdType cellId, svtkIdType oldPtId, svtkIdType newPtId)
{
  svtkNew<svtkIdList> ids;
  this->GetCellPoints(cellId, ids);
  for (svtkIdType i = 0; i < ids->GetNumberOfIds(); i++)
  {
    if (ids->GetId(i) == oldPtId)
    {
      ids->SetId(i, newPtId);
      break;
    }
  }
  this->ReplaceCell(cellId, static_cast<int>(ids->GetNumberOfIds()), ids->GetPointer(0));
}

//------------------------------------------------------------------------------
inline unsigned char svtkPolyData::GetCellPoints(
  svtkIdType cellId, svtkIdType& npts, svtkIdType const*& pts)
{
  if (!this->Cells)
  {
    this->BuildCells();
  }

  const TaggedCellId tag = this->Cells->GetTag(cellId);
  if (tag.IsDeleted())
  {
    npts = 0;
    pts = nullptr;
    return SVTK_EMPTY_CELL;
  }

  svtkCellArray* cells = this->GetCellArrayInternal(tag);
  cells->GetCellAtId(tag.GetCellId(), npts, pts);
  return tag.GetCellType();
}

#endif
