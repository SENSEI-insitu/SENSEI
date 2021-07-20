/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnstructuredGrid.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkUnstructuredGrid
 * @brief   dataset represents arbitrary combinations of
 * all possible cell types
 *
 * svtkUnstructuredGrid is a data object that is a concrete implementation of
 * svtkDataSet. svtkUnstructuredGrid represents any combinations of any cell
 * types. This includes 0D (e.g., points), 1D (e.g., lines, polylines), 2D
 * (e.g., triangles, polygons), and 3D (e.g., hexahedron, tetrahedron,
 * polyhedron, etc.). svtkUnstructuredGrid provides random access to cells, as
 * well as topological information (such as lists of cells using each point).
 */

#ifndef svtkUnstructuredGrid_h
#define svtkUnstructuredGrid_h

#include "svtkCellArray.h"             //inline GetCellPoints()
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkIdTypeArray.h"           //inline GetCellPoints()
#include "svtkUnstructuredGridBase.h"

#include "svtkSmartPointer.h" // for smart pointer

class svtkCellArray;
class svtkAbstractCellLinks;
class svtkBezierCurve;
class svtkBezierQuadrilateral;
class svtkBezierHexahedron;
class svtkBezierTriangle;
class svtkBezierTetra;
class svtkBezierWedge;
class svtkConvexPointSet;
class svtkEmptyCell;
class svtkHexahedron;
class svtkIdList;
class svtkIdTypeArray;
class svtkLagrangeCurve;
class svtkLagrangeQuadrilateral;
class svtkLagrangeHexahedron;
class svtkLagrangeTriangle;
class svtkLagrangeTetra;
class svtkLagrangeWedge;
class svtkLine;
class svtkPixel;
class svtkPolyLine;
class svtkPolyVertex;
class svtkPolygon;
class svtkPyramid;
class svtkPentagonalPrism;
class svtkHexagonalPrism;
class svtkQuad;
class svtkQuadraticEdge;
class svtkQuadraticHexahedron;
class svtkQuadraticWedge;
class svtkQuadraticPolygon;
class svtkQuadraticPyramid;
class svtkQuadraticQuad;
class svtkQuadraticTetra;
class svtkQuadraticTriangle;
class svtkTetra;
class svtkTriangle;
class svtkTriangleStrip;
class svtkUnsignedCharArray;
class svtkVertex;
class svtkVoxel;
class svtkWedge;
class svtkTriQuadraticHexahedron;
class svtkQuadraticLinearWedge;
class svtkQuadraticLinearQuad;
class svtkBiQuadraticQuad;
class svtkBiQuadraticQuadraticWedge;
class svtkBiQuadraticQuadraticHexahedron;
class svtkBiQuadraticTriangle;
class svtkCubicLine;
class svtkPolyhedron;
class svtkIdTypeArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkUnstructuredGrid : public svtkUnstructuredGridBase
{
public:
  /**
   * Standard instantiation method.
   */
  static svtkUnstructuredGrid* New();

  //@{
  /**
   * Standard methods for type information and printing.
   */
  svtkTypeMacro(svtkUnstructuredGrid, svtkUnstructuredGridBase);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Standard svtkDataSet API methods. See svtkDataSet for more information.
   */
  int GetDataObjectType() override { return SVTK_UNSTRUCTURED_GRID; }

  /**
   * @brief Pre-allocate memory in internal data structures. Does not change
   * the number of cells, only the array capacities. Existing data is NOT
   * preserved.
   * @param numCells The number of expected cells in the dataset.
   * @param maxCellSize The number of points per cell to allocate memory for.
   * @return True if allocation succeeds.
   * @sa Squeeze();
   */
  bool AllocateEstimate(svtkIdType numCells, svtkIdType maxCellSize)
  {
    return this->AllocateExact(numCells, numCells * maxCellSize);
  }

  /**
   * @brief Pre-allocate memory in internal data structures. Does not change
   * the number of cells, only the array capacities. Existing data is NOT
   * preserved.
   * @param numCells The number of expected cells in the dataset.
   * @param connectivitySize The total number of pointIds stored for all cells.
   * @return True if allocation succeeds.
   * @sa Squeeze();
   */
  bool AllocateExact(svtkIdType numCells, svtkIdType connectivitySize);

  /**
   * Method allocates initial storage for the cell connectivity. Use this
   * method before the method InsertNextCell(). The array capacity is
   * doubled when the inserting a cell exceeds the current capacity.
   * extSize is no longer used.
   *
   * @note Prefer AllocateExact or AllocateEstimate, which give more control
   * over how allocations are distributed.
   */
  void Allocate(svtkIdType numCells = 1000, int svtkNotUsed(extSize) = 1000) override
  {
    this->AllocateExact(numCells, numCells);
  }

  //@{
  /**
   * Standard svtkDataSet methods; see svtkDataSet.h for documentation.
   */
  void Reset();
  void CopyStructure(svtkDataSet* ds) override;
  svtkIdType GetNumberOfCells() override;

  svtkCell* GetCell(svtkIdType cellId) override;
  void GetCell(svtkIdType cellId, svtkGenericCell* cell) override;
  svtkCell* GetCell(int i, int j, int k) override;

  void GetCellBounds(svtkIdType cellId, double bounds[6]) override;
  void GetCellPoints(svtkIdType cellId, svtkIdList* ptIds) override;
  void GetPointCells(svtkIdType ptId, svtkIdList* cellIds) override;
  svtkCellIterator* NewCellIterator() override;
  //@}

  /**
   * Get the type of the cell with the given cellId.
   */
  int GetCellType(svtkIdType cellId) override;

  /**
   * Get a list of types of cells in a dataset. The list consists of an array
   * of types (not necessarily in any order), with a single entry per type.
   * For example a dataset with 5 triangles, 3 lines, and 100 hexahedra would
   * result in a list of three entries, corresponding to the types SVTK_TRIANGLE,
   * SVTK_LINE, and SVTK_HEXAHEDRON. This override implements an optimization that
   * recomputes cell types only when the types of cells may have changed.
   *
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  void GetCellTypes(svtkCellTypes* types) override;

  /**
   * A higher-performing variant of the virtual svtkDataSet::GetCellPoints()
   * for unstructured grids. Given a cellId, return the number of defining
   * points and the list of points defining the cell.
   *
   * @warning Subsequent calls to this method may invalidate previous call
   * results.
   *
   * The @a pts pointer must not be modified.
   */
  void GetCellPoints(svtkIdType cellId, svtkIdType& npts, svtkIdType const*& pts)
  {
    this->Connectivity->GetCellAtId(cellId, npts, pts);
  }

  //@{
  /**
   * Special (efficient) operation to return the list of cells using the
   * specified point ptId. Use carefully (i.e., make sure that BuildLinks()
   * has been called).
   */
  void GetPointCells(svtkIdType ptId, svtkIdType& ncells, svtkIdType*& cells)
    SVTK_SIZEHINT(cells, ncells);

#if !defined(SWIG)
#ifndef SVTK_LEGACY_REMOVE
  SVTK_LEGACY(void GetPointCells(svtkIdType ptId, unsigned short& ncells, svtkIdType*& cells))
  SVTK_SIZEHINT(cells, ncells);
#endif
#endif
  //@}

  /**
   * Get the array of all cell types in the grid. Each single-component
   * tuple in the array at an index that corresponds to the type of the cell
   * with the same index. To get an array of only the distinct cell types in
   * the dataset, use GetCellTypes().
   */
  svtkUnsignedCharArray* GetCellTypesArray();

  /**
   * Squeeze all arrays in the grid to conserve memory.
   */
  void Squeeze() override;

  /**
   * Reset the grid to an empty state and free any memory.
   */
  void Initialize() override;

  /**
   * Get the size, in number of points, of the largest cell.
   */
  int GetMaxCellSize() override;

  /**
   * Build topological links from points to lists of cells that use each point.
   * See svtkAbstractCellLinks for more information.
   */
  void BuildLinks();

  /**
   * Get the cell links. The cell links will be one of nullptr=0;
   * svtkCellLinks=1; svtkStaticCellLinksTemplate<SVTK_UNSIGNED_SHORT>=2;
   * svtkStaticCellLinksTemplate<SVTK_UNSIGNED_INT>=3;
   * svtkStaticCellLinksTemplate<SVTK_ID_TYPE>=4.  (See enum types defined in
   * svtkAbstractCellLinks.)
   */
  svtkAbstractCellLinks* GetCellLinks();

  /**
   * Get the face stream of a polyhedron cell in the following format:
   * (numCellFaces, numFace0Pts, id1, id2, id3, numFace1Pts,id1, id2, id3, ...).
   * If the requested cell is not a polyhedron, then the standard GetCellPoints
   * is called to return a list of unique point ids (id1, id2, id3, ...).
   */
  void GetFaceStream(svtkIdType cellId, svtkIdList* ptIds);

  /**
   * Get the number of faces and the face stream of a polyhedral cell.
   * The output \a ptIds has the following format:
   * (numFace0Pts, id1, id2, id3, numFace1Pts,id1, id2, id3, ...).
   * If the requested cell is not a polyhedron, then the standard GetCellPoints
   * is called to return the number of points and a list of unique point ids
   * (id1, id2, id3, ...).
   */
  void GetFaceStream(svtkIdType cellId, svtkIdType& nfaces, svtkIdType const*& ptIds);

  //@{
  /**
   * Provide cell information to define the dataset.
   *
   * Cells like svtkPolyhedron require points plus a list of faces. To handle
   * svtkPolyhedron, SetCells() support a special input cellConnectivities format
   * (numCellFaces, numFace0Pts, id1, id2, id3, numFace1Pts,id1, id2, id3, ...)
   * The functions use svtkPolyhedron::DecomposeAPolyhedronCell() to convert
   * polyhedron cells into standard format.
   */
  void SetCells(int type, svtkCellArray* cells);
  void SetCells(int* types, svtkCellArray* cells);
  void SetCells(svtkUnsignedCharArray* cellTypes, svtkCellArray* cells);
  void SetCells(svtkUnsignedCharArray* cellTypes, svtkCellArray* cells, svtkIdTypeArray* faceLocations,
    svtkIdTypeArray* faces);
  //@}

  /**
   * Return the unstructured grid connectivity array.
   */
  svtkCellArray* GetCells() { return this->Connectivity; }

  /**
   * Topological inquiry to get all cells using list of points exclusive of
   * cell specified (e.g., cellId).
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  void GetCellNeighbors(svtkIdType cellId, svtkIdList* ptIds, svtkIdList* cellIds) override;

  //@{
  /**
   * Use these methods only if the dataset has been specified as
   * Editable. See svtkPointSet for more information.
   */
  svtkIdType InsertNextLinkedCell(int type, int npts, const svtkIdType pts[]) SVTK_SIZEHINT(pts, npts);
  void RemoveReferenceToCell(svtkIdType ptId, svtkIdType cellId);
  void AddReferenceToCell(svtkIdType ptId, svtkIdType cellId);
  void ResizeCellList(svtkIdType ptId, int size);
  //@}

  //@{
  /**
   * Set / Get the piece and the number of pieces. Similar to extent in 3D.
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
   * Fill svtkIdTypeArray container with list of cell Ids.  This
   * method traverses all cells and, for a particular cell type,
   * inserts the cell Id into the container.
   */
  void GetIdsOfCellsOfType(int type, svtkIdTypeArray* array) override;

  /**
   * Returns whether cells are all of the same type.
   */
  int IsHomogeneous() override;

  /**
   * This method will remove any cell that is marked as ghost
   * (has the svtkDataSetAttributes::DUPLICATECELL bit set).
   */
  void RemoveGhostCells();

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkUnstructuredGrid* GetData(svtkInformation* info);
  static svtkUnstructuredGrid* GetData(svtkInformationVector* v, int i = 0);
  //@}

  /**
   * Special support for polyhedron. Return nullptr for all other cell types.
   */
  svtkIdType* GetFaces(svtkIdType cellId);

  //@{
  /**
   * Get pointer to faces and facelocations. Support for polyhedron cells.
   */
  svtkIdTypeArray* GetFaces();
  svtkIdTypeArray* GetFaceLocations();
  //@}

  /**
   * Special function used by svtkUnstructuredGridReader.
   * By default svtkUnstructuredGrid does not contain face information, which is
   * only used by polyhedron cells. If so far no polyhedron cells have been
   * added, Faces and FaceLocations pointers will be nullptr. In this case, need to
   * initialize the arrays and assign values to the previous non-polyhedron cells.
   */
  int InitializeFacesRepresentation(svtkIdType numPrevCells);

  /**
   * Return the mesh (geometry/topology) modification time.
   * This time is different from the usual MTime which also takes into
   * account the modification of data arrays. This function can be used to
   * track the changes on the mesh separately from the data arrays
   * (eg. static mesh over time with transient data).
   */
  virtual svtkMTimeType GetMeshMTime();

  /**
   * A static method for converting a polyhedron svtkCellArray of format
   * [nCellFaces, nFace0Pts, i, j, k, nFace1Pts, i, j, k, ...]
   * into three components: (1) an integer indicating the number of faces
   * (2) a standard svtkCellArray storing point ids [nCell0Pts, i, j, k]
   * and (3) an svtkIdTypeArray storing face connectivity in format
   * [nFace0Pts, i, j, k, nFace1Pts, i, j, k, ...]
   * Note: input is assumed to contain only one polyhedron cell.
   * Outputs (2) and (3) will be stacked at the end of the input
   * cellArray and faces. The original data in the input will not
   * be touched.
   */
  static void DecomposeAPolyhedronCell(svtkCellArray* polyhedronCellArray, svtkIdType& nCellpts,
    svtkIdType& nCellfaces, svtkCellArray* cellArray, svtkIdTypeArray* faces);

  static void DecomposeAPolyhedronCell(const svtkIdType* polyhedronCellStream, svtkIdType& nCellpts,
    svtkIdType& nCellfaces, svtkCellArray* cellArray, svtkIdTypeArray* faces);

  /**
   * A static method for converting an input polyhedron cell stream of format
   * [nFace0Pts, i, j, k, nFace1Pts, i, j, k, ...]
   * into three components: (1) an integer indicating the number of faces
   * (2) a standard svtkCellArray storing point ids [nCell0Pts, i, j, k]
   * and (3) an svtkIdTypeArray storing face connectivity in format
   * [nFace0Pts, i, j, k, nFace1Pts, i, j, k, ...]
   * Note: input is assumed to contain only one polyhedron cell.
   * Outputs (2) and (3) will be stacked at the end of the input
   * cellArray and faces. The original data in the input will not
   * be touched.
   */
  static void DecomposeAPolyhedronCell(svtkIdType nCellFaces, const svtkIdType* inFaceStream,
    svtkIdType& nCellpts, svtkCellArray* cellArray, svtkIdTypeArray* faces);

  /**
   * Convert pid in a face stream into idMap[pid]. The face stream is of format
   * [nCellFaces, nFace0Pts, i, j, k, nFace1Pts, i, j, k, ...]. The user is
   * responsible to make sure all the Ids in faceStream do not exceed the
   * range of idMap.
   */
  static void ConvertFaceStreamPointIds(svtkIdList* faceStream, svtkIdType* idMap);

  /**
   * Convert pid in a face stream into idMap[pid]. The face stream is of format
   * [nFace0Pts, i, j, k, nFace1Pts, i, j, k, ...]. The user is responsible to
   * make sure all the Ids in faceStream do not exceed the range of idMap.
   */
  static void ConvertFaceStreamPointIds(svtkIdType nfaces, svtkIdType* faceStream, svtkIdType* idMap);

  //====================== Begin Legacy Methods ================================

  /**
   * Get the array of all the starting indices of cell definitions
   * in the cell array.
   *
   * @warning svtkCellArray supports random access now. This array is no
   * longer used.
   */
  svtkIdTypeArray* GetCellLocationsArray();

  //@{
  /**
   * Special methods specific to svtkUnstructuredGrid for defining the cells
   * composing the dataset. Most cells require just arrays of cellTypes,
   * cellLocations and cellConnectivities which implicitly define the set of
   * points in each cell and their ordering. In those cases the
   * cellConnectivities are of the format
   * (numFace0Pts, id1, id2, id3, numFace1Pts, id1, id2, id3...). However, some
   * cells like svtkPolyhedron require points plus a list of faces. To handle
   * svtkPolyhedron, SetCells() support a special input cellConnectivities format
   * (numCellFaces, numFace0Pts, id1, id2, id3, numFace1Pts,id1, id2, id3, ...)
   * The functions use svtkPolyhedron::DecomposeAPolyhedronCell() to convert
   * polyhedron cells into standard format.
   *
   * @warning The cellLocations array is no longer used; this information
   * is stored in svtkCellArray. Use the other SetCells overloads.
   */
  void SetCells(
    svtkUnsignedCharArray* cellTypes, svtkIdTypeArray* cellLocations, svtkCellArray* cells);
  void SetCells(svtkUnsignedCharArray* cellTypes, svtkIdTypeArray* cellLocations, svtkCellArray* cells,
    svtkIdTypeArray* faceLocations, svtkIdTypeArray* faces);
  //@}

  //====================== End Legacy Methods ==================================

protected:
  svtkUnstructuredGrid();
  ~svtkUnstructuredGrid() override;

  // These are all the cells that svtkUnstructuredGrid can represent. Used by
  // GetCell() (and similar) methods.
  svtkVertex* Vertex;
  svtkPolyVertex* PolyVertex;
  svtkBezierCurve* BezierCurve;
  svtkBezierQuadrilateral* BezierQuadrilateral;
  svtkBezierHexahedron* BezierHexahedron;
  svtkBezierTriangle* BezierTriangle;
  svtkBezierTetra* BezierTetra;
  svtkBezierWedge* BezierWedge;
  svtkLagrangeCurve* LagrangeCurve;
  svtkLagrangeQuadrilateral* LagrangeQuadrilateral;
  svtkLagrangeHexahedron* LagrangeHexahedron;
  svtkLagrangeTriangle* LagrangeTriangle;
  svtkLagrangeTetra* LagrangeTetra;
  svtkLagrangeWedge* LagrangeWedge;
  svtkLine* Line;
  svtkPolyLine* PolyLine;
  svtkTriangle* Triangle;
  svtkTriangleStrip* TriangleStrip;
  svtkPixel* Pixel;
  svtkQuad* Quad;
  svtkPolygon* Polygon;
  svtkTetra* Tetra;
  svtkVoxel* Voxel;
  svtkHexahedron* Hexahedron;
  svtkWedge* Wedge;
  svtkPyramid* Pyramid;
  svtkPentagonalPrism* PentagonalPrism;
  svtkHexagonalPrism* HexagonalPrism;
  svtkQuadraticEdge* QuadraticEdge;
  svtkQuadraticTriangle* QuadraticTriangle;
  svtkQuadraticQuad* QuadraticQuad;
  svtkQuadraticPolygon* QuadraticPolygon;
  svtkQuadraticTetra* QuadraticTetra;
  svtkQuadraticHexahedron* QuadraticHexahedron;
  svtkQuadraticWedge* QuadraticWedge;
  svtkQuadraticPyramid* QuadraticPyramid;
  svtkQuadraticLinearQuad* QuadraticLinearQuad;
  svtkBiQuadraticQuad* BiQuadraticQuad;
  svtkTriQuadraticHexahedron* TriQuadraticHexahedron;
  svtkQuadraticLinearWedge* QuadraticLinearWedge;
  svtkBiQuadraticQuadraticWedge* BiQuadraticQuadraticWedge;
  svtkBiQuadraticQuadraticHexahedron* BiQuadraticQuadraticHexahedron;
  svtkBiQuadraticTriangle* BiQuadraticTriangle;
  svtkCubicLine* CubicLine;
  svtkConvexPointSet* ConvexPointSet;
  svtkPolyhedron* Polyhedron;
  svtkEmptyCell* EmptyCell;

  // Points derived from svtkPointSet.
  // Attribute data (i.e., point and cell data (i.e., scalars, vectors, normals, tcoords)
  // derived from svtkDataSet.

  // The heart of the data represention. The points are managed by the
  // superclass svtkPointSet. A cell is defined by its connectivity (i.e., the
  // point ids that define the cell) and the cell type, represented by the
  // Connectivity and Types arrays.
  // Finally, when certain topological information is needed (e.g.,
  // all the cells that use a point), the cell links array is built.
  svtkSmartPointer<svtkCellArray> Connectivity;
  svtkSmartPointer<svtkAbstractCellLinks> Links;
  svtkSmartPointer<svtkUnsignedCharArray> Types;

  // Set of all cell types present in the grid. All entries are unique.
  svtkSmartPointer<svtkCellTypes> DistinctCellTypes;

  // The DistinctCellTypes is cached, so we keep track of the last time it was
  // updated so we can compare it to the modified time of the Types array.
  svtkMTimeType DistinctCellTypesUpdateMTime;

  // Special support for polyhedra/cells with explicit face representations.
  // The Faces class represents polygonal faces using a modified svtkCellArray
  // structure. Each cell face list begins with the total number of faces in
  // the cell, followed by a svtkCellArray data organization
  // (n,i,j,k,n,i,j,k,...).
  svtkSmartPointer<svtkIdTypeArray> Faces;
  svtkSmartPointer<svtkIdTypeArray> FaceLocations;

  // Legacy support -- stores the old-style cell array locations.
  svtkSmartPointer<svtkIdTypeArray> CellLocations;

  svtkIdType InternalInsertNextCell(int type, svtkIdType npts, const svtkIdType ptIds[]) override;
  svtkIdType InternalInsertNextCell(int type, svtkIdList* ptIds) override;
  svtkIdType InternalInsertNextCell(int type, svtkIdType npts, const svtkIdType ptIds[],
    svtkIdType nfaces, const svtkIdType faces[]) override;
  void InternalReplaceCell(svtkIdType cellId, int npts, const svtkIdType pts[]) override;

private:
  // Hide these from the user and the compiler.
  svtkUnstructuredGrid(const svtkUnstructuredGrid&) = delete;
  void operator=(const svtkUnstructuredGrid&) = delete;

  void Cleanup();
};

#endif
