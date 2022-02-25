/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataSet.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDataSet
 * @brief   abstract class to specify dataset behavior
 *
 * svtkDataSet is an abstract class that specifies an interface for dataset
 * objects. svtkDataSet also provides methods to provide information about
 * the data, such as center, bounding box, and representative length.
 *
 * In svtk a dataset consists of a structure (geometry and topology) and
 * attribute data. The structure is defined implicitly or explicitly as
 * a collection of cells. The geometry of the structure is contained in
 * point coordinates plus the cell interpolation functions. The topology
 * of the dataset structure is defined by cell types and how the cells
 * share their defining points.
 *
 * Attribute data in svtk is either point data (data at points) or cell data
 * (data at cells). Typically filters operate on point data, but some may
 * operate on cell data, both cell and point data, either one, or none.
 *
 * @sa
 * svtkPointSet svtkStructuredPoints svtkStructuredGrid svtkUnstructuredGrid
 * svtkRectilinearGrid svtkPolyData svtkPointData svtkCellData
 * svtkDataObject svtkFieldData
 */

#ifndef svtkDataSet_h
#define svtkDataSet_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataObject.h"

class svtkCell;
class svtkCellData;
class svtkCellIterator;
class svtkCellTypes;
class svtkGenericCell;
class svtkIdList;
class svtkPointData;
class svtkUnsignedCharArray;
class svtkCallbackCommand;

class SVTKCOMMONDATAMODEL_EXPORT svtkDataSet : public svtkDataObject
{
public:
  svtkTypeMacro(svtkDataSet, svtkDataObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Copy the geometric and topological structure of an object. Note that
   * the invoking object and the object pointed to by the parameter ds must
   * be of the same type.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  virtual void CopyStructure(svtkDataSet* ds) = 0;

  /**
   * Copy the attributes associated with the specified dataset to this
   * instance of svtkDataSet.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  virtual void CopyAttributes(svtkDataSet* ds);

  /**
   * Determine the number of points composing the dataset.
   * THIS METHOD IS THREAD SAFE
   */
  virtual svtkIdType GetNumberOfPoints() = 0;

  /**
   * Determine the number of cells composing the dataset.
   * THIS METHOD IS THREAD SAFE
   */
  virtual svtkIdType GetNumberOfCells() = 0;

  /**
   * Get point coordinates with ptId such that: 0 <= ptId < NumberOfPoints.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  virtual double* GetPoint(svtkIdType ptId) SVTK_SIZEHINT(3) = 0;

  /**
   * Copy point coordinates into user provided array x[3] for specified
   * point id.
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  virtual void GetPoint(svtkIdType id, double x[3]);

  /**
   * Return an iterator that traverses the cells in this data set.
   */
  SVTK_NEWINSTANCE
  virtual svtkCellIterator* NewCellIterator();

  /**
   * Get cell with cellId such that: 0 <= cellId < NumberOfCells.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  virtual svtkCell* GetCell(svtkIdType cellId) = 0;
  virtual svtkCell* GetCell(int svtkNotUsed(i), int svtkNotUsed(j), int svtkNotUsed(k))
  {
    svtkErrorMacro("ijk indices are only valid with structured data!");
    return nullptr;
  }

  void SetCellOrderAndRationalWeights(svtkIdType cellId, svtkGenericCell* cell);

  /**
   * Get cell with cellId such that: 0 <= cellId < NumberOfCells.
   * This is a thread-safe alternative to the previous GetCell()
   * method.
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  virtual void GetCell(svtkIdType cellId, svtkGenericCell* cell) = 0;

  /**
   * Get the bounds of the cell with cellId such that:
   * 0 <= cellId < NumberOfCells.
   * A subclass may be able to determine the bounds of cell without using
   * an expensive GetCell() method. A default implementation is provided
   * that actually uses a GetCell() call.  This is to ensure the method
   * is available to all datasets.  Subclasses should override this method
   * to provide an efficient implementation.
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  virtual void GetCellBounds(svtkIdType cellId, double bounds[6]);

  /**
   * Get type of cell with cellId such that: 0 <= cellId < NumberOfCells.
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  virtual int GetCellType(svtkIdType cellId) = 0;

  /**
   * Get a list of types of cells in a dataset. The list consists of an array
   * of types (not necessarily in any order), with a single entry per type.
   * For example a dataset 5 triangles, 3 lines, and 100 hexahedra would
   * result a list of three entries, corresponding to the types SVTK_TRIANGLE,
   * SVTK_LINE, and SVTK_HEXAHEDRON.
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  virtual void GetCellTypes(svtkCellTypes* types);

  /**
   * Topological inquiry to get points defining cell.
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  virtual void GetCellPoints(svtkIdType cellId, svtkIdList* ptIds) = 0;

  /**
   * Topological inquiry to get cells using point.
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  virtual void GetPointCells(svtkIdType ptId, svtkIdList* cellIds) = 0;

  /**
   * Topological inquiry to get all cells using list of points exclusive of
   * cell specified (e.g., cellId). Note that the list consists of only
   * cells that use ALL the points provided.
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  virtual void GetCellNeighbors(svtkIdType cellId, svtkIdList* ptIds, svtkIdList* cellIds);

  //@{
  /**
   * Locate the closest point to the global coordinate x. Return the
   * point id. If point id < 0; then no point found. (This may arise
   * when point is outside of dataset.)
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  svtkIdType FindPoint(double x, double y, double z)
  {
    double xyz[3];
    xyz[0] = x;
    xyz[1] = y;
    xyz[2] = z;
    return this->FindPoint(xyz);
  }
  virtual svtkIdType FindPoint(double x[3]) = 0;
  //@}

  /**
   * Locate cell based on global coordinate x and tolerance
   * squared. If cell and cellId is non-nullptr, then search starts from
   * this cell and looks at immediate neighbors.  Returns cellId >= 0
   * if inside, < 0 otherwise.  The parametric coordinates are
   * provided in pcoords[3]. The interpolation weights are returned in
   * weights[]. (The number of weights is equal to the number of
   * points in the found cell). Tolerance is used to control how close
   * the point is to be considered "in" the cell.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  virtual svtkIdType FindCell(double x[3], svtkCell* cell, svtkIdType cellId, double tol2, int& subId,
    double pcoords[3], double* weights) = 0;

  /**
   * This is a version of the above method that can be used with
   * multithreaded applications. A svtkGenericCell must be passed in
   * to be used in internal calls that might be made to GetCell()
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  virtual svtkIdType FindCell(double x[3], svtkCell* cell, svtkGenericCell* gencell, svtkIdType cellId,
    double tol2, int& subId, double pcoords[3], double* weights) = 0;

  /**
   * Locate the cell that contains a point and return the cell. Also returns
   * the subcell id, parametric coordinates and weights for subsequent
   * interpolation. This method combines the derived class methods
   * int FindCell and svtkCell *GetCell. Derived classes may provide a more
   * efficient implementation. See for example svtkStructuredPoints.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  virtual svtkCell* FindAndGetCell(double x[3], svtkCell* cell, svtkIdType cellId, double tol2,
    int& subId, double pcoords[3], double* weights);

  /**
   * Datasets are composite objects and need to check each part for MTime
   * THIS METHOD IS THREAD SAFE
   */
  svtkMTimeType GetMTime() override;

  /**
   * Return a pointer to this dataset's cell data.
   * THIS METHOD IS THREAD SAFE
   */
  svtkCellData* GetCellData() { return this->CellData; }

  /**
   * Return a pointer to this dataset's point data.
   * THIS METHOD IS THREAD SAFE
   */
  svtkPointData* GetPointData() { return this->PointData; }

  /**
   * Reclaim any extra memory used to store data.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  virtual void Squeeze();

  /**
   * Compute the data bounding box from data points.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  virtual void ComputeBounds();

  /**
   * Return a pointer to the geometry bounding box in the form
   * (xmin,xmax, ymin,ymax, zmin,zmax).
   * THIS METHOD IS NOT THREAD SAFE.
   */
  double* GetBounds() SVTK_SIZEHINT(6);

  /**
   * Return a pointer to the geometry bounding box in the form
   * (xmin,xmax, ymin,ymax, zmin,zmax).
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  void GetBounds(double bounds[6]);

  /**
   * Get the center of the bounding box.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  double* GetCenter() SVTK_SIZEHINT(3);

  /**
   * Get the center of the bounding box.
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  void GetCenter(double center[3]);

  /**
   * Return the length of the diagonal of the bounding box.
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  double GetLength();

  /**
   * Restore data object to initial state.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  void Initialize() override;

  /**
   * Convenience method to get the range of the first component (and only
   * the first component) of any scalars in the data set.  If the data has
   * both point data and cell data, it returns the (min/max) range of
   * combined point and cell data.  If there are no point or cell scalars
   * the method will return (0,1).  Note: It might be necessary to call
   * Update to create or refresh the scalars before calling this method.
   * THIS METHOD IS THREAD SAFE IF FIRST CALLED FROM A SINGLE THREAD AND
   * THE DATASET IS NOT MODIFIED
   */
  virtual void GetScalarRange(double range[2]);

  /**
   * Convenience method to get the range of the first component (and only
   * the first component) of any scalars in the data set.  If the data has
   * both point data and cell data, it returns the (min/max) range of
   * combined point and cell data.  If there are no point or cell scalars
   * the method will return (0,1).  Note: It might be necessary to call
   * Update to create or refresh the scalars before calling this method.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  double* GetScalarRange() SVTK_SIZEHINT(2);

  /**
   * Convenience method returns largest cell size in dataset. This is generally
   * used to allocate memory for supporting data structures.
   * THIS METHOD IS THREAD SAFE
   */
  virtual int GetMaxCellSize() = 0;

  /**
   * Return the actual size of the data in kibibytes (1024 bytes). This number
   * is valid only after the pipeline has updated. The memory size
   * returned is guaranteed to be greater than or equal to the
   * memory required to represent the data (e.g., extra space in
   * arrays, etc. are not included in the return value). THIS METHOD
   * IS THREAD SAFE.
   */
  unsigned long GetActualMemorySize() override;

  /**
   * Return the type of data object.
   */
  int GetDataObjectType() override { return SVTK_DATA_SET; }

  //@{
  /**
   * Shallow and Deep copy.
   */
  void ShallowCopy(svtkDataObject* src) override;
  void DeepCopy(svtkDataObject* src) override;
  //@}

  enum FieldDataType
  {
    DATA_OBJECT_FIELD = 0,
    POINT_DATA_FIELD = 1,
    CELL_DATA_FIELD = 2
  };

  /**
   * This method checks to see if the cell and point attributes
   * match the geometry.  Many filters will crash if the number of
   * tuples in an array is less than the number of points/cells.
   * This method returns 1 if there is a mismatch,
   * and 0 if everything is ok.  It prints an error if an
   * array is too short, and a warning if an array is too long.
   */
  int CheckAttributes();

  //@{
  /**
   * Normally called by pipeline executives or algorithms only. This method
   * computes the ghost arrays for a given dataset. The zeroExt argument
   * specifies the extent of the region which ghost type = 0.
   */
  virtual void GenerateGhostArray(int zeroExt[6]) { this->GenerateGhostArray(zeroExt, false); }
  virtual void GenerateGhostArray(int zeroExt[6], bool cellOnly);
  //@}

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkDataSet* GetData(svtkInformation* info);
  static svtkDataSet* GetData(svtkInformationVector* v, int i = 0);
  //@}

  /**
   * Returns the attributes of the data object as a svtkFieldData.
   * This returns non-null values in all the same cases as GetAttributes,
   * in addition to the case of FIELD, which will return the field data
   * for any svtkDataObject subclass.
   */
  svtkFieldData* GetAttributesAsFieldData(int type) override;

  /**
   * Get the number of elements for a specific attribute type (POINT, CELL, etc.).
   */
  svtkIdType GetNumberOfElements(int type) override;

  /**
   * Returns 1 if there are any ghost cells
   * 0 otherwise.
   */
  bool HasAnyGhostCells();
  /**
   * Returns 1 if there are any ghost points
   * 0 otherwise.
   */
  bool HasAnyGhostPoints();
  /**
   * Returns 1 if there are any blanking cells
   * 0 otherwise. Blanking is supported only for svtkStructuredGrid
   * and svtkUniformGrid
   */
  virtual bool HasAnyBlankCells() { return 0; }
  /**
   * Returns 1 if there are any blanking points
   * 0 otherwise. Blanking is supported only for svtkStructuredGrid
   * and svtkUniformGrid
   */
  virtual bool HasAnyBlankPoints() { return 0; }

  /**
   * Gets the array that defines the ghost type of each point.
   * We cache the pointer to the array to save a lookup involving string comparisons
   */
  svtkUnsignedCharArray* GetPointGhostArray();
  /**
   * Updates the pointer to the point ghost array.
   */
  void UpdatePointGhostArrayCache();

  /**
   * Allocate ghost array for points.
   */
  svtkUnsignedCharArray* AllocatePointGhostArray();

  /**
   * Get the array that defines the ghost type of each cell.
   * We cache the pointer to the array to save a lookup involving string comparisons
   */
  svtkUnsignedCharArray* GetCellGhostArray();
  /**
   * Updates the pointer to the cell ghost array.
   */
  void UpdateCellGhostArrayCache();

  /**
   * Allocate ghost array for cells.
   */
  svtkUnsignedCharArray* AllocateCellGhostArray();

protected:
  // Constructor with default bounds (0,1, 0,1, 0,1).
  svtkDataSet();
  ~svtkDataSet() override;

  /**
   * Compute the range of the scalars and cache it into ScalarRange
   * only if the cache became invalid (ScalarRangeComputeTime).
   */
  virtual void ComputeScalarRange();

  /**
   * Helper function that tests if any of the values in 'a' have bitFlag set.
   * The test performed is (value & bitFlag).
   */
  bool IsAnyBitSet(svtkUnsignedCharArray* a, int bitFlag);

  svtkCellData* CellData;            // Scalars, vectors, etc. associated w/ each cell
  svtkPointData* PointData;          // Scalars, vectors, etc. associated w/ each point
  svtkCallbackCommand* DataObserver; // Observes changes to cell/point data
  svtkTimeStamp ComputeTime;         // Time at which bounds, center, etc. computed
  double Bounds[6];                 // (xmin,xmax, ymin,ymax, zmin,zmax) geometric bounds
  double Center[3];

  // Cached scalar range
  double ScalarRange[2];

  // Time at which scalar range is computed
  svtkTimeStamp ScalarRangeComputeTime;

  //@{
  /**
   * These arrays pointers are caches used to avoid a string comparison (when
   * getting ghost arrays using GetArray(name))
   */
  svtkUnsignedCharArray* PointGhostArray;
  svtkUnsignedCharArray* CellGhostArray;
  bool PointGhostArrayCached;
  bool CellGhostArrayCached;
  //@}

private:
  void InternalDataSetCopy(svtkDataSet* src);
  /**
   * Called when point/cell data is modified
   * Updates caches to point/cell ghost arrays.
   */
  static void OnDataModified(
    svtkObject* source, unsigned long eid, void* clientdata, void* calldata);

  friend class svtkImageAlgorithmToDataSetFriendship;

private:
  svtkDataSet(const svtkDataSet&) = delete;
  void operator=(const svtkDataSet&) = delete;
};

inline void svtkDataSet::GetPoint(svtkIdType id, double x[3])
{
  double* pt = this->GetPoint(id);
  x[0] = pt[0];
  x[1] = pt[1];
  x[2] = pt[2];
}

#endif
