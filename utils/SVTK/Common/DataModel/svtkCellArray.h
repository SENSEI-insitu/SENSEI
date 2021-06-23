/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCellArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCellArray
 * @brief   object to represent cell connectivity
 *
 * svtkCellArray stores dataset topologies as an explicit connectivity table
 * listing the point ids that make up each cell.
 *
 * Internally, the connectivity table is represented as two arrays: Offsets and
 * Connectivity.
 *
 * Offsets is an array of [numCells+1] values indicating the index in the
 * Connectivity array where each cell's points start. The last value is always
 * the length of the Connectivity array.
 *
 * The Connectivity array stores the lists of point ids for each cell.
 *
 * Thus, for a dataset consisting of 2 triangles, a quad, and a line, the
 * internal arrays will appear as follows:
 *
 * ```
 * Topology:
 * ---------
 * Cell 0: Triangle | point ids: {0, 1, 2}
 * Cell 1: Triangle | point ids: {5, 7, 2}
 * Cell 2: Quad     | point ids: {3, 4, 6, 7}
 * Cell 4: Line     | point ids: {5, 8}
 *
 * svtkCellArray (current):
 * -----------------------
 * Offsets:      {0, 3, 6, 10, 12}
 * Connectivity: {0, 1, 2, 5, 7, 2, 3, 4, 6, 7, 5, 8}
 * ```
 *
 * While this class provides traversal methods (the legacy InitTraversal(),
 * GetNextCell() methods, and the newer method GetCellAtId()) these are in
 * general not thread-safe. Whenever possible it is preferrable to use a
 * local thread-safe, svtkCellArrayIterator object, which can be obtained via:
 *
 * ```
 * auto iter = svtk::TakeSmartPointer(cellArray->NewIterator());
 * for (iter->GoToFirstCell(); !iter->IsDoneWithTraversal(); iter->GoToNextCell())
 * {
 *   // do work with iter
 * }
 * ```
 * (Note however that depending on the type and structure of internal
 * storage, a cell array iterator may be significantly slower than direct
 * traversal over the cell array due to extra data copying. Factors of 3-4X
 * are not uncommon. See svtkCellArrayIterator for more information. Also note
 * that an iterator may become invalid if the internal svtkCellArray storage
 * is modified.)
 *
 * Other methods are also available for allocation and memory-related
 * management; insertion of new cells into the svtkCellArray; and limited
 * editing operations such as replacing one cell with a new cell of the
 * same size.
 *
 * The internal arrays may store either 32- or 64-bit values, though most of
 * the API will prefer to use svtkIdType to refer to items in these
 * arrays. This enables significant memory savings when svtkIdType is 64-bit,
 * but 32 bits are sufficient to store all of the values in the connectivity
 * table. Using 64-bit storage with a 32-bit svtkIdType is permitted, but
 * values too large to fit in a 32-bit signed integer will be truncated when
 * accessed through the API. (The particular internal storage type has
 * implications on performance depending on svtkIdType. If the internal
 * storage is equivalent to svtkIdType, then methods that return pointers to
 * arrays of point ids can share the internal storage; otherwise a copy of
 * internal memory must be performed.)
 *
 * Methods for managing the storage type are:
 *
 * - `bool IsStorage64Bit()`
 * - `bool IsStorageShareable() // Can pointers to internal storage be shared`
 * - `void Use32BitStorage()`
 * - `void Use64BitStorage()`
 * - `void UseDefaultStorage() // Depends on svtkIdType`
 * - `bool CanConvertTo32BitStorage()`
 * - `bool CanConvertTo64BitStorage()`
 * - `bool CanConvertToDefaultStorage() // Depends on svtkIdType`
 * - `bool ConvertTo32BitStorage()`
 * - `bool ConvertTo64BitStorage()`
 * - `bool ConvertToDefaultStorage() // Depends on svtkIdType`
 * - `bool ConvertToSmallestStorage() // Depends on current values in arrays`
 *
 * Note that some legacy methods are still available that reflect the
 * previous storage format of this data, which embedded the cell sizes into
 * the Connectivity array:
 *
 * ```
 * svtkCellArray (legacy):
 * ----------------------
 * Connectivity: {3, 0, 1, 2, 3, 5, 7, 2, 4, 3, 4, 6, 7, 2, 5, 8}
 *                |--Cell 0--||--Cell 1--||----Cell 2---||--C3-|
 * ```
 *
 * The methods require an external lookup table to allow random access, which
 * was historically stored in the svtkCellTypes object. The following methods in
 * svtkCellArray still support this style of indexing for compatibility
 * purposes, but these are slow as they must perform some complex computations
 * to convert the old "location" into the new "offset" and should be avoided.
 * These methods (and their modern equivalents) are:
 *
 * - GetCell (Prefer GetCellAtId)
 * - GetInsertLocation (Prefer GetNumberOfCells)
 * - GetTraversalLocation (Prefer GetTraversalCellId, or better, NewIterator)
 * - SetTraversalLocation (Prefer SetTraversalLocation, or better, NewIterator)
 * - ReverseCell (Prefer ReverseCellAtId)
 * - ReplaceCell (Prefer ReplaceCellAtId)
 * - SetCells (Use ImportLegacyFormat, or SetData)
 * - GetData (Use ExportLegacyFormat, or Get[Offsets|Connectivity]Array[|32|64])
 *
 * Some other legacy methods were completely removed, such as GetPointer() /
 * WritePointer(), since they are cannot be effectively emulated under the
 * current design. If external code needs to support both the old and new
 * version of the svtkCellArray API, the SVTK_CELL_ARRAY_V2 preprocessor
 * definition may be used to detect which API is being compiled against.
 *
 * @sa svtkCellTypes svtkCellLinks
 */

#ifndef svtkCellArray_h
#define svtkCellArray_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

#include "svtkAOSDataArrayTemplate.h" // Needed for inline methods
#include "svtkCell.h"                 // Needed for inline methods
#include "svtkDataArrayRange.h"       // Needed for inline methods
#include "svtkSmartPointer.h"         // For svtkSmartPointer
#include "svtkTypeInt32Array.h"       // Needed for inline methods
#include "svtkTypeInt64Array.h"       // Needed for inline methods
#include "svtkTypeList.h"             // Needed for ArrayList definition

#include <cassert>          // for assert
#include <initializer_list> // for API
#include <type_traits>      // for std::is_same
#include <utility>          // for std::forward

/**
 * @def SVTK_CELL_ARRAY_V2
 * @brief This preprocessor definition indicates that the updated svtkCellArray
 * is being used. It may be used to conditionally switch between old and new
 * API when both must be supported.
 *
 * For example:
 *
 * ```
 * svtkIdType npts;
 *
 * #ifdef SVTK_CELL_ARRAY_V2
 * const svtkIdType *pts;
 * #else // SVTK_CELL_ARRAY_V2
 * svtkIdType *pts'
 * #endif // SVTK_CELL_ARRAY_V2
 *
 * cellArray->GetCell(legacyLocation, npts, pts);
 * ```
 */
#define SVTK_CELL_ARRAY_V2

class svtkCellArrayIterator;
class svtkIdTypeArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkCellArray : public svtkObject
{
public:
  using ArrayType32 = svtkTypeInt32Array;
  using ArrayType64 = svtkTypeInt64Array;

  //@{
  /**
   * Standard methods for instantiation, type information, and
   * printing.
   */
  static svtkCellArray* New();
  svtkTypeMacro(svtkCellArray, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  void PrintDebug(ostream& os);
  //@}

  /**
   * List of possible array types used for storage. May be used with
   * svtkArrayDispatch::Dispatch[2]ByArray to process internal arrays.
   * Both the Connectivity and Offset arrays are guaranteed to have the same
   * type.
   *
   * @sa svtkCellArray::Visit() for a simpler mechanism.
   */
  using StorageArrayList = svtkTypeList::Create<ArrayType32, ArrayType64>;

  /**
   * List of possible ArrayTypes that are compatible with internal storage.
   * Single component AOS-layout arrays holding one of these types may be
   * passed to the method SetData to setup the cell array state.
   *
   * This can be used with svtkArrayDispatch::DispatchByArray, etc to
   * check input arrays before assigning them to a cell array.
   */
  using InputArrayList =
    typename svtkTypeList::Unique<svtkTypeList::Create<svtkAOSDataArrayTemplate<int>,
      svtkAOSDataArrayTemplate<long>, svtkAOSDataArrayTemplate<long long> > >::Result;

  /**
   * Allocate memory.
   *
   * This currently allocates both the offsets and connectivity arrays to @a sz.
   *
   * @note It is preferrable to use AllocateEstimate(numCells, maxCellSize)
   * or AllocateExact(numCells, connectivitySize) instead.
   */
  svtkTypeBool Allocate(svtkIdType sz, svtkIdType svtkNotUsed(ext) = 1000)
  {
    return this->AllocateExact(sz, sz) ? 1 : 0;
  }

  /**
   * @brief Pre-allocate memory in internal data structures. Does not change
   * the number of cells, only the array capacities. Existing data is NOT
   * preserved.
   * @param numCells The number of expected cells in the dataset.
   * @param maxCellSize The number of points per cell to allocate memory for.
   * @return True if allocation succeeds.
   * @sa Squeeze AllocateExact AllocateCopy
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
   * @sa Squeeze AllocateEstimate AllocateCopy
   */
  bool AllocateExact(svtkIdType numCells, svtkIdType connectivitySize);

  /**
   * @brief Pre-allocate memory in internal data structures to match the used
   * size of the input svtkCellArray. Does not change
   * the number of cells, only the array capacities. Existing data is NOT
   * preserved.
   * @param other The svtkCellArray to use as a reference.
   * @return True if allocation succeeds.
   * @sa Squeeze AllocateEstimate AllocateExact
   */
  bool AllocateCopy(svtkCellArray* other)
  {
    return this->AllocateExact(other->GetNumberOfCells(), other->GetNumberOfConnectivityIds());
  }

  /**
   * @brief ResizeExact() resizes the internal structures to hold @a numCells
   * total cell offsets and @a connectivitySize total pointIds. Old data is
   * preserved, and newly-available memory is not initialized.
   *
   * @warning For advanced use only. You probably want an Allocate method.
   *
   * @return True if allocation succeeds.
   */
  bool ResizeExact(svtkIdType numCells, svtkIdType connectivitySize);

  /**
   * Free any memory and reset to an empty state.
   */
  void Initialize();

  /**
   * Reuse list. Reset to initial state without freeing memory.
   */
  void Reset();

  /**
   * Reclaim any extra memory while preserving data.
   *
   * @sa ConvertToSmallestStorage
   */
  void Squeeze();

  /**
   * Check that internal storage is consistent and in a valid state.
   *
   * Specifically, this function returns true if and only if:
   * - The offset and connectivity arrays have exactly one component.
   * - The offset array has at least one value and starts at 0.
   * - The offset array values never decrease.
   * - The connectivity array has as many entries as the last value in the
   *   offset array.
   */
  bool IsValid();

  /**
   * Get the number of cells in the array.
   */
  svtkIdType GetNumberOfCells() const
  {
    if (this->Storage.Is64Bit())
    {
      return this->Storage.GetArrays64().Offsets->GetNumberOfValues() - 1;
    }
    else
    {
      return this->Storage.GetArrays32().Offsets->GetNumberOfValues() - 1;
    }
  }

  /**
   * Get the number of elements in the offsets array. This will be the number of
   * cells + 1.
   */
  svtkIdType GetNumberOfOffsets() const
  {
    if (this->Storage.Is64Bit())
    {
      return this->Storage.GetArrays64().Offsets->GetNumberOfValues();
    }
    else
    {
      return this->Storage.GetArrays32().Offsets->GetNumberOfValues();
    }
  }

  /**
   * Get the size of the connectivity array that stores the point ids.
   * @note Do not confuse this with the deprecated
   * GetNumberOfConnectivityEntries(), which refers to the legacy memory
   * layout.
   */
  svtkIdType GetNumberOfConnectivityIds() const
  {
    if (this->Storage.Is64Bit())
    {
      return this->Storage.GetArrays64().Connectivity->GetNumberOfValues();
    }
    else
    {
      return this->Storage.GetArrays32().Connectivity->GetNumberOfValues();
    }
  }

  /**
   * @brief NewIterator returns a new instance of svtkCellArrayIterator that
   * is initialized to point at the first cell's data. The caller is responsible
   * for Delete()'ing the object.
   */
  SVTK_NEWINSTANCE svtkCellArrayIterator* NewIterator();

#ifndef __SVTK_WRAP__ // The wrappers have issues with some of these templates
  /**
   * Set the internal data arrays to the supplied offsets and connectivity
   * arrays.
   *
   * Note that the input arrays may be copied and not used directly. To avoid
   * copying, use svtkIdTypeArray, svtkCellArray::ArrayType32, or
   * svtkCellArray::ArrayType64.
   *
   * @{
   */
  void SetData(svtkTypeInt32Array* offsets, svtkTypeInt32Array* connectivity);
  void SetData(svtkTypeInt64Array* offsets, svtkTypeInt64Array* connectivity);
  void SetData(svtkIdTypeArray* offsets, svtkIdTypeArray* connectivity);
  void SetData(svtkAOSDataArrayTemplate<int>* offsets, svtkAOSDataArrayTemplate<int>* connectivity);
  void SetData(svtkAOSDataArrayTemplate<long>* offsets, svtkAOSDataArrayTemplate<long>* connectivity);
  void SetData(
    svtkAOSDataArrayTemplate<long long>* offsets, svtkAOSDataArrayTemplate<long long>* connectivity);
  /**@}*/
#endif // __SVTK_WRAP__

  /**
   * Sets the internal arrays to the supplied offsets and connectivity arrays.
   *
   * This is a convenience method, and may fail if the following conditions
   * are not met:
   *
   * - Both arrays must be of the same type.
   * - The array type must be one of the types in InputArrayList.
   *
   * If invalid arrays are passed in, an error is logged and the function
   * will return false.
   */
  bool SetData(svtkDataArray* offsets, svtkDataArray* connectivity);

  /**
   * @return True if the internal storage is using 64 bit arrays. If false,
   * the storage is using 32 bit arrays.
   */
  bool IsStorage64Bit() const { return this->Storage.Is64Bit(); }

  /**
   * @return True if the internal storage can be shared as a
   * pointer to svtkIdType, i.e., the type and organization of internal
   * storage is such that copying of data can be avoided, and instead
   * a pointer to svtkIdType can be used.
   */
  bool IsStorageShareable() const
  {
    if (this->Storage.Is64Bit())
    {
      return this->Storage.GetArrays64().ValueTypeIsSameAsIdType;
    }
    else
    {
      return this->Storage.GetArrays32().ValueTypeIsSameAsIdType;
    }
  }

  /**
   * Initialize internal data structures to use 32- or 64-bit storage.
   * If selecting default storage, the storage depends on the SVTK_USE_64BIT_IDS
   * setting.
   *
   * All existing data is erased.
   * @{
   */
  void Use32BitStorage();
  void Use64BitStorage();
  void UseDefaultStorage();
  /**@}*/

  /**
   * Check if the existing data can safely be converted to use 32- or 64- bit
   * storage. Ensures that all values can be converted to the target storage
   * without truncating.
   * If selecting default storage, the storage depends on the SVTK_USE_64BIT_IDS
   * setting.
   * @{
   */
  bool CanConvertTo32BitStorage() const;
  bool CanConvertTo64BitStorage() const;
  bool CanConvertToDefaultStorage() const;
  /**@}*/

  /**
   * Convert internal data structures to use 32- or 64-bit storage.
   *
   * If selecting default storage, the storage depends on the SVTK_USE_64BIT_IDS
   * setting.
   *
   * If selecting smallest storage, the data is checked to see what the smallest
   * safe storage for the existing data is, and then converts to it.
   *
   * Existing data is preserved.
   *
   * @return True on success, false on failure. If this algorithm fails, the
   * cell array will be in an unspecified state.
   *
   * @{
   */
  bool ConvertTo32BitStorage();
  bool ConvertTo64BitStorage();
  bool ConvertToDefaultStorage();
  bool ConvertToSmallestStorage();
  /**@}*/

  /**
   * Return the array used to store cell offsets. The 32/64 variants are only
   * valid when IsStorage64Bit() returns the appropriate value.
   * @{
   */
  svtkDataArray* GetOffsetsArray()
  {
    if (this->Storage.Is64Bit())
    {
      return this->GetOffsetsArray64();
    }
    else
    {
      return this->GetOffsetsArray32();
    }
  }
  ArrayType32* GetOffsetsArray32() { return this->Storage.GetArrays32().Offsets; }
  ArrayType64* GetOffsetsArray64() { return this->Storage.GetArrays64().Offsets; }
  /**@}*/

  /**
   * Return the array used to store the point ids that define the cells'
   * connectivity. The 32/64 variants are only valid when IsStorage64Bit()
   * returns the appropriate value.
   * @{
   */
  svtkDataArray* GetConnectivityArray()
  {
    if (this->Storage.Is64Bit())
    {
      return this->GetConnectivityArray64();
    }
    else
    {
      return this->GetConnectivityArray32();
    }
  }
  ArrayType32* GetConnectivityArray32() { return this->Storage.GetArrays32().Connectivity; }
  ArrayType64* GetConnectivityArray64() { return this->Storage.GetArrays64().Connectivity; }
  /**@}*/

  /**
   * Check if all cells have the same number of vertices.
   *
   * The return value is coded as:
   * * -1 = heterogeneous
   * * 0 = Cell array empty
   * * n (positive integer) = homogeneous array of cell size n
   */
  svtkIdType IsHomogeneous();

  /**
   * @warning This method is not thread-safe. Consider using the NewIterator()
   * iterator instead.
   *
   * InitTraversal() initializes the traversal of the list of cells.
   *
   * @note This method is not thread-safe and has tricky syntax to use
   * correctly. Prefer the use of svtkCellArrayIterator (see NewIterator()).
   */
  void InitTraversal();

  /**
   * @warning This method is not thread-safe. Consider using the NewIterator()
   * iterator instead.
   *
   * GetNextCell() gets the next cell in the list. If end of list
   * is encountered, 0 is returned. A value of 1 is returned whenever
   * npts and pts have been updated without error.
   *
   * Do not modify the returned @a pts pointer, as it may point to shared
   * memory.
   *
   * @note This method is not thread-safe and has tricky syntax to use
   * correctly. Prefer the use of svtkCellArrayIterator (see NewIterator()).
   */
  int GetNextCell(svtkIdType& npts, svtkIdType const*& pts) SVTK_SIZEHINT(pts, npts);

  /**
   * @warning This method is not thread-safe. Consider using the NewIterator()
   * iterator instead.
   *
   * GetNextCell() gets the next cell in the list. If end of list is
   * encountered, 0 is returned.
   *
   * @note This method is not thread-safe and has tricky syntax to use
   * correctly. Prefer the use of svtkCellArrayIterator (see NewIterator()).
   */
  int GetNextCell(svtkIdList* pts);

  /**
   * Return the point ids for the cell at @a cellId.
   *
   * @warning Subsequent calls to this method may invalidate previous call
   * results if the internal storage type is not the same as svtkIdType and
   * cannot be shared through the @a cellPoints pointer. In other words, the
   * method may not be thread safe. Check if shareable (using
   * IsStorageShareable()), or use a svtkCellArrayIterator to guarantee thread
   * safety.
   */
  void GetCellAtId(svtkIdType cellId, svtkIdType& cellSize, svtkIdType const*& cellPoints)
    SVTK_SIZEHINT(cellPoints, cellSize) SVTK_EXPECTS(0 <= cellId && cellId < GetNumberOfCells());

  /**
   * Return the point ids for the cell at @a cellId. This always copies
   * the cell ids (i.e., the list of points @a pts into the supplied
   * svtkIdList). This method is thread safe.
   */
  void GetCellAtId(svtkIdType cellId, svtkIdList* pts)
    SVTK_EXPECTS(0 <= cellId && cellId < GetNumberOfCells());

  /**
   * Return the size of the cell at @a cellId.
   */
  svtkIdType GetCellSize(const svtkIdType cellId) const;

  /**
   * Insert a cell object. Return the cell id of the cell.
   */
  svtkIdType InsertNextCell(svtkCell* cell);

  /**
   * Create a cell by specifying the number of points and an array of point
   * id's.  Return the cell id of the cell.
   */
  svtkIdType InsertNextCell(svtkIdType npts, const svtkIdType* pts) SVTK_SIZEHINT(pts, npts);

  /**
   * Create a cell by specifying a list of point ids. Return the cell id of
   * the cell.
   */
  svtkIdType InsertNextCell(svtkIdList* pts);

  /**
   * Overload that allows `InsertNextCell({0, 1, 2})` syntax.
   *
   * @warning This approach is useful for testing, but beware that trying to
   * pass a single value (eg. `InsertNextCell({3})`) will call the
   * `InsertNextCell(int)` overload instead.
   */
  svtkIdType InsertNextCell(const std::initializer_list<svtkIdType>& cell)
  {
    return this->InsertNextCell(static_cast<svtkIdType>(cell.size()), cell.begin());
  }

  /**
   * Create cells by specifying a count of total points to be inserted, and
   * then adding points one at a time using method InsertCellPoint(). If you
   * don't know the count initially, use the method UpdateCellCount() to
   * complete the cell. Return the cell id of the cell.
   */
  svtkIdType InsertNextCell(int npts);

  /**
   * Used in conjunction with InsertNextCell(npts) to add another point
   * to the list of cells.
   */
  void InsertCellPoint(svtkIdType id);

  /**
   * Used in conjunction with InsertNextCell(int npts) and InsertCellPoint() to
   * update the number of points defining the cell.
   */
  void UpdateCellCount(int npts);

  /**
   * Get/Set the current cellId for traversal.
   *
   * @note This method is not thread-safe and has tricky syntax to use
   * correctly. Prefer the use of svtkCellArrayIterator (see NewIterator()).
   * @{
   */
  svtkIdType GetTraversalCellId();
  void SetTraversalCellId(svtkIdType cellId);
  /**@}*/

  /**
   * Reverses the order of the point ids for the specified cell.
   */
  void ReverseCellAtId(svtkIdType cellId) SVTK_EXPECTS(0 <= cellId && cellId < GetNumberOfCells());

  /**
   * Replaces the point ids for the specified cell with the supplied list.
   *
   * @warning This can ONLY replace the cell if the size does not change.
   * Attempting to change cell size through this method will have undefined
   * results.
   * @{
   */
  void ReplaceCellAtId(svtkIdType cellId, svtkIdList* list);
  void ReplaceCellAtId(svtkIdType cellId, svtkIdType cellSize, const svtkIdType* cellPoints)
    SVTK_EXPECTS(0 <= cellId && cellId < GetNumberOfCells()) SVTK_SIZEHINT(cellPoints, cellSize);
  /**@}*/

  /**
   * Overload that allows `ReplaceCellAtId(cellId, {0, 1, 2})` syntax.
   *
   * @warning This can ONLY replace the cell if the size does not change.
   * Attempting to change cell size through this method will have undefined
   * results.
   */
  void ReplaceCellAtId(svtkIdType cellId, const std::initializer_list<svtkIdType>& cell)
  {
    return this->ReplaceCellAtId(cellId, static_cast<svtkIdType>(cell.size()), cell.begin());
  }

  /**
   * Returns the size of the largest cell. The size is the number of points
   * defining the cell.
   */
  int GetMaxCellSize();

  /**
   * Perform a deep copy (no reference counting) of the given cell array.
   */
  void DeepCopy(svtkCellArray* ca);

  /**
   * Shallow copy @a ca into this cell array.
   */
  void ShallowCopy(svtkCellArray* ca);

  /**
   * Append cells from src into this. Point ids are offset by @a pointOffset.
   */
  void Append(svtkCellArray* src, svtkIdType pointOffset = 0);

  /**
   * Fill @a data with the old-style svtkCellArray data layout, e.g.
   *
   * ```
   * { n0, p0_0, p0_1, ..., p0_n, n1, p1_0, p1_1, ..., p1_n, ... }
   * ```
   *
   * where `n0` is the number of points in cell 0, and `pX_Y` is the Y'th point
   * in cell X.
   */
  void ExportLegacyFormat(svtkIdTypeArray* data);

  /**
   * Import an array of data with the legacy svtkCellArray layout, e.g.:
   *
   * ```
   * { n0, p0_0, p0_1, ..., p0_n, n1, p1_0, p1_1, ..., p1_n, ... }
   * ```
   *
   * where `n0` is the number of points in cell 0, and `pX_Y` is the Y'th point
   * in cell X.
   * @{
   */
  void ImportLegacyFormat(svtkIdTypeArray* data);
  void ImportLegacyFormat(const svtkIdType* data, svtkIdType len) SVTK_SIZEHINT(data, len);
  /** @} */

  /**
   * Append an array of data with the legacy svtkCellArray layout, e.g.:
   *
   * ```
   * { n0, p0_0, p0_1, ..., p0_n, n1, p1_0, p1_1, ..., p1_n, ... }
   * ```
   *
   * where `n0` is the number of points in cell 0, and `pX_Y` is the Y'th point
   * in cell X.
   * @{
   */
  void AppendLegacyFormat(svtkIdTypeArray* data, svtkIdType ptOffset = 0);
  void AppendLegacyFormat(const svtkIdType* data, svtkIdType len, svtkIdType ptOffset = 0)
    SVTK_SIZEHINT(data, len);
  /** @} */

  /**
   * Return the memory in kibibytes (1024 bytes) consumed by this cell array. Used to
   * support streaming and reading/writing data. The value returned is
   * guaranteed to be greater than or equal to the memory required to
   * actually represent the data represented by this object. The
   * information returned is valid only after the pipeline has
   * been updated.
   */
  unsigned long GetActualMemorySize() const;

  // The following code is used to support

  // The wrappers get understandably confused by some of the template code below
#ifndef __SVTK_WRAP__

  // Holds connectivity and offset arrays of the given ArrayType.
  template <typename ArrayT>
  struct VisitState
  {
    using ArrayType = ArrayT;
    using ValueType = typename ArrayType::ValueType;
    using CellRangeType = decltype(svtk::DataArrayValueRange<1>(std::declval<ArrayType>()));

    // We can't just use is_same here, since binary compatible representations
    // (e.g. int and long) are distinct types. Instead, ensure that ValueType
    // is a signed integer the same size as svtkIdType.
    // If this value is true, ValueType pointers may be safely converted to
    // svtkIdType pointers via reinterpret cast.
    static constexpr bool ValueTypeIsSameAsIdType = std::is_integral<ValueType>::value &&
      std::is_signed<ValueType>::value && (sizeof(ValueType) == sizeof(svtkIdType));

    ArrayType* GetOffsets() { return this->Offsets; }
    const ArrayType* GetOffsets() const { return this->Offsets; }

    ArrayType* GetConnectivity() { return this->Connectivity; }
    const ArrayType* GetConnectivity() const { return this->Connectivity; }

    svtkIdType GetNumberOfCells() const;

    svtkIdType GetBeginOffset(svtkIdType cellId) const;

    svtkIdType GetEndOffset(svtkIdType cellId) const;

    svtkIdType GetCellSize(svtkIdType cellId) const;

    CellRangeType GetCellRange(svtkIdType cellId);

    friend class svtkCellArray;

  protected:
    VisitState()
      : Connectivity(svtkSmartPointer<ArrayType>::New())
      , Offsets(svtkSmartPointer<ArrayType>::New())
    {
      this->Offsets->InsertNextValue(0);
    }
    ~VisitState() = default;

    svtkSmartPointer<ArrayType> Connectivity;
    svtkSmartPointer<ArrayType> Offsets;

  private:
    VisitState(const VisitState&) = delete;
    VisitState& operator=(const VisitState&) = delete;
  };

private: // Helpers that allow Visit to return a value:
  template <typename Functor, typename... Args>
  using GetReturnType = decltype(
    std::declval<Functor>()(std::declval<VisitState<ArrayType32>&>(), std::declval<Args>()...));

  template <typename Functor, typename... Args>
  struct ReturnsVoid : std::is_same<GetReturnType<Functor, Args...>, void>
  {
  };

public:
  /**
   * @warning Advanced use only.
   *
   * The Visit methods allow efficient bulk modification of the svtkCellArray
   * internal arrays by dispatching a functor with the current storage arrays.
   * The simplest functor is of the form:
   *
   * ```
   * // Functor definition:
   * struct Worker
   * {
   *   template <typename CellStateT>
   *   void operator()(CellStateT &state)
   *   {
   *     // Do work on state object
   *   }
   * };
   *
   * // Functor usage:
   * svtkCellArray *cellArray = ...;
   * cellArray->Visit(Worker{});
   * ```
   *
   * where `state` is an instance of the svtkCellArray::VisitState<ArrayT> class,
   * instantiated for the current storage type of the cell array. See that
   * class for usage details.
   *
   * The functor may also:
   * - Return a value from `operator()`
   * - Pass additional arguments to `operator()`
   * - Hold state.
   *
   * A more advanced functor that does these things is shown below, along
   * with its usage. This functor scans a range of cells and returns the largest
   * cell's id:
   *
   * ```
   * struct FindLargestCellInRange
   * {
   *   template <typename CellStateT>
   *   svtkIdType operator()(CellStateT &state,
   *                        svtkIdType rangeBegin,
   *                        svtkIdType rangeEnd)
   *   {
   *     svtkIdType largest = rangeBegin;
   *     svtkIdType largestSize = state.GetCellSize(rangeBegin);
   *     ++rangeBegin;
   *     for (; rangeBegin < rangeEnd; ++rangeBegin)
   *     {
   *       const svtkIdType curSize = state.GetCellSize(rangeBegin);
   *       if (curSize > largestSize)
   *       {
   *         largest = rangeBegin;
   *         largestSize = curSize;
   *       }
   *     }
   *
   *     return largest;
   *   }
   * };
   *
   * // Usage:
   * // Scan cells in range [128, 1024) and return the id of the largest.
   * svtkCellArray cellArray = ...;
   * svtkIdType largest = cellArray->Visit(FindLargestCellInRange{},
   *                                      128, 1024);
   * ```
   * @{
   */
  template <typename Functor, typename... Args,
    typename = typename std::enable_if<ReturnsVoid<Functor, Args...>::value>::type>
  void Visit(Functor&& functor, Args&&... args)
  {
    if (this->Storage.Is64Bit())
    {
      // If you get an error on the next line, a call to Visit(functor, Args...)
      // is being called with arguments that do not match the functor's call
      // signature. See the Visit documentation for details.
      functor(this->Storage.GetArrays64(), std::forward<Args>(args)...);
    }
    else
    {
      // If you get an error on the next line, a call to Visit(functor, Args...)
      // is being called with arguments that do not match the functor's call
      // signature. See the Visit documentation for details.
      functor(this->Storage.GetArrays32(), std::forward<Args>(args)...);
    }
  }

  template <typename Functor, typename... Args,
    typename = typename std::enable_if<ReturnsVoid<Functor, Args...>::value>::type>
  void Visit(Functor&& functor, Args&&... args) const
  {
    if (this->Storage.Is64Bit())
    {
      // If you get an error on the next line, a call to Visit(functor, Args...)
      // is being called with arguments that do not match the functor's call
      // signature. See the Visit documentation for details.
      functor(this->Storage.GetArrays64(), std::forward<Args>(args)...);
    }
    else
    {
      // If you get an error on the next line, a call to Visit(functor, Args...)
      // is being called with arguments that do not match the functor's call
      // signature. See the Visit documentation for details.
      functor(this->Storage.GetArrays32(), std::forward<Args>(args)...);
    }
  }

  template <typename Functor, typename... Args,
    typename = typename std::enable_if<!ReturnsVoid<Functor, Args...>::value>::type>
  GetReturnType<Functor, Args...> Visit(Functor&& functor, Args&&... args)
  {
    if (this->Storage.Is64Bit())
    {
      // If you get an error on the next line, a call to Visit(functor, Args...)
      // is being called with arguments that do not match the functor's call
      // signature. See the Visit documentation for details.
      return functor(this->Storage.GetArrays64(), std::forward<Args>(args)...);
    }
    else
    {
      // If you get an error on the next line, a call to Visit(functor, Args...)
      // is being called with arguments that do not match the functor's call
      // signature. See the Visit documentation for details.
      return functor(this->Storage.GetArrays32(), std::forward<Args>(args)...);
    }
  }
  template <typename Functor, typename... Args,
    typename = typename std::enable_if<!ReturnsVoid<Functor, Args...>::value>::type>
  GetReturnType<Functor, Args...> Visit(Functor&& functor, Args&&... args) const
  {
    if (this->Storage.Is64Bit())
    {
      // If you get an error on the next line, a call to Visit(functor, Args...)
      // is being called with arguments that do not match the functor's call
      // signature. See the Visit documentation for details.
      return functor(this->Storage.GetArrays64(), std::forward<Args>(args)...);
    }
    else
    {
      // If you get an error on the next line, a call to Visit(functor, Args...)
      // is being called with arguments that do not match the functor's call
      // signature. See the Visit documentation for details.
      return functor(this->Storage.GetArrays32(), std::forward<Args>(args)...);
    }
  }

  /** @} */

#endif // __SVTK_WRAP__

  //=================== Begin Legacy Methods ===================================
  // These should be deprecated at some point as they are confusing or very slow

  /**
   * Set the number of cells in the array.
   * DO NOT do any kind of allocation, advanced use only.
   *
   * @note This call has no effect.
   */
  virtual void SetNumberOfCells(svtkIdType);

  /**
   * Utility routines help manage memory of cell array. EstimateSize()
   * returns a value used to initialize and allocate memory for array based
   * on number of cells and maximum number of points making up cell.  If
   * every cell is the same size (in terms of number of points), then the
   * memory estimate is guaranteed exact. (If not exact, use Squeeze() to
   * reclaim any extra memory.)
   *
   * @note This method was often misused (e.g. called alone and then
   * discarding the result). Use AllocateEstimate directly instead.
   */
  svtkIdType EstimateSize(svtkIdType numCells, int maxPtsPerCell);

  /**
   * Get the size of the allocated connectivity array.
   *
   * @warning This returns the allocated capacity of the internal arrays as a
   * number of elements, NOT the number of elements in use.
   *
   * @note Method incompatible with current internal storage.
   */
  svtkIdType GetSize();

  /**
   * Return the size of the array that would be returned from
   * ExportLegacyFormat().
   *
   * @note Method incompatible with current internal storage.
   */
  svtkIdType GetNumberOfConnectivityEntries();

  /**
   * Internal method used to retrieve a cell given a legacy offset location.
   *
   * @warning Subsequent calls to this method may invalidate previous call
   * results.
   *
   * @note The location-based API is now a super-slow compatibility layer.
   * Prefer GetCellAtId.
   */
  void GetCell(svtkIdType loc, svtkIdType& npts, const svtkIdType*& pts)
    SVTK_EXPECTS(0 <= loc && loc < GetNumberOfConnectivityEntries()) SVTK_SIZEHINT(pts, npts);

  /**
   * Internal method used to retrieve a cell given a legacy offset location.
   *
   * @note The location-based API is now a super-slow compatibility layer.
   * Prefer GetCellAtId.
   */
  void GetCell(svtkIdType loc, svtkIdList* pts)
    SVTK_EXPECTS(0 <= loc && loc < GetNumberOfConnectivityEntries());

  /**
   * Computes the current legacy insertion location within the internal array.
   * Used in conjunction with GetCell(int loc,...).
   *
   * @note The location-based API is now a super-slow compatibility layer.
   */
  svtkIdType GetInsertLocation(int npts);

  /**
   * Get/Set the current traversal legacy location.
   *
   * @note The location-based API is now a super-slow compatibility layer.
   * Prefer Get/SetTraversalCellId.
   * @{
   */
  svtkIdType GetTraversalLocation();
  svtkIdType GetTraversalLocation(svtkIdType npts);
  void SetTraversalLocation(svtkIdType loc);
  /**@}*/

  /**
   * Special method inverts ordering of cell at the specified legacy location.
   * Must be called carefully or the cell topology may be corrupted.
   *
   * @note The location-based API is now a super-slow compatibility layer.
   * Prefer ReverseCellAtId;
   */
  void ReverseCell(svtkIdType loc) SVTK_EXPECTS(0 <= loc && loc < GetNumberOfConnectivityEntries());

  /**
   * Replace the point ids of the cell at the legacy location with a different
   * list of point ids. Calling this method does not mark the svtkCellArray as
   * modified. This is the responsibility of the caller and may be done after
   * multiple calls to ReplaceCell. This call does not support changing the
   * number of points in the cell -- the caller must ensure that the target
   * cell has npts points.
   *
   * @note The location-based API is now a super-slow compatibility layer.
   * Prefer ReplaceCellAtId.
   */
  void ReplaceCell(svtkIdType loc, int npts, const svtkIdType pts[])
    SVTK_EXPECTS(0 <= loc && loc < GetNumberOfConnectivityEntries()) SVTK_SIZEHINT(pts, npts);

  /**
   * Define multiple cells by providing a connectivity list. The list is in
   * the form (npts,p0,p1,...p(npts-1), repeated for each cell). Be careful
   * using this method because it discards the old cells, and anything
   * referring these cells becomes invalid (for example, if BuildCells() has
   * been called see svtkPolyData).  The traversal location is reset to the
   * beginning of the list; the insertion location is set to the end of the
   * list.
   *
   * @warning The svtkCellArray will not hold a reference to `cells`. This
   * function merely calls ImportLegacyFormat.
   *
   * @note Use ImportLegacyFormat or SetData instead.
   */
  void SetCells(svtkIdType ncells, svtkIdTypeArray* cells);

  /**
   * Return the underlying data as a data array.
   *
   * @warning The returned array is not the actual internal representation used
   * by svtkCellArray. Modifications to the returned array will not change the
   * svtkCellArray's topology.
   *
   * @note Use ExportLegacyFormat, or GetOffsetsArray/GetConnectivityArray
   * instead.
   */
  svtkIdTypeArray* GetData();

  //=================== End Legacy Methods =====================================

  friend class svtkCellArrayIterator;

protected:
  svtkCellArray();
  ~svtkCellArray() override;

  // Encapsulates storage of the internal arrays as a discriminated union
  // between 32-bit and 64-bit storage.
  struct Storage
  {
    // Union type that switches 32 and 64 bit array storage
    union ArraySwitch {
      ArraySwitch() {}  // handled by Storage
      ~ArraySwitch() {} // handle by Storage

      VisitState<ArrayType32> Int32;
      VisitState<ArrayType64> Int64;
    };

    Storage()
    {
      // Default to the compile-time setting:
#ifdef SVTK_USE_64BIT_IDS

      new (&this->Arrays.Int64) VisitState<ArrayType64>;
      this->StorageIs64Bit = true;

#else // SVTK_USE_64BIT_IDS

      new (&this->Arrays.Int32) VisitState<ArrayType32>;
      this->StorageIs64Bit = false;

#endif // SVTK_USE_64BIT_IDS
    }

    ~Storage()
    {
      if (this->StorageIs64Bit)
      {
        this->Arrays.Int64.~VisitState();
      }
      else
      {
        this->Arrays.Int32.~VisitState();
      }
    }

    // Switch the internal arrays to be 32-bit. Any old data is lost. Returns
    // true if the storage changes.
    bool Use32BitStorage()
    {
      if (!this->StorageIs64Bit)
      {
        return false;
      }

      this->Arrays.Int64.~VisitState();
      new (&this->Arrays.Int32) VisitState<ArrayType32>;
      this->StorageIs64Bit = false;

      return true;
    }

    // Switch the internal arrays to be 64-bit. Any old data is lost. Returns
    // true if the storage changes.
    bool Use64BitStorage()
    {
      if (this->StorageIs64Bit)
      {
        return false;
      }

      this->Arrays.Int32.~VisitState();
      new (&this->Arrays.Int64) VisitState<ArrayType64>;
      this->StorageIs64Bit = true;

      return true;
    }

    // Returns true if the storage is currently configured to be 64 bit.
    bool Is64Bit() const { return this->StorageIs64Bit; }

    // Get the VisitState for 32-bit arrays
    VisitState<ArrayType32>& GetArrays32()
    {
      assert(!this->StorageIs64Bit);
      return this->Arrays.Int32;
    }

    const VisitState<ArrayType32>& GetArrays32() const
    {
      assert(!this->StorageIs64Bit);
      return this->Arrays.Int32;
    }

    // Get the VisitState for 64-bit arrays
    VisitState<ArrayType64>& GetArrays64()
    {
      assert(this->StorageIs64Bit);
      return this->Arrays.Int64;
    }

    const VisitState<ArrayType64>& GetArrays64() const
    {
      assert(this->StorageIs64Bit);
      return this->Arrays.Int64;
    }

  private:
    // Access restricted to ensure proper union construction/destruction thru
    // API.
    ArraySwitch Arrays;
    bool StorageIs64Bit;
  };

  Storage Storage;
  svtkNew<svtkIdList> TempCell;
  svtkIdType TraversalCellId{ 0 };

  svtkNew<svtkIdTypeArray> LegacyData; // For GetData().

private:
  svtkCellArray(const svtkCellArray&) = delete;
  void operator=(const svtkCellArray&) = delete;
};

template <typename ArrayT>
svtkIdType svtkCellArray::VisitState<ArrayT>::GetNumberOfCells() const
{
  return this->Offsets->GetNumberOfValues() - 1;
}

template <typename ArrayT>
svtkIdType svtkCellArray::VisitState<ArrayT>::GetBeginOffset(svtkIdType cellId) const
{
  return static_cast<svtkIdType>(this->Offsets->GetValue(cellId));
}

template <typename ArrayT>
svtkIdType svtkCellArray::VisitState<ArrayT>::GetEndOffset(svtkIdType cellId) const
{
  return static_cast<svtkIdType>(this->Offsets->GetValue(cellId + 1));
}

template <typename ArrayT>
svtkIdType svtkCellArray::VisitState<ArrayT>::GetCellSize(svtkIdType cellId) const
{
  return this->GetEndOffset(cellId) - this->GetBeginOffset(cellId);
}

template <typename ArrayT>
typename svtkCellArray::VisitState<ArrayT>::CellRangeType
svtkCellArray::VisitState<ArrayT>::GetCellRange(svtkIdType cellId)
{
  return svtk::DataArrayValueRange<1>(
    this->GetConnectivity(), this->GetBeginOffset(cellId), this->GetEndOffset(cellId));
}

namespace svtkCellArray_detail
{

struct InsertNextCellImpl
{
  // Insert full cell
  template <typename CellStateT>
  svtkIdType operator()(CellStateT& state, const svtkIdType npts, const svtkIdType pts[])
  {
    using ValueType = typename CellStateT::ValueType;
    auto* conn = state.GetConnectivity();
    auto* offsets = state.GetOffsets();

    const svtkIdType cellId = offsets->GetNumberOfValues() - 1;

    offsets->InsertNextValue(static_cast<ValueType>(conn->GetNumberOfValues() + npts));

    for (svtkIdType i = 0; i < npts; ++i)
    {
      conn->InsertNextValue(static_cast<ValueType>(pts[i]));
    }

    return cellId;
  }

  // Just update offset table (for incremental API)
  template <typename CellStateT>
  svtkIdType operator()(CellStateT& state, const svtkIdType npts)
  {
    using ValueType = typename CellStateT::ValueType;
    auto* conn = state.GetConnectivity();
    auto* offsets = state.GetOffsets();

    const svtkIdType cellId = offsets->GetNumberOfValues() - 1;

    offsets->InsertNextValue(static_cast<ValueType>(conn->GetNumberOfValues() + npts));

    return cellId;
  }
};

// for incremental API:
struct UpdateCellCountImpl
{
  template <typename CellStateT>
  void operator()(CellStateT& state, const svtkIdType npts)
  {
    using ValueType = typename CellStateT::ValueType;

    auto* offsets = state.GetOffsets();
    const ValueType cellBegin = offsets->GetValue(offsets->GetMaxId() - 1);
    offsets->SetValue(offsets->GetMaxId(), static_cast<ValueType>(cellBegin + npts));
  }
};

struct GetCellSizeImpl
{
  template <typename CellStateT>
  svtkIdType operator()(CellStateT& state, const svtkIdType cellId)
  {
    return state.GetCellSize(cellId);
  }
};

struct GetCellAtIdImpl
{
  template <typename CellStateT>
  void operator()(CellStateT& state, const svtkIdType cellId, svtkIdList* ids)
  {
    using ValueType = typename CellStateT::ValueType;

    const auto cellPts = state.GetCellRange(cellId);

    ids->SetNumberOfIds(cellPts.size());
    svtkIdType* idPtr = ids->GetPointer(0);

    for (ValueType ptId : cellPts)
    {
      *idPtr++ = static_cast<svtkIdType>(ptId);
    }
  }

  // SFINAE helper to check if a VisitState's connectivity array's memory
  // can be used as a svtkIdType*.
  template <typename CellStateT>
  struct CanShareConnPtr
  {
  private:
    using ValueType = typename CellStateT::ValueType;
    using ArrayType = typename CellStateT::ArrayType;
    using AOSArrayType = svtkAOSDataArrayTemplate<ValueType>;
    static constexpr bool ValueTypeCompat = CellStateT::ValueTypeIsSameAsIdType;
    static constexpr bool ArrayTypeCompat = std::is_base_of<AOSArrayType, ArrayType>::value;

  public:
    static constexpr bool value = ValueTypeCompat && ArrayTypeCompat;
  };

  template <typename CellStateT>
  typename std::enable_if<CanShareConnPtr<CellStateT>::value, void>::type operator()(
    CellStateT& state, const svtkIdType cellId, svtkIdType& cellSize, svtkIdType const*& cellPoints,
    svtkIdList* svtkNotUsed(temp))
  {
    const svtkIdType beginOffset = state.GetBeginOffset(cellId);
    const svtkIdType endOffset = state.GetEndOffset(cellId);
    cellSize = endOffset - beginOffset;
    // This is safe, see CanShareConnPtr helper above.
    cellPoints = reinterpret_cast<svtkIdType*>(state.GetConnectivity()->GetPointer(beginOffset));
  }

  template <typename CellStateT>
  typename std::enable_if<!CanShareConnPtr<CellStateT>::value, void>::type operator()(
    CellStateT& state, const svtkIdType cellId, svtkIdType& cellSize, svtkIdType const*& cellPoints,
    svtkIdList* temp)
  {
    using ValueType = typename CellStateT::ValueType;

    const auto cellPts = state.GetCellRange(cellId);
    cellSize = cellPts.size();

    // ValueType differs from svtkIdType, so we have to copy into a temporary
    // buffer:
    temp->SetNumberOfIds(cellSize);
    svtkIdType* tempPtr = temp->GetPointer(0);
    for (ValueType ptId : cellPts)
    {
      *tempPtr++ = static_cast<svtkIdType>(ptId);
    }

    cellPoints = temp->GetPointer(0);
  }
};

struct ResetImpl
{
  template <typename CellStateT>
  void operator()(CellStateT& state)
  {
    state.GetOffsets()->Reset();
    state.GetConnectivity()->Reset();
    state.GetOffsets()->InsertNextValue(0);
  }
};

} // end namespace svtkCellArray_detail

//----------------------------------------------------------------------------
inline void svtkCellArray::InitTraversal()
{
  this->TraversalCellId = 0;
}

//----------------------------------------------------------------------------
inline int svtkCellArray::GetNextCell(svtkIdType& npts, svtkIdType const*& pts) SVTK_SIZEHINT(pts, npts)
{
  if (this->TraversalCellId < this->GetNumberOfCells())
  {
    this->GetCellAtId(this->TraversalCellId, npts, pts);
    ++this->TraversalCellId;
    return 1;
  }

  npts = 0;
  pts = nullptr;
  return 0;
}

//----------------------------------------------------------------------------
inline int svtkCellArray::GetNextCell(svtkIdList* pts)
{
  if (this->TraversalCellId < this->GetNumberOfCells())
  {
    this->GetCellAtId(this->TraversalCellId, pts);
    ++this->TraversalCellId;
    return 1;
  }

  pts->Reset();
  return 0;
}
//----------------------------------------------------------------------------
inline svtkIdType svtkCellArray::GetCellSize(const svtkIdType cellId) const
{
  return this->Visit(svtkCellArray_detail::GetCellSizeImpl{}, cellId);
}

//----------------------------------------------------------------------------
inline void svtkCellArray::GetCellAtId(svtkIdType cellId, svtkIdType& cellSize,
  svtkIdType const*& cellPoints) SVTK_SIZEHINT(cellPoints, cellSize)
{
  this->Visit(svtkCellArray_detail::GetCellAtIdImpl{}, cellId, cellSize, cellPoints, this->TempCell);
}

//----------------------------------------------------------------------------
inline void svtkCellArray::GetCellAtId(svtkIdType cellId, svtkIdList* pts)
{
  this->Visit(svtkCellArray_detail::GetCellAtIdImpl{}, cellId, pts);
}

//----------------------------------------------------------------------------
inline svtkIdType svtkCellArray::InsertNextCell(svtkIdType npts, const svtkIdType pts[])
  SVTK_SIZEHINT(pts, npts)
{
  return this->Visit(svtkCellArray_detail::InsertNextCellImpl{}, npts, pts);
}

//----------------------------------------------------------------------------
inline svtkIdType svtkCellArray::InsertNextCell(int npts)
{
  return this->Visit(svtkCellArray_detail::InsertNextCellImpl{}, npts);
}

//----------------------------------------------------------------------------
inline void svtkCellArray::InsertCellPoint(svtkIdType id)
{
  if (this->Storage.Is64Bit())
  {
    using ValueType = typename ArrayType64::ValueType;
    this->Storage.GetArrays64().Connectivity->InsertNextValue(static_cast<ValueType>(id));
  }
  else
  {
    using ValueType = typename ArrayType32::ValueType;
    this->Storage.GetArrays32().Connectivity->InsertNextValue(static_cast<ValueType>(id));
  }
}

//----------------------------------------------------------------------------
inline void svtkCellArray::UpdateCellCount(int npts)
{
  this->Visit(svtkCellArray_detail::UpdateCellCountImpl{}, npts);
}

//----------------------------------------------------------------------------
inline svtkIdType svtkCellArray::InsertNextCell(svtkIdList* pts)
{
  return this->Visit(
    svtkCellArray_detail::InsertNextCellImpl{}, pts->GetNumberOfIds(), pts->GetPointer(0));
}

//----------------------------------------------------------------------------
inline svtkIdType svtkCellArray::InsertNextCell(svtkCell* cell)
{
  svtkIdList* pts = cell->GetPointIds();
  return this->Visit(
    svtkCellArray_detail::InsertNextCellImpl{}, pts->GetNumberOfIds(), pts->GetPointer(0));
}

//----------------------------------------------------------------------------
inline void svtkCellArray::Reset()
{
  this->Visit(svtkCellArray_detail::ResetImpl{});
}

#endif // svtkCellArray.h
