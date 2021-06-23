/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDataArray
 * @brief   abstract superclass for arrays of numeric data
 *
 *
 * svtkDataArray is an abstract superclass for data array objects
 * containing numeric data.  It extends the API defined in
 * svtkAbstractArray.  svtkDataArray is an abstract superclass for data
 * array objects. This class defines an API that all array objects
 * must support. Note that the concrete subclasses of this class
 * represent data in native form (char, int, etc.) and often have
 * specialized more efficient methods for operating on this data (for
 * example, getting pointers to data or getting/inserting data in
 * native form).  Subclasses of svtkDataArray are assumed to contain
 * data whose components are meaningful when cast to and from double.
 *
 * @sa
 * svtkBitArray svtkGenericDataArray
 */

#ifndef svtkDataArray_h
#define svtkDataArray_h

#include "svtkAbstractArray.h"
#include "svtkCommonCoreModule.h"          // For export macro
#include "svtkSVTK_USE_SCALED_SOA_ARRAYS.h" // For #define of SVTK_USE_SCALED_SOA_ARRAYS

class svtkDoubleArray;
class svtkIdList;
class svtkInformationStringKey;
class svtkInformationDoubleVectorKey;
class svtkLookupTable;
class svtkPoints;

class SVTKCOMMONCORE_EXPORT svtkDataArray : public svtkAbstractArray
{
public:
  svtkTypeMacro(svtkDataArray, svtkAbstractArray);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Perform a fast, safe cast from a svtkAbstractArray to a svtkDataArray.
   * This method checks if source->GetArrayType() returns DataArray
   * or a more derived type, and performs a static_cast to return
   * source as a svtkDataArray pointer. Otherwise, nullptr is returned.
   */
  static svtkDataArray* FastDownCast(svtkAbstractArray* source);

  /**
   * This method is here to make backward compatibility easier.  It
   * must return true if and only if an array contains numeric data.
   * All svtkDataArray subclasses contain numeric data, hence this method
   * always returns 1(true).
   */
  int IsNumeric() const override { return 1; }

  /**
   * Return the size, in bytes, of the lowest-level element of an
   * array.  For svtkDataArray and subclasses this is the size of the
   * data type.
   */
  int GetElementComponentSize() const override { return this->GetDataTypeSize(); }

  // Reimplemented virtuals (doc strings are inherited from superclass):
  void InsertTuple(svtkIdType dstTupleIdx, svtkIdType srcTupleIdx, svtkAbstractArray* source) override;
  svtkIdType InsertNextTuple(svtkIdType srcTupleIdx, svtkAbstractArray* source) override;
  void InsertTuples(svtkIdList* dstIds, svtkIdList* srcIds, svtkAbstractArray* source) override;
  void InsertTuples(
    svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkAbstractArray* source) override;
  void GetTuples(svtkIdList* tupleIds, svtkAbstractArray* output) override;
  void GetTuples(svtkIdType p1, svtkIdType p2, svtkAbstractArray* output) override;
  void InterpolateTuple(svtkIdType dstTupleIdx, svtkIdList* ptIndices, svtkAbstractArray* source,
    double* weights) override;
  void InterpolateTuple(svtkIdType dstTupleIdx, svtkIdType srcTupleIdx1, svtkAbstractArray* source1,
    svtkIdType srcTupleIdx2, svtkAbstractArray* source2, double t) override;

  /**
   * Get the data tuple at tupleIdx. Return it as a pointer to an array.
   * Note: this method is not thread-safe, and the pointer is only valid
   * as long as another method invocation to a svtk object is not performed.
   */
  virtual double* GetTuple(svtkIdType tupleIdx)
    SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples()) = 0;

  /**
   * Get the data tuple at tupleIdx by filling in a user-provided array,
   * Make sure that your array is large enough to hold the NumberOfComponents
   * amount of data being returned.
   */
  virtual void GetTuple(svtkIdType tupleIdx, double* tuple)
    SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples()) = 0;

  //@{
  /**
   * These methods are included as convenience for the wrappers.
   * GetTuple() and SetTuple() which return/take arrays can not be
   * used from wrapped languages. These methods can be used instead.
   */
  double GetTuple1(svtkIdType tupleIdx) SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples());
  double* GetTuple2(svtkIdType tupleIdx) SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples())
    SVTK_SIZEHINT(2);
  double* GetTuple3(svtkIdType tupleIdx) SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples())
    SVTK_SIZEHINT(3);
  double* GetTuple4(svtkIdType tupleIdx) SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples())
    SVTK_SIZEHINT(4);
  double* GetTuple6(svtkIdType tupleIdx) SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples())
    SVTK_SIZEHINT(6);
  double* GetTuple9(svtkIdType tupleIdx) SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples())
    SVTK_SIZEHINT(9);
  //@}

  void SetTuple(svtkIdType dstTupleIdx, svtkIdType srcTupleIdx, svtkAbstractArray* source) override;

  //@{
  /**
   * Set the data tuple at tupleIdx. Note that range checking or
   * memory allocation is not performed; use this method in conjunction
   * with SetNumberOfTuples() to allocate space.
   */
  virtual void SetTuple(svtkIdType tupleIdx, const float* tuple)
    SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples());
  virtual void SetTuple(svtkIdType tupleIdx, const double* tuple)
    SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples());
  //@}

  //@{
  /**
   * These methods are included as convenience for the wrappers.
   * GetTuple() and SetTuple() which return/take arrays can not be
   * used from wrapped languages. These methods can be used instead.
   */
  void SetTuple1(svtkIdType tupleIdx, double value)
    SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples());
  void SetTuple2(svtkIdType tupleIdx, double val0, double val1)
    SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples());
  void SetTuple3(svtkIdType tupleIdx, double val0, double val1, double val2)
    SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples());
  void SetTuple4(svtkIdType tupleIdx, double val0, double val1, double val2, double val3)
    SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples());
  void SetTuple6(svtkIdType tupleIdx, double val0, double val1, double val2, double val3,
    double val4, double val5) SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples());
  void SetTuple9(svtkIdType tupleIdx, double val0, double val1, double val2, double val3,
    double val4, double val5, double val6, double val7, double val8)
    SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples());
  //@}

  //@{
  /**
   * Insert the data tuple at tupleIdx. Note that memory allocation
   * is performed as necessary to hold the data.
   */
  virtual void InsertTuple(svtkIdType tupleIdx, const float* tuple) SVTK_EXPECTS(0 <= tupleIdx) = 0;
  virtual void InsertTuple(svtkIdType tupleIdx, const double* tuple) SVTK_EXPECTS(0 <= tupleIdx) = 0;
  //@}

  //@{
  /**
   * These methods are included as convenience for the wrappers.
   * InsertTuple() which takes arrays can not be
   * used from wrapped languages. These methods can be used instead.
   */
  void InsertTuple1(svtkIdType tupleIdx, double value) SVTK_EXPECTS(0 <= tupleIdx);
  void InsertTuple2(svtkIdType tupleIdx, double val0, double val1) SVTK_EXPECTS(0 <= tupleIdx);
  void InsertTuple3(svtkIdType tupleIdx, double val0, double val1, double val2)
    SVTK_EXPECTS(0 <= tupleIdx);
  void InsertTuple4(svtkIdType tupleIdx, double val0, double val1, double val2, double val3)
    SVTK_EXPECTS(0 <= tupleIdx);
  void InsertTuple6(svtkIdType tupleIdx, double val0, double val1, double val2, double val3,
    double val4, double val5) SVTK_EXPECTS(0 <= tupleIdx);
  void InsertTuple9(svtkIdType tupleIdx, double val0, double val1, double val2, double val3,
    double val4, double val5, double val6, double val7, double val8) SVTK_EXPECTS(0 <= tupleIdx);
  //@}

  //@{
  /**
   * Insert the data tuple at the end of the array and return the tuple index at
   * which the data was inserted. Memory is allocated as necessary to hold
   * the data.
   */
  virtual svtkIdType InsertNextTuple(const float* tuple) = 0;
  virtual svtkIdType InsertNextTuple(const double* tuple) = 0;
  //@}

  //@{
  /**
   * These methods are included as convenience for the wrappers.
   * InsertTuple() which takes arrays can not be
   * used from wrapped languages. These methods can be used instead.
   */
  void InsertNextTuple1(double value);
  void InsertNextTuple2(double val0, double val1);
  void InsertNextTuple3(double val0, double val1, double val2);
  void InsertNextTuple4(double val0, double val1, double val2, double val3);
  void InsertNextTuple6(
    double val0, double val1, double val2, double val3, double val4, double val5);
  void InsertNextTuple9(double val0, double val1, double val2, double val3, double val4,
    double val5, double val6, double val7, double val8);
  //@}

  //@{
  /**
   * These methods remove tuples from the data array. They shift data and
   * resize array, so the data array is still valid after this operation. Note,
   * this operation is fairly slow.
   */
  virtual void RemoveTuple(svtkIdType tupleIdx)
    SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples()) = 0;
  virtual void RemoveFirstTuple() { this->RemoveTuple(0); }
  virtual void RemoveLastTuple();
  //@}

  /**
   * Return the data component at the location specified by tupleIdx and
   * compIdx.
   */
  virtual double GetComponent(svtkIdType tupleIdx, int compIdx) SVTK_EXPECTS(0 <= tupleIdx &&
    tupleIdx < GetNumberOfTuples()) SVTK_EXPECTS(0 <= compIdx && compIdx < GetNumberOfComponents());

  /**
   * Set the data component at the location specified by tupleIdx and compIdx
   * to value.
   * Note that i is less than NumberOfTuples and j is less than
   * NumberOfComponents. Make sure enough memory has been allocated
   * (use SetNumberOfTuples() and SetNumberOfComponents()).
   */
  virtual void SetComponent(svtkIdType tupleIdx, int compIdx, double value)
    SVTK_EXPECTS(0 <= tupleIdx && tupleIdx < GetNumberOfTuples())
      SVTK_EXPECTS(0 <= compIdx && compIdx < GetNumberOfComponents());

  /**
   * Insert value at the location specified by tupleIdx and compIdx.
   * Note that memory allocation is performed as necessary to hold the data.
   */
  virtual void InsertComponent(svtkIdType tupleIdx, int compIdx, double value)
    SVTK_EXPECTS(0 <= tupleIdx) SVTK_EXPECTS(0 <= compIdx && compIdx < GetNumberOfComponents());

  /**
   * Get the data as a double array in the range (tupleMin,tupleMax) and
   * (compMin, compMax). The resulting double array consists of all data in
   * the tuple range specified and only the component range specified. This
   * process typically requires casting the data from native form into
   * doubleing point values. This method is provided as a convenience for data
   * exchange, and is not very fast.
   */
  virtual void GetData(
    svtkIdType tupleMin, svtkIdType tupleMax, int compMin, int compMax, svtkDoubleArray* data);

  //@{
  /**
   * Deep copy of data. Copies data from different data arrays even if
   * they are different types (using doubleing-point exchange).
   */
  void DeepCopy(svtkAbstractArray* aa) override;
  virtual void DeepCopy(svtkDataArray* da);
  //@}

  /**
   * Create a shallow copy of other into this, if possible. Shallow copies are
   * only possible:
   * (a) if both arrays are the same data type
   * (b) if both arrays are the same array type (e.g. AOS vs. SOA)
   * (c) if both arrays support shallow copies (e.g. svtkBitArray currently
   * does not.)
   * If a shallow copy is not possible, a deep copy will be performed instead.
   */
  virtual void ShallowCopy(svtkDataArray* other);

  /**
   * Fill a component of a data array with a specified value. This method
   * sets the specified component to specified value for all tuples in the
   * data array.  This methods can be used to initialize or reinitialize a
   * single component of a multi-component array.
   */
  virtual void FillComponent(int compIdx, double value)
    SVTK_EXPECTS(0 <= compIdx && compIdx < GetNumberOfComponents());

  /**
   * Fill all values of a data array with a specified value.
   */
  virtual void Fill(double value);

  /**
   * Copy a component from one data array into a component on this data array.
   * This method copies the specified component ("srcComponent") from the
   * specified data array ("src") to the specified component ("dstComponent")
   * over all the tuples in this data array.  This method can be used to extract
   * a component (column) from one data array and paste that data into
   * a component on this data array.
   */
  virtual void CopyComponent(int dstComponent, svtkDataArray* src, int srcComponent);

  /**
   * Get the address of a particular data index. Make sure data is allocated
   * for the number of items requested. If needed, increase MaxId to mark any
   * new value ranges as in-use.
   */
  virtual void* WriteVoidPointer(svtkIdType valueIdx, svtkIdType numValues) = 0;

  /**
   * Return the memory in kibibytes (1024 bytes) consumed by this data array. Used to
   * support streaming and reading/writing data. The value returned is
   * guaranteed to be greater than or equal to the memory required to
   * actually represent the data represented by this object. The
   * information returned is valid only after the pipeline has
   * been updated.
   */
  unsigned long GetActualMemorySize() const override;

  /**
   * Create default lookup table. Generally used to create one when none
   * is available.
   */
  void CreateDefaultLookupTable();

  //@{
  /**
   * Set/get the lookup table associated with this scalar data, if any.
   */
  void SetLookupTable(svtkLookupTable* lut);
  svtkGetObjectMacro(LookupTable, svtkLookupTable);
  //@}

  /**
   * The range of the data array values for the given component will be
   * returned in the provided range array argument. If comp is -1, the range
   * of the magnitude (L2 norm) over all components will be provided. The
   * range is computed and then cached, and will not be re-computed on
   * subsequent calls to GetRange() unless the array is modified or the
   * requested component changes.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  void GetRange(double range[2], int comp) { this->ComputeRange(range, comp); }

  //@{
  /**
   * Return the range of the data array values for the given component. If
   * comp is -1, return the range of the magnitude (L2 norm) over all
   * components.The range is computed and then cached, and will not be
   * re-computed on subsequent calls to GetRange() unless the array is
   * modified or the requested component changes.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  double* GetRange(int comp) SVTK_SIZEHINT(2)
  {
    this->GetRange(this->Range, comp);
    return this->Range;
  }
  //@}

  /**
   * Return the range of the data array. If the array has multiple components,
   * then this will return the range of only the first component (component
   * zero). The range is computed and then cached, and will not be re-computed
   * on subsequent calls to GetRange() unless the array is modified.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  double* GetRange() SVTK_SIZEHINT(2) { return this->GetRange(0); }

  /**
   * The range of the data array values will be returned in the provided
   * range array argument. If the data array has multiple components, then
   * this will return the range of only the first component (component zero).
   * The range is computend and then cached, and will not be re-computed on
   * subsequent calls to GetRange() unless the array is modified.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  void GetRange(double range[2]) { this->GetRange(range, 0); }

  /**
   * The range of the data array values for the given component will be
   * returned in the provided range array argument. If comp is -1, the range
   * of the magnitude (L2 norm) over all components will be provided. The
   * range is computed and then cached, and will not be re-computed on
   * subsequent calls to GetRange() unless the array is modified or the
   * requested component changes.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  void GetFiniteRange(double range[2], int comp) { this->ComputeFiniteRange(range, comp); }

  //@{
  /**
   * Return the range of the data array values for the given component. If
   * comp is -1, return the range of the magnitude (L2 norm) over all
   * components.The range is computed and then cached, and will not be
   * re-computed on subsequent calls to GetRange() unless the array is
   * modified or the requested component changes.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  double* GetFiniteRange(int comp) SVTK_SIZEHINT(2)
  {
    this->GetFiniteRange(this->FiniteRange, comp);
    return this->FiniteRange;
  }
  //@}

  /**
   * Return the range of the data array. If the array has multiple components,
   * then this will return the range of only the first component (component
   * zero). The range is computed and then cached, and will not be re-computed
   * on subsequent calls to GetRange() unless the array is modified.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  double* GetFiniteRange() SVTK_SIZEHINT(2) { return this->GetFiniteRange(0); }

  /**
   * The range of the data array values will be returned in the provided
   * range array argument. If the data array has multiple components, then
   * this will return the range of only the first component (component zero).
   * The range is computend and then cached, and will not be re-computed on
   * subsequent calls to GetRange() unless the array is modified.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  void GetFiniteRange(double range[2]) { this->GetFiniteRange(range, 0); }

  //@{
  /**
   * These methods return the Min and Max possible range of the native
   * data type. For example if a svtkScalars consists of unsigned char
   * data these will return (0,255).
   */
  void GetDataTypeRange(double range[2]);
  double GetDataTypeMin();
  double GetDataTypeMax();
  static void GetDataTypeRange(int type, double range[2]);
  static double GetDataTypeMin(int type);
  static double GetDataTypeMax(int type);
  //@}

  /**
   * Return the maximum norm for the tuples.
   * Note that the max. is computed every time GetMaxNorm is called.
   */
  virtual double GetMaxNorm();

  /**
   * Creates an array for dataType where dataType is one of
   * SVTK_BIT, SVTK_CHAR, SVTK_SIGNED_CHAR, SVTK_UNSIGNED_CHAR, SVTK_SHORT,
   * SVTK_UNSIGNED_SHORT, SVTK_INT, SVTK_UNSIGNED_INT, SVTK_LONG,
   * SVTK_UNSIGNED_LONG, SVTK_DOUBLE, SVTK_DOUBLE, SVTK_ID_TYPE.
   * Note that the data array returned has be deleted by the
   * user.
   */
  SVTK_NEWINSTANCE
  static svtkDataArray* CreateDataArray(int dataType);

  /**
   * This key is used to hold tight bounds on the range of
   * one component over all tuples of the array.
   * Two values (a minimum and maximum) are stored for each component.
   * When GetRange() is called when no tuples are present in the array
   * this value is set to { SVTK_DOUBLE_MAX, SVTK_DOUBLE_MIN }.
   */
  static svtkInformationDoubleVectorKey* COMPONENT_RANGE();

  /**
   * This key is used to hold tight bounds on the $L_2$ norm
   * of tuples in the array.
   * Two values (a minimum and maximum) are stored for each component.
   * When GetRange() is called when no tuples are present in the array
   * this value is set to { SVTK_DOUBLE_MAX, SVTK_DOUBLE_MIN }.
   */
  static svtkInformationDoubleVectorKey* L2_NORM_RANGE();

  /**
   * This key is used to hold tight bounds on the $L_2$ norm
   * of tuples in the array.
   * Two values (a minimum and maximum) are stored for each component.
   * When GetFiniteRange() is called when no tuples are present in the array
   * this value is set to { SVTK_DOUBLE_MAX, SVTK_DOUBLE_MIN }.
   */
  static svtkInformationDoubleVectorKey* L2_NORM_FINITE_RANGE();

  /**
   * Removes out-of-date L2_NORM_RANGE() and L2_NORM_FINITE_RANGE() values.
   */
  void Modified() override;

  /**
   * A human-readable string indicating the units for the array data.
   */
  static svtkInformationStringKey* UNITS_LABEL();

  /**
   * Copy information instance. Arrays use information objects
   * in a variety of ways. It is important to have flexibility in
   * this regard because certain keys should not be copied, while
   * others must be. NOTE: Up to the implmenter to make sure that
   * keys not intended to be copied are excluded here.
   */
  int CopyInformation(svtkInformation* infoFrom, int deep = 1) override;

  /**
   * Method for type-checking in FastDownCast implementations.
   */
  int GetArrayType() const override { return DataArray; }

protected:
  friend class svtkPoints;

  /**
   * Compute the range for a specific component. If comp is set -1
   * then L2 norm is computed on all components. Call ClearRange
   * to force a recomputation if it is needed. The range is copied
   * to the range argument.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  virtual void ComputeRange(double range[2], int comp);

  /**
   * Compute the range for a specific component. If comp is set -1
   * then L2 norm is computed on all components. Call ClearRange
   * to force a recomputation if it is needed. The range is copied
   * to the range argument.
   * THIS METHOD IS NOT THREAD SAFE.
   */
  virtual void ComputeFiniteRange(double range[2], int comp);

  /**
   * Computes the range for each component of an array, the length
   * of \a ranges must be two times the number of components.
   * Returns true if the range was computed. Will return false
   * if you try to compute the range of an array of length zero.
   */
  virtual bool ComputeScalarRange(double* ranges);

  /**
   * Returns true if the range was computed. Will return false
   * if you try to compute the range of an array of length zero.
   */
  virtual bool ComputeVectorRange(double range[2]);

  /**
   * Computes the range for each component of an array, the length
   * of \a ranges must be two times the number of components.
   * Returns true if the range was computed. Will return false
   * if you try to compute the range of an array of length zero.
   */
  virtual bool ComputeFiniteScalarRange(double* ranges);

  /**
   * Returns true if the range was computed. Will return false
   * if you try to compute the range of an array of length zero.
   */
  virtual bool ComputeFiniteVectorRange(double range[2]);

  // Construct object with default tuple dimension (number of components) of 1.
  svtkDataArray();
  ~svtkDataArray() override;

  svtkLookupTable* LookupTable;
  double Range[2];
  double FiniteRange[2];

private:
  double* GetTupleN(svtkIdType i, int n);

private:
  svtkDataArray(const svtkDataArray&) = delete;
  void operator=(const svtkDataArray&) = delete;
};

//------------------------------------------------------------------------------
inline svtkDataArray* svtkDataArray::FastDownCast(svtkAbstractArray* source)
{
  if (source)
  {
    switch (source->GetArrayType())
    {
      case AoSDataArrayTemplate:
      case SoADataArrayTemplate:
      case TypedDataArray:
      case DataArray:
      case MappedDataArray:
        return static_cast<svtkDataArray*>(source);
      default:
        break;
    }
  }
  return nullptr;
}

svtkArrayDownCast_FastCastMacro(svtkDataArray);

// These are used by svtkDataArrayPrivate.txx, but need to be available to
// svtkGenericDataArray.h as well.
namespace svtkDataArrayPrivate
{
struct AllValues
{
};
struct FiniteValues
{
};
}

#endif
