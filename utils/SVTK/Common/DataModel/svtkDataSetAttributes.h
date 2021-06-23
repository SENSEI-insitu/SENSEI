/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataSetAttributes.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDataSetAttributes
 * @brief   represent and manipulate attribute data in a dataset
 *
 * svtkDataSetAttributes is a class that is used to represent and manipulate
 * attribute data (e.g., scalars, vectors, normals, texture coordinates,
 * tensors, global ids, pedigree ids, and field data).
 *
 * This adds to svtkFieldData the ability to pick one of the arrays from the
 * field as the currently active array for each attribute type. In other
 * words, you pick one array to be called "THE" Scalars, and then filters down
 * the pipeline will treat that array specially. For example svtkContourFilter
 * will contour "THE" Scalar array unless a different array is asked for.
 *
 * Additionally svtkDataSetAttributes provides methods that filters call to
 * pass data through, copy data into, and interpolate from Fields. PassData
 * passes entire arrays from the source to the destination. Copy passes
 * through some subset of the tuples from the source to the destination.
 * Interpolate interpolates from the chosen tuple(s) in the source data, using
 * the provided weights, to produce new tuples in the destination.
 * Each attribute type has pass, copy and interpolate "copy" flags that
 * can be set in the destination to choose which attribute arrays will be
 * transferred from the source to the destination.
 *
 * Finally this class provides a mechanism to determine which attributes a
 * group of sources have in common, and to copy tuples from a source into
 * the destination, for only those attributes that are held by all.
 */

#ifndef svtkDataSetAttributes_h
#define svtkDataSetAttributes_h

#include "svtkCommonDataModelModule.h"      // For export macro
#include "svtkDataSetAttributesFieldList.h" // for svtkDataSetAttributesFieldList
#include "svtkFieldData.h"

class svtkLookupTable;

class SVTKCOMMONDATAMODEL_EXPORT svtkDataSetAttributes : public svtkFieldData
{
public:
  /**
   * Construct object with copying turned on for all data.
   */
  static svtkDataSetAttributes* New();

  svtkTypeMacro(svtkDataSetAttributes, svtkFieldData);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Initialize all of the object's data to nullptr
   * Also, clear the copy flags.
   */
  void Initialize() override;

  /**
   * Attributes have a chance to bring themselves up to date; right
   * now this is ignored.
   */
  virtual void Update() {}

  // -- shallow and deep copy -----------------------------------------------

  /**
   * Deep copy of data (i.e., create new data arrays and
   * copy from input data).
   * Ignores the copy flags but preserves them in the output.
   */
  void DeepCopy(svtkFieldData* pd) override;

  /**
   * Shallow copy of data (i.e., use reference counting).
   * Ignores the copy flags but preserves them in the output.
   */
  void ShallowCopy(svtkFieldData* pd) override;

  // -- attribute types -----------------------------------------------------

  // Always keep NUM_ATTRIBUTES as the last entry
  enum AttributeTypes
  {
    SCALARS = 0,
    VECTORS = 1,
    NORMALS = 2,
    TCOORDS = 3,
    TENSORS = 4,
    GLOBALIDS = 5,
    PEDIGREEIDS = 6,
    EDGEFLAG = 7,
    TANGENTS = 8,
    RATIONALWEIGHTS = 9,
    HIGHERORDERDEGREES = 10,
    NUM_ATTRIBUTES
  };

  enum AttributeLimitTypes
  {
    MAX,
    EXACT,
    NOLIMIT
  };

  // ----------- ghost points and ghost cells -------------------------------------------
  // The following bit fields are consistent with VisIt ghost zones specification
  // For details, see http://www.visitusers.org/index.php?title=Representing_ghost_data

  enum CellGhostTypes
  {
    DUPLICATECELL = 1,        // the cell is present on multiple processors
    HIGHCONNECTIVITYCELL = 2, // the cell has more neighbors than in a regular mesh
    LOWCONNECTIVITYCELL = 4,  // the cell has less neighbors than in a regular mesh
    REFINEDCELL = 8,          // other cells are present that refines it.
    EXTERIORCELL = 16,        // the cell is on the exterior of the data set
    HIDDENCELL =
      32 // the cell is needed to maintain connectivity, but the data values should be ignored.
  };

  enum PointGhostTypes
  {
    DUPLICATEPOINT = 1, // the cell is present on multiple processors
    HIDDENPOINT =
      2 // the point is needed to maintain connectivity, but the data values should be ignored.
  };

  // A svtkDataArray with this name must be of type svtkUnsignedCharArray.
  // Each value must be assigned according to the bit fields described in
  // PointGhostTypes or CellGhostType
  static const char* GhostArrayName() { return "svtkGhostType"; }

  //-----------------------------------------------------------------------------------

  //@{
  /**
   * Set/Get the scalar data.
   */
  int SetScalars(svtkDataArray* da);
  int SetActiveScalars(const char* name);
  svtkDataArray* GetScalars();
  //@}

  //@{
  /**
   * Set/Get the vector data.
   */
  int SetVectors(svtkDataArray* da);
  int SetActiveVectors(const char* name);
  svtkDataArray* GetVectors();
  //@}

  //@{
  /**
   * Set/get the normal data.
   */
  int SetNormals(svtkDataArray* da);
  int SetActiveNormals(const char* name);
  svtkDataArray* GetNormals();
  //@}

  //@{
  /**
   * Set/get the tangent data.
   */
  int SetTangents(svtkDataArray* da);
  int SetActiveTangents(const char* name);
  svtkDataArray* GetTangents();
  //@}

  //@{
  /**
   * Set/Get the texture coordinate data.
   */
  int SetTCoords(svtkDataArray* da);
  int SetActiveTCoords(const char* name);
  svtkDataArray* GetTCoords();
  //@}

  //@{
  /**
   * Set/Get the tensor data.
   */
  int SetTensors(svtkDataArray* da);
  int SetActiveTensors(const char* name);
  svtkDataArray* GetTensors();
  //@}

  //@{
  /**
   * Set/Get the global id data.
   */
  int SetGlobalIds(svtkDataArray* da);
  int SetActiveGlobalIds(const char* name);
  svtkDataArray* GetGlobalIds();
  //@}

  //@{
  /**
   * Set/Get the pedigree id data.
   */
  int SetPedigreeIds(svtkAbstractArray* da);
  int SetActivePedigreeIds(const char* name);
  svtkAbstractArray* GetPedigreeIds();
  //@}

  //@{
  /**
   * Set/Get the rational weights data.
   */
  int SetRationalWeights(svtkDataArray* da);
  int SetActiveRationalWeights(const char* name);
  svtkDataArray* GetRationalWeights();
  //@}

  //@{
  /**
   * Set/Get the rational degrees data.
   */
  int SetHigherOrderDegrees(svtkDataArray* da);
  int SetActiveHigherOrderDegrees(const char* name);
  svtkDataArray* GetHigherOrderDegrees();
  //@}

  //@{
  /**
   * This will first look for an array with the correct name.
   * If one exists, it is returned. Otherwise, the name argument
   * is ignored, and the active attribute is returned.
   */
  svtkDataArray* GetScalars(const char* name);
  svtkDataArray* GetVectors(const char* name);
  svtkDataArray* GetNormals(const char* name);
  svtkDataArray* GetTangents(const char* name);
  svtkDataArray* GetTCoords(const char* name);
  svtkDataArray* GetTensors(const char* name);
  svtkDataArray* GetGlobalIds(const char* name);
  svtkAbstractArray* GetPedigreeIds(const char* name);
  svtkDataArray* GetRationalWeights(const char* name);
  svtkDataArray* GetHigherOrderDegrees(const char* name);
  //@}

  /**
   * Make the array with the given name the active attribute.
   * Attribute types are:
   * svtkDataSetAttributes::SCALARS = 0
   * svtkDataSetAttributes::VECTORS = 1
   * svtkDataSetAttributes::NORMALS = 2
   * svtkDataSetAttributes::TCOORDS = 3
   * svtkDataSetAttributes::TENSORS = 4
   * svtkDataSetAttributes::GLOBALIDS = 5
   * svtkDataSetAttributes::PEDIGREEIDS = 6
   * svtkDataSetAttributes::EDGEFLAG = 7
   * svtkDataSetAttributes::TANGENTS = 8
   * Returns the index of the array if successful, -1 if the array
   * is not in the list of arrays.
   */
  int SetActiveAttribute(const char* name, int attributeType);

  /**
   * Make the array with the given index the active attribute.
   */
  int SetActiveAttribute(int index, int attributeType);

  /**
   * Get the field data array indices corresponding to scalars,
   * vectors, tensors, etc.
   */
  void GetAttributeIndices(int* indexArray);

  /**
   * Determine whether a data array of index idx is considered a data set
   * attribute (i.e., scalar, vector, tensor, etc). Return less-than zero
   * if it is, otherwise an index 0<=idx<NUM_ATTRIBUTES to indicate
   * which attribute.
   */
  int IsArrayAnAttribute(int idx);

  /**
   * Set an array to use as the given attribute type (i.e.,
   * svtkDataSetAttributes::SCALAR, svtkDataSetAttributes::VECTOR,
   * svtkDataSetAttributes::TENSOR, etc.). If this attribute was
   * previously set to another array, that array is removed from the
   * svtkDataSetAttributes object and the array aa is used as the
   * attribute.

   * Returns the index of aa within the svtkDataSetAttributes object
   * (i.e., the index to pass to the method GetArray(int) to obtain
   * aa) if the attribute was set to aa successfully. If aa was
   * already set as the given attributeType, returns the index of
   * aa.

   * Returns -1 in the following cases:

   * - aa is nullptr (used to unset an attribute; not an error indicator)
   * - aa is not a subclass of svtkDataArray, unless the attributeType
   * is svtkDataSetAttributes::PEDIGREEIDS (error indicator)
   * - aa has a number of components incompatible with the attribute type
   * (error indicator)
   */
  int SetAttribute(svtkAbstractArray* aa, int attributeType);

  /**
   * Return an attribute given the attribute type
   * (see svtkDataSetAttributes::AttributeTypes).
   * Some attributes (such as PEDIGREEIDS) may not be svtkDataArray subclass,
   * so in that case use GetAbstractAttribute().
   */
  svtkDataArray* GetAttribute(int attributeType);

  /**
   * Return an attribute given the attribute type
   * (see svtkDataSetAttributes::AttributeTypes).
   * This is the same as GetAttribute(), except that the returned array
   * is a svtkAbstractArray instead of svtkDataArray.
   * Some attributes (such as PEDIGREEIDS) may not be svtkDataArray subclass.
   */
  svtkAbstractArray* GetAbstractAttribute(int attributeType);

  //@{
  /**
   * Remove an array (with the given name) from the list of arrays.
   */
  using svtkFieldData::RemoveArray;
  void RemoveArray(int index) override;
  //@}

  //@{
  /**
   * Given an integer attribute type, this static method returns a string type
   * for the attribute (i.e. type = 0: returns "Scalars").
   */
  static const char* GetAttributeTypeAsString(int attributeType);
  static const char* GetLongAttributeTypeAsString(int attributeType);
  //@}

  // -- attribute copy properties ------------------------------------------

  enum AttributeCopyOperations
  {
    COPYTUPLE = 0,
    INTERPOLATE = 1,
    PASSDATA = 2,
    ALLCOPY // all of the above
  };

  /**
   * Turn on/off the copying of attribute data.
   * ctype is one of the AttributeCopyOperations, and controls copy,
   * interpolate and passdata behavior.
   * For set, ctype=ALLCOPY means set all three flags to the same value.
   * For get, ctype=ALLCOPY returns true only if all three flags are true.

   * During copying, interpolation and passdata, the following rules are
   * followed for each array:
   * 1. If the copy/interpolate/pass flag for an attribute is set (on or off),
   * it is applied. This overrides rules 2 and 3.
   * 2. If the copy flag for an array is set (on or off), it is applied
   * This overrides rule 3.
   * 3. If CopyAllOn is set, copy the array.
   * If CopyAllOff is set, do not copy the array

   * For interpolation, the flag values can be as follows:
   * 0: Do not interpolate.
   * 1: Weighted interpolation.
   * 2: Nearest neighbor interpolation.
   */
  void SetCopyAttribute(int index, int value, int ctype = ALLCOPY);

  /**
   * Get the attribute copy flag for copy operation <ctype> of attribute
   * <index>.
   */
  int GetCopyAttribute(int index, int ctype);

  /// @copydoc svtkDataSetAttributes::SetCopyAttribute()
  void SetCopyScalars(svtkTypeBool i, int ctype = ALLCOPY);
  svtkTypeBool GetCopyScalars(int ctype = ALLCOPY);
  svtkBooleanMacro(CopyScalars, svtkTypeBool);

  /// @copydoc svtkDataSetAttributes::SetCopyAttribute()
  void SetCopyVectors(svtkTypeBool i, int ctype = ALLCOPY);
  svtkTypeBool GetCopyVectors(int ctype = ALLCOPY);
  svtkBooleanMacro(CopyVectors, svtkTypeBool);

  /// @copydoc svtkDataSetAttributes::SetCopyAttribute()
  void SetCopyNormals(svtkTypeBool i, int ctype = ALLCOPY);
  svtkTypeBool GetCopyNormals(int ctype = ALLCOPY);
  svtkBooleanMacro(CopyNormals, svtkTypeBool);

  /// @copydoc svtkDataSetAttributes::SetCopyAttribute()
  void SetCopyTangents(svtkTypeBool i, int ctype = ALLCOPY);
  svtkTypeBool GetCopyTangents(int ctype = ALLCOPY);
  svtkBooleanMacro(CopyTangents, svtkTypeBool);

  /// @copydoc svtkDataSetAttributes::SetCopyAttribute()
  void SetCopyTCoords(svtkTypeBool i, int ctype = ALLCOPY);
  svtkTypeBool GetCopyTCoords(int ctype = ALLCOPY);
  svtkBooleanMacro(CopyTCoords, svtkTypeBool);

  /// @copydoc svtkDataSetAttributes::SetCopyAttribute()
  void SetCopyTensors(svtkTypeBool i, int ctype = ALLCOPY);
  svtkTypeBool GetCopyTensors(int ctype = ALLCOPY);
  svtkBooleanMacro(CopyTensors, svtkTypeBool);

  /// @copydoc svtkDataSetAttributes::SetCopyAttribute()
  void SetCopyGlobalIds(svtkTypeBool i, int ctype = ALLCOPY);
  svtkTypeBool GetCopyGlobalIds(int ctype = ALLCOPY);
  svtkBooleanMacro(CopyGlobalIds, svtkTypeBool);

  /// @copydoc svtkDataSetAttributes::SetCopyAttribute()
  void SetCopyPedigreeIds(svtkTypeBool i, int ctype = ALLCOPY);
  svtkTypeBool GetCopyPedigreeIds(int ctype = ALLCOPY);
  svtkBooleanMacro(CopyPedigreeIds, svtkTypeBool);

  /// @copydoc svtkDataSetAttributes::SetCopyAttribute()
  void SetCopyRationalWeights(svtkTypeBool i, int ctype = ALLCOPY);
  svtkTypeBool GetCopyRationalWeights(int ctype = ALLCOPY);
  svtkBooleanMacro(CopyRationalWeights, svtkTypeBool);

  /// @copydoc svtkDataSetAttributes::SetCopyAttribute()
  void SetCopyHigherOrderDegrees(svtkTypeBool i, int ctype = ALLCOPY);
  svtkTypeBool GetCopyHigherOrderDegrees(int ctype = ALLCOPY);
  svtkBooleanMacro(CopyHigherOrderDegrees, svtkTypeBool);

  /// @copydoc svtkDataSetAttributes::SetCopyAttribute()
  void CopyAllOn(int ctype = ALLCOPY) override;

  /// @copydoc svtkDataSetAttributes::SetCopyAttribute()
  void CopyAllOff(int ctype = ALLCOPY) override;

  // -- passthrough operations ----------------------------------------------

  /**
   * Pass entire arrays of input data through to output. Obey the "copy"
   * flags. When passing a field,  the following copying rules are
   * followed: 1) Check if a field is an attribute, if yes and if there
   * is a PASSDATA copy flag for that attribute (on or off), obey the flag
   * for that attribute, ignore (2) and (3), 2) if there is a copy field for
   * that field (on or off), obey the flag, ignore (3) 3) obey
   * CopyAllOn/Off
   */
  void PassData(svtkFieldData* fd) override;

  // -- copytuple operations ------------------------------------------------

  //@{
  /**
   * Allocates point data for point-by-point (or cell-by-cell) copy operation.
   * If sze=0, then use the input DataSetAttributes to create (i.e., find
   * initial size of) new objects; otherwise use the sze variable.
   * Note that pd HAS to be the svtkDataSetAttributes object which
   * will later be used with CopyData. If this is not the case,
   * consider using the alternative forms of CopyAllocate and CopyData.
   * ext is no longer used.
   * If shallowCopyArrays is true, input arrays are copied to the output
   * instead of new ones being allocated.
   */
  void CopyAllocate(svtkDataSetAttributes* pd, svtkIdType sze = 0, svtkIdType ext = 1000)
  {
    this->CopyAllocate(pd, sze, ext, 0);
  }
  void CopyAllocate(svtkDataSetAttributes* pd, svtkIdType sze, svtkIdType ext, int shallowCopyArrays);
  //@}

  /**
   * Create a mapping between the input attributes and this object
   * so that methods like CopyData() and CopyStructuredData()
   * can be called. This method assumes that this object has the
   * same arrays as the input and that they are ordered the same
   * way (same array indices).
   */
  void SetupForCopy(svtkDataSetAttributes* pd);

  /**
   * This method is used to copy data arrays in images.
   * You should call CopyAllocate or SetupForCopy before
   * calling this method. If setSize is true, this method
   * will set the size of the output arrays according to
   * the output extent. This is required when CopyAllocate()
   * was used to setup output arrays.
   */
  void CopyStructuredData(
    svtkDataSetAttributes* inDsa, const int* inExt, const int* outExt, bool setSize = true);

  //@{
  /**
   * Copy the attribute data from one id to another. Make sure CopyAllocate()
   * has been invoked before using this method. When copying a field,
   * the following copying rules are
   * followed: 1) Check if a field is an attribute, if yes and if there
   * is a COPYTUPLE copy flag for that attribute (on or off), obey the flag
   * for that attribute, ignore (2) and (3), 2) if there is a copy field for
   * that field (on or off), obey the flag, ignore (3) 3) obey
   * CopyAllOn/Off
   */
  void CopyData(svtkDataSetAttributes* fromPd, svtkIdType fromId, svtkIdType toId);
  void CopyData(svtkDataSetAttributes* fromPd, svtkIdList* fromIds, svtkIdList* toIds);
  //@}

  /**
   * Copy n consecutive attributes starting at srcStart from fromPd to this
   * container, starting at the dstStart location.
   * Note that memory allocation is performed as necessary to hold the data.
   */
  void CopyData(svtkDataSetAttributes* fromPd, svtkIdType dstStart, svtkIdType n, svtkIdType srcStart);

  //@{
  /**
   * Copy a tuple (or set of tuples) of data from one data array to another.
   * This method assumes that the fromData and toData objects are of the
   * same type, and have the same number of components. This is true if you
   * invoke CopyAllocate() or InterpolateAllocate().
   */
  void CopyTuple(
    svtkAbstractArray* fromData, svtkAbstractArray* toData, svtkIdType fromId, svtkIdType toId);
  void CopyTuples(
    svtkAbstractArray* fromData, svtkAbstractArray* toData, svtkIdList* fromIds, svtkIdList* toIds);
  void CopyTuples(svtkAbstractArray* fromData, svtkAbstractArray* toData, svtkIdType dstStart,
    svtkIdType n, svtkIdType srcStart);
  //@}

  // -- interpolate operations ----------------------------------------------

  //@{
  /**
   * Initialize point interpolation method.
   * Note that pd HAS to be the svtkDataSetAttributes object which
   * will later be used with InterpolatePoint or InterpolateEdge.
   * ext is no longer used.
   * If shallowCopyArrays is true, input arrays are copied to the output
   * instead of new ones being allocated.
   */
  void InterpolateAllocate(svtkDataSetAttributes* pd, svtkIdType sze = 0, svtkIdType ext = 1000)
  {
    this->InterpolateAllocate(pd, sze, ext, 0);
  }
  void InterpolateAllocate(
    svtkDataSetAttributes* pd, svtkIdType sze, svtkIdType ext, int shallowCopyArrays);
  //@}

  /**
   * Interpolate data set attributes from other data set attributes
   * given cell or point ids and associated interpolation weights.
   * If the INTERPOLATION copy flag is set to 0 for an array, interpolation
   * is prevented. If the flag is set to 1, weighted interpolation occurs.
   * If the flag is set to 2, nearest neighbor interpolation is used.
   */
  void InterpolatePoint(
    svtkDataSetAttributes* fromPd, svtkIdType toId, svtkIdList* ids, double* weights);

  /**
   * Interpolate data from the two points p1,p2 (forming an edge) and an
   * interpolation factor, t, along the edge. The weight ranges from (0,1),
   * with t=0 located at p1. Make sure that the method InterpolateAllocate()
   * has been invoked before using this method.
   * If the INTERPOLATION copy flag is set to 0 for an array, interpolation
   * is prevented. If the flag is set to 1, weighted interpolation occurs.
   * If the flag is set to 2, nearest neighbor interpolation is used.
   */
  void InterpolateEdge(
    svtkDataSetAttributes* fromPd, svtkIdType toId, svtkIdType p1, svtkIdType p2, double t);

  /**
   * Interpolate data from the same id (point or cell) at different points
   * in time (parameter t). Two input data set attributes objects are input.
   * The parameter t lies between (0<=t<=1). IMPORTANT: it is assumed that
   * the number of attributes and number of components is the same for both
   * from1 and from2, and the type of data for from1 and from2 are the same.
   * Make sure that the method InterpolateAllocate() has been invoked before
   * using this method.
   * If the INTERPOLATION copy flag is set to 0 for an array, interpolation
   * is prevented. If the flag is set to 1, weighted interpolation occurs.
   * If the flag is set to 2, nearest neighbor interpolation is used.
   */
  void InterpolateTime(
    svtkDataSetAttributes* from1, svtkDataSetAttributes* from2, svtkIdType id, double t);

  using FieldList = svtkDataSetAttributesFieldList;

  // field list copy operations ------------------------------------------

  /**
   * A special form of CopyAllocate() to be used with FieldLists. Use it
   * when you are copying data from a set of svtkDataSetAttributes.
   */
  void CopyAllocate(svtkDataSetAttributes::FieldList& list, svtkIdType sze = 0, svtkIdType ext = 1000);

  /**
   * Special forms of CopyData() to be used with FieldLists. Use it when
   * you are copying data from a set of svtkDataSetAttributes. Make sure
   * that you have called the special form of CopyAllocate that accepts
   * FieldLists.
   */
  void CopyData(svtkDataSetAttributes::FieldList& list, svtkDataSetAttributes* dsa, int idx,
    svtkIdType fromId, svtkIdType toId);
  void CopyData(svtkDataSetAttributes::FieldList& list, svtkDataSetAttributes* dsa, int idx,
    svtkIdType dstStart, svtkIdType n, svtkIdType srcStart);

  /**
   * A special form of InterpolateAllocate() to be used with FieldLists. Use it
   * when you are interpolating data from a set of svtkDataSetAttributes.
   * \c Warning: This does not copy the Information object associated with
   * each data array. This behavior may change in the future.
   */
  void InterpolateAllocate(
    svtkDataSetAttributes::FieldList& list, svtkIdType sze = 0, svtkIdType ext = 1000);

  /**
   * Interpolate data set attributes from other data set attributes
   * given cell or point ids and associated interpolation weights.
   * Make sure that special form of InterpolateAllocate() that accepts
   * FieldList has been used.
   */
  void InterpolatePoint(svtkDataSetAttributes::FieldList& list, svtkDataSetAttributes* fromPd,
    int idx, svtkIdType toId, svtkIdList* ids, double* weights);

protected:
  svtkDataSetAttributes();
  ~svtkDataSetAttributes() override;

  void InternalCopyAllocate(svtkDataSetAttributes* pd, int ctype, svtkIdType sze = 0,
    svtkIdType ext = 1000, int shallowCopyArrays = 0, bool createNewArrays = true);

  /**
   * Initialize all of the object's data to nullptr
   */
  void InitializeFields() override;

  int AttributeIndices[NUM_ATTRIBUTES];            // index to attribute array in field data
  int CopyAttributeFlags[ALLCOPY][NUM_ATTRIBUTES]; // copy flag for attribute data

  svtkFieldData::BasicIterator RequiredArrays;

  int* TargetIndices;

  static const int NumberOfAttributeComponents[NUM_ATTRIBUTES];
  static const int AttributeLimits[NUM_ATTRIBUTES];
  static const char AttributeNames[NUM_ATTRIBUTES][19];
  static const char LongAttributeNames[NUM_ATTRIBUTES][42];

private:
  static int CheckNumberOfComponents(svtkAbstractArray* da, int attributeType);

  svtkFieldData::BasicIterator ComputeRequiredArrays(svtkDataSetAttributes* pd, int ctype);

private:
  svtkDataSetAttributes(const svtkDataSetAttributes&) = delete;
  void operator=(const svtkDataSetAttributes&) = delete;

  friend class svtkDataSetAttributesFieldList;
};

#endif
