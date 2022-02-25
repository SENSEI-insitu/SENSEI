/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataObject.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDataObject
 * @brief   general representation of visualization data
 *
 * svtkDataObject is an general representation of visualization data. It serves
 * to encapsulate instance variables and methods for visualization network
 * execution, as well as representing data consisting of a field (i.e., just
 * an unstructured pile of data). This is to be compared with a svtkDataSet,
 * which is data with geometric and/or topological structure.
 *
 * svtkDataObjects are used to represent arbitrary repositories of data via the
 * svtkFieldData instance variable. These data must be eventually mapped into a
 * concrete subclass of svtkDataSet before they can actually be displayed.
 *
 * @sa
 * svtkDataSet svtkFieldData svtkDataObjectToDataSetFilter
 * svtkFieldDataToAttributeDataFilter
 */

#ifndef svtkDataObject_h
#define svtkDataObject_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkAbstractArray;
class svtkDataArray;
class svtkDataSetAttributes;
class svtkFieldData;
class svtkInformation;
class svtkInformationDataObjectKey;
class svtkInformationDoubleKey;
class svtkInformationDoubleVectorKey;
class svtkInformationIntegerKey;
class svtkInformationIntegerPointerKey;
class svtkInformationIntegerVectorKey;
class svtkInformationStringKey;
class svtkInformationVector;
class svtkInformationInformationVectorKey;

#define SVTK_PIECES_EXTENT 0
#define SVTK_3D_EXTENT 1
#define SVTK_TIME_EXTENT 2

class SVTKCOMMONDATAMODEL_EXPORT svtkDataObject : public svtkObject
{
public:
  static svtkDataObject* New();

  svtkTypeMacro(svtkDataObject, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Set/Get the information object associated with this data object.
   */
  svtkGetObjectMacro(Information, svtkInformation);
  virtual void SetInformation(svtkInformation*);
  //@}

  /**
   * Data objects are composite objects and need to check each part for MTime.
   * The information object also needs to be considered.
   */
  svtkMTimeType GetMTime() override;

  /**
   * Restore data object to initial state,
   */
  virtual void Initialize();

  /**
   * Release data back to system to conserve memory resource. Used during
   * visualization network execution.  Releasing this data does not make
   * down-stream data invalid.
   */
  void ReleaseData();

  //@{
  /**
   * Get the flag indicating the data has been released.
   */
  svtkGetMacro(DataReleased, int);
  //@}

  //@{
  /**
   * Turn on/off flag to control whether every object releases its data
   * after being used by a filter.
   */
  static void SetGlobalReleaseDataFlag(int val);
  void GlobalReleaseDataFlagOn() { this->SetGlobalReleaseDataFlag(1); }
  void GlobalReleaseDataFlagOff() { this->SetGlobalReleaseDataFlag(0); }
  static int GetGlobalReleaseDataFlag();
  //@}

  //@{
  /**
   * Assign or retrieve a general field data to this data object.
   */
  virtual void SetFieldData(svtkFieldData*);
  svtkGetObjectMacro(FieldData, svtkFieldData);
  //@}

  /**
   * Return class name of data type. This is one of SVTK_STRUCTURED_GRID,
   * SVTK_STRUCTURED_POINTS, SVTK_UNSTRUCTURED_GRID, SVTK_POLY_DATA, or
   * SVTK_RECTILINEAR_GRID (see svtkSetGet.h for definitions).
   * THIS METHOD IS THREAD SAFE
   */
  virtual int GetDataObjectType() { return SVTK_DATA_OBJECT; }

  /**
   * Used by Threaded ports to determine if they should initiate an
   * asynchronous update (still in development).
   */
  svtkMTimeType GetUpdateTime();

  /**
   * Return the actual size of the data in kibibytes (1024 bytes). This number
   * is valid only after the pipeline has updated. The memory size
   * returned is guaranteed to be greater than or equal to the
   * memory required to represent the data (e.g., extra space in
   * arrays, etc. are not included in the return value).
   */
  virtual unsigned long GetActualMemorySize();

  /**
   * Copy from the pipeline information to the data object's own information.
   * Called right before the main execution pass.
   */
  virtual void CopyInformationFromPipeline(svtkInformation* svtkNotUsed(info)) {}

  /**
   * Copy information from this data object to the pipeline information.
   * This is used by the svtkTrivialProducer that is created when someone
   * calls SetInputData() to connect a data object to a pipeline.
   */
  virtual void CopyInformationToPipeline(svtkInformation* svtkNotUsed(info)) {}

  /**
   * Return the information object within the input information object's
   * field data corresponding to the specified association
   * (FIELD_ASSOCIATION_POINTS or FIELD_ASSOCIATION_CELLS) and attribute
   * (SCALARS, VECTORS, NORMALS, TCOORDS, or TENSORS)
   */
  static svtkInformation* GetActiveFieldInformation(
    svtkInformation* info, int fieldAssociation, int attributeType);

  /**
   * Return the information object within the input information object's
   * field data corresponding to the specified association
   * (FIELD_ASSOCIATION_POINTS or FIELD_ASSOCIATION_CELLS) and name.
   */
  static svtkInformation* GetNamedFieldInformation(
    svtkInformation* info, int fieldAssociation, const char* name);

  /**
   * Remove the info associated with an array
   */
  static void RemoveNamedFieldInformation(
    svtkInformation* info, int fieldAssociation, const char* name);

  /**
   * Set the named array to be the active field for the specified type
   * (SCALARS, VECTORS, NORMALS, TCOORDS, or TENSORS) and association
   * (FIELD_ASSOCIATION_POINTS or FIELD_ASSOCIATION_CELLS).  Returns the
   * active field information object and creates on entry if one not found.
   */
  static svtkInformation* SetActiveAttribute(
    svtkInformation* info, int fieldAssociation, const char* attributeName, int attributeType);

  /**
   * Set the name, array type, number of components, and number of tuples
   * within the passed information object for the active attribute of type
   * attributeType (in specified association, FIELD_ASSOCIATION_POINTS or
   * FIELD_ASSOCIATION_CELLS).  If there is not an active attribute of the
   * specified type, an entry in the information object is created.  If
   * arrayType, numComponents, or numTuples equal to -1, or name=nullptr the
   * value is not changed.
   */
  static void SetActiveAttributeInfo(svtkInformation* info, int fieldAssociation, int attributeType,
    const char* name, int arrayType, int numComponents, int numTuples);

  /**
   * Convenience version of previous method for use (primarily) by the Imaging
   * filters. If arrayType or numComponents == -1, the value is not changed.
   */
  static void SetPointDataActiveScalarInfo(svtkInformation* info, int arrayType, int numComponents);

  /**
   * This method is called by the source when it executes to generate data.
   * It is sort of the opposite of ReleaseData.
   * It sets the DataReleased flag to 0, and sets a new UpdateTime.
   */
  void DataHasBeenGenerated();

  /**
   * make the output data ready for new data to be inserted. For most
   * objects we just call Initialize. But for svtkImageData we leave the old
   * data in case the memory can be reused.
   */
  virtual void PrepareForNewData() { this->Initialize(); }

  //@{
  /**
   * Shallow and Deep copy.  These copy the data, but not any of the
   * pipeline connections.
   */
  virtual void ShallowCopy(svtkDataObject* src);
  virtual void DeepCopy(svtkDataObject* src);
  //@}

  /**
   * The ExtentType will be left as SVTK_PIECES_EXTENT for data objects
   * such as svtkPolyData and svtkUnstructuredGrid. The ExtentType will be
   * changed to SVTK_3D_EXTENT for data objects with 3D structure such as
   * svtkImageData (and its subclass svtkStructuredPoints), svtkRectilinearGrid,
   * and svtkStructuredGrid. The default is the have an extent in pieces,
   * with only one piece (no streaming possible).
   */
  virtual int GetExtentType() { return SVTK_PIECES_EXTENT; }

  /**
   * This method crops the data object (if necessary) so that the extent
   * matches the update extent.
   */
  virtual void Crop(const int* updateExtent);

  /**
   * Possible values for the FIELD_ASSOCIATION information entry.
   */
  enum FieldAssociations
  {
    FIELD_ASSOCIATION_POINTS,
    FIELD_ASSOCIATION_CELLS,
    FIELD_ASSOCIATION_NONE,
    FIELD_ASSOCIATION_POINTS_THEN_CELLS,
    FIELD_ASSOCIATION_VERTICES,
    FIELD_ASSOCIATION_EDGES,
    FIELD_ASSOCIATION_ROWS,
    NUMBER_OF_ASSOCIATIONS
  };

  /**
   * Possible attribute types.
   * POINT_THEN_CELL is provided for consistency with FieldAssociations.
   */
  enum AttributeTypes
  {
    POINT,
    CELL,
    FIELD,
    POINT_THEN_CELL,
    VERTEX,
    EDGE,
    ROW,
    NUMBER_OF_ATTRIBUTE_TYPES
  };

  /**
   * Returns the attributes of the data object of the specified
   * attribute type. The type may be:
   * <ul>
   * <li>POINT  - Defined in svtkDataSet subclasses.
   * <li>CELL   - Defined in svtkDataSet subclasses.
   * <li>VERTEX - Defined in svtkGraph subclasses.
   * <li>EDGE   - Defined in svtkGraph subclasses.
   * <li>ROW    - Defined in svtkTable.
   * </ul>
   * The other attribute type, FIELD, will return nullptr since
   * field data is stored as a svtkFieldData instance, not a
   * svtkDataSetAttributes instance. To retrieve field data, use
   * GetAttributesAsFieldData.
   *
   * @warning This method NEEDS to be
   * overriden in subclasses to work as documented.
   * If not, it returns nullptr for any type but FIELD.
   */
  virtual svtkDataSetAttributes* GetAttributes(int type);

  /**
   * Returns the ghost arrays of the data object of the specified
   * atribute type. The type may be:
   * <ul>
   * <li>POINT    - Defined in svtkDataSet subclasses
   * <li>CELL   - Defined in svtkDataSet subclasses.
   * </ul>
   * The other attribute types, will return nullptr since
   * ghosts arrays are not defined for now outside of
   * point or cell.
   */
  virtual svtkDataArray* GetGhostArray(int type);

  /**
   * Returns the attributes of the data object as a svtkFieldData.
   * This returns non-null values in all the same cases as GetAttributes,
   * in addition to the case of FIELD, which will return the field data
   * for any svtkDataObject subclass.
   */
  virtual svtkFieldData* GetAttributesAsFieldData(int type);

  /**
   * Retrieves the attribute type that an array came from.
   * This is useful for obtaining which attribute type a input array
   * to an algorithm came from (retrieved from GetInputAbstractArrayToProcesss).
   */
  virtual int GetAttributeTypeForArray(svtkAbstractArray* arr);

  /**
   * Get the number of elements for a specific attribute type (POINT, CELL, etc.).
   */
  virtual svtkIdType GetNumberOfElements(int type);

  /**
   * Possible values for the FIELD_OPERATION information entry.
   */
  enum FieldOperations
  {
    FIELD_OPERATION_PRESERVED,
    FIELD_OPERATION_REINTERPOLATED,
    FIELD_OPERATION_MODIFIED,
    FIELD_OPERATION_REMOVED
  };

  /**
   * Given an integer association type, this static method returns a string type
   * for the attribute (i.e. type = 0: returns "Points").
   */
  static const char* GetAssociationTypeAsString(int associationType);

  /**
   * Given an integer association type, this static method returns a string type
   * for the attribute (i.e. type = 0: returns "Points").
   */
  static int GetAssociationTypeFromString(const char* associationType);

  // \ingroup InformationKeys
  static svtkInformationStringKey* DATA_TYPE_NAME();
  // \ingroup InformationKeys
  static svtkInformationDataObjectKey* DATA_OBJECT();
  // \ingroup InformationKeys
  static svtkInformationIntegerKey* DATA_EXTENT_TYPE();
  // \ingroup InformationKeys
  static svtkInformationIntegerPointerKey* DATA_EXTENT();
  // \ingroup InformationKeys
  static svtkInformationIntegerVectorKey* ALL_PIECES_EXTENT();
  // \ingroup InformationKeys
  static svtkInformationIntegerKey* DATA_PIECE_NUMBER();
  // \ingroup InformationKeys
  static svtkInformationIntegerKey* DATA_NUMBER_OF_PIECES();
  // \ingroup InformationKeys
  static svtkInformationIntegerKey* DATA_NUMBER_OF_GHOST_LEVELS();
  // \ingroup InformationKeys
  static svtkInformationDoubleKey* DATA_TIME_STEP();
  // \ingroup InformationKeys
  static svtkInformationInformationVectorKey* POINT_DATA_VECTOR();
  // \ingroup InformationKeys
  static svtkInformationInformationVectorKey* CELL_DATA_VECTOR();
  // \ingroup InformationKeys
  static svtkInformationInformationVectorKey* VERTEX_DATA_VECTOR();
  // \ingroup InformationKeys
  static svtkInformationInformationVectorKey* EDGE_DATA_VECTOR();
  // \ingroup InformationKeys
  static svtkInformationIntegerKey* FIELD_ARRAY_TYPE();
  // \ingroup InformationKeys
  static svtkInformationIntegerKey* FIELD_ASSOCIATION();
  // \ingroup InformationKeys
  static svtkInformationIntegerKey* FIELD_ATTRIBUTE_TYPE();
  // \ingroup InformationKeys
  static svtkInformationIntegerKey* FIELD_ACTIVE_ATTRIBUTE();
  // \ingroup InformationKeys
  static svtkInformationIntegerKey* FIELD_NUMBER_OF_COMPONENTS();
  // \ingroup InformationKeys
  static svtkInformationIntegerKey* FIELD_NUMBER_OF_TUPLES();
  // \ingroup InformationKeys
  static svtkInformationIntegerKey* FIELD_OPERATION();
  // \ingroup InformationKeys
  static svtkInformationDoubleVectorKey* FIELD_RANGE();
  // \ingroup InformationKeys
  static svtkInformationIntegerVectorKey* PIECE_EXTENT();
  // \ingroup InformationKeys
  static svtkInformationStringKey* FIELD_NAME();
  // \ingroup InformationKeys
  static svtkInformationDoubleVectorKey* ORIGIN();
  // \ingroup InformationKeys
  static svtkInformationDoubleVectorKey* SPACING();
  // \ingroup InformationKeys
  static svtkInformationDoubleVectorKey* DIRECTION();
  // \ingroup InformationKeys
  static svtkInformationDoubleVectorKey* BOUNDING_BOX();

  // Key used to put SIL information in the output information by readers.
  // \ingroup InformationKeys
  static svtkInformationDataObjectKey* SIL();

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkDataObject* GetData(svtkInformation* info);
  static svtkDataObject* GetData(svtkInformationVector* v, int i = 0);
  //@}

protected:
  svtkDataObject();
  ~svtkDataObject() override;

  // General field data associated with data object
  svtkFieldData* FieldData;

  // Keep track of data release during network execution
  int DataReleased;

  // When was this data last generated?
  svtkTimeStamp UpdateTime;

  // Arbitrary extra information associated with this data object.
  svtkInformation* Information;

private:
  // Helper method for the ShallowCopy and DeepCopy methods.
  void InternalDataObjectCopy(svtkDataObject* src);

private:
  svtkDataObject(const svtkDataObject&) = delete;
  void operator=(const svtkDataObject&) = delete;
};

#endif
