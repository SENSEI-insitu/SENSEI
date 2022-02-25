/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImageData.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkImageData
 * @brief   topologically and geometrically regular array of data
 *
 * svtkImageData is a data object that is a concrete implementation of
 * svtkDataSet. svtkImageData represents a geometric structure that is
 * a topological and geometrical regular array of points. Examples include
 * volumes (voxel data) and pixmaps.
 */

#ifndef svtkImageData_h
#define svtkImageData_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataSet.h"

#include "svtkStructuredData.h" // Needed for inline methods

class svtkDataArray;
class svtkLine;
class svtkMatrix3x3;
class svtkMatrix4x4;
class svtkPixel;
class svtkVertex;
class svtkVoxel;

class SVTKCOMMONDATAMODEL_EXPORT svtkImageData : public svtkDataSet
{
public:
  static svtkImageData* New();

  svtkTypeMacro(svtkImageData, svtkDataSet);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Copy the geometric and topological structure of an input image data
   * object.
   */
  void CopyStructure(svtkDataSet* ds) override;

  /**
   * Return what type of dataset this is.
   */
  int GetDataObjectType() override { return SVTK_IMAGE_DATA; }

  //@{
  /**
   * Standard svtkDataSet API methods. See svtkDataSet for more information.
   * \warning If GetCell(int,int,int) gets overridden in a subclass, it is
   * necessary to override GetCell(svtkIdType) in that class as well since
   * svtkImageData::GetCell(svtkIdType) will always call
   * vkImageData::GetCell(int,int,int)
   */
  svtkIdType GetNumberOfCells() override;
  svtkIdType GetNumberOfPoints() override;
  double* GetPoint(svtkIdType ptId) SVTK_SIZEHINT(3) override;
  void GetPoint(svtkIdType id, double x[3]) override;
  svtkCell* GetCell(svtkIdType cellId) override;
  svtkCell* GetCell(int i, int j, int k) override;
  void GetCell(svtkIdType cellId, svtkGenericCell* cell) override;
  void GetCellBounds(svtkIdType cellId, double bounds[6]) override;
  virtual svtkIdType FindPoint(double x, double y, double z)
  {
    return this->svtkDataSet::FindPoint(x, y, z);
  }
  svtkIdType FindPoint(double x[3]) override;
  svtkIdType FindCell(double x[3], svtkCell* cell, svtkIdType cellId, double tol2, int& subId,
    double pcoords[3], double* weights) override;
  svtkIdType FindCell(double x[3], svtkCell* cell, svtkGenericCell* gencell, svtkIdType cellId,
    double tol2, int& subId, double pcoords[3], double* weights) override;
  svtkCell* FindAndGetCell(double x[3], svtkCell* cell, svtkIdType cellId, double tol2, int& subId,
    double pcoords[3], double* weights) override;
  int GetCellType(svtkIdType cellId) override;
  void GetCellPoints(svtkIdType cellId, svtkIdList* ptIds) override
  {
    svtkStructuredData::GetCellPoints(cellId, ptIds, this->DataDescription, this->GetDimensions());
  }
  void GetPointCells(svtkIdType ptId, svtkIdList* cellIds) override
  {
    svtkStructuredData::GetPointCells(ptId, cellIds, this->GetDimensions());
  }
  void ComputeBounds() override;
  int GetMaxCellSize() override { return 8; } // voxel is the largest
  //@}

  /**
   * Restore data object to initial state.
   */
  void Initialize() override;

  /**
   * Same as SetExtent(0, i-1, 0, j-1, 0, k-1)
   */
  virtual void SetDimensions(int i, int j, int k);

  /**
   * Same as SetExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)
   */
  virtual void SetDimensions(const int dims[3]);

  /**
   * Get dimensions of this structured points dataset.
   * It is the number of points on each axis.
   * Dimensions are computed from Extents during this call.
   * \warning Non thread-safe, use second signature if you want it to be.
   */
  virtual int* GetDimensions() SVTK_SIZEHINT(3);

  /**
   * Get dimensions of this structured points dataset.
   * It is the number of points on each axis.
   * This method is thread-safe.
   * \warning The Dimensions member variable is not updated during this call.
   */
  virtual void GetDimensions(int dims[3]);
#if SVTK_ID_TYPE_IMPL != SVTK_INT
  virtual void GetDimensions(svtkIdType dims[3]);
#endif

  /**
   * Convenience function computes the structured coordinates for a point x[3].
   * The voxel is specified by the array ijk[3], and the parametric coordinates
   * in the cell are specified with pcoords[3]. The function returns a 0 if the
   * point x is outside of the volume, and a 1 if inside the volume.
   */
  virtual int ComputeStructuredCoordinates(const double x[3], int ijk[3], double pcoords[3]);

  /**
   * Given structured coordinates (i,j,k) for a voxel cell, compute the eight
   * gradient values for the voxel corners. The order in which the gradient
   * vectors are arranged corresponds to the ordering of the voxel points.
   * Gradient vector is computed by central differences (except on edges of
   * volume where forward difference is used). The scalars s are the scalars
   * from which the gradient is to be computed. This method will treat
   * only 3D structured point datasets (i.e., volumes).
   */
  virtual void GetVoxelGradient(int i, int j, int k, svtkDataArray* s, svtkDataArray* g);

  /**
   * Given structured coordinates (i,j,k) for a point in a structured point
   * dataset, compute the gradient vector from the scalar data at that point.
   * The scalars s are the scalars from which the gradient is to be computed.
   * This method will treat structured point datasets of any dimension.
   */
  virtual void GetPointGradient(int i, int j, int k, svtkDataArray* s, double g[3]);

  /**
   * Return the dimensionality of the data.
   */
  virtual int GetDataDimension();

  /**
   * Given a location in structured coordinates (i-j-k), return the point id.
   */
  virtual svtkIdType ComputePointId(int ijk[3])
  {
    return svtkStructuredData::ComputePointIdForExtent(this->Extent, ijk);
  }

  /**
   * Given a location in structured coordinates (i-j-k), return the cell id.
   */
  virtual svtkIdType ComputeCellId(int ijk[3])
  {
    return svtkStructuredData::ComputeCellIdForExtent(this->Extent, ijk);
  }

  //@{
  /**
   * Set / Get the extent on just one axis
   */
  virtual void SetAxisUpdateExtent(
    int axis, int min, int max, const int* updateExtent, int* axisUpdateExtent);
  virtual void GetAxisUpdateExtent(int axis, int& min, int& max, const int* updateExtent);
  //@}

  //@{
  /**
   * Set/Get the extent. On each axis, the extent is defined by the index
   * of the first point and the index of the last point.  The extent should
   * be set before the "Scalars" are set or allocated.  The Extent is
   * stored in the order (X, Y, Z).
   * The dataset extent does not have to start at (0,0,0). (0,0,0) is just the
   * extent of the origin.
   * The first point (the one with Id=0) is at extent
   * (Extent[0],Extent[2],Extent[4]). As for any dataset, a data array on point
   * data starts at Id=0.
   */
  virtual void SetExtent(int extent[6]);
  virtual void SetExtent(int x1, int x2, int y1, int y2, int z1, int z2);
  svtkGetVector6Macro(Extent, int);
  //@}

  //@{
  /**
   * These returns the minimum and maximum values the ScalarType can hold
   * without overflowing.
   */
  virtual double GetScalarTypeMin(svtkInformation* meta_data);
  virtual double GetScalarTypeMin();
  virtual double GetScalarTypeMax(svtkInformation* meta_data);
  virtual double GetScalarTypeMax();
  //@}

  //@{
  /**
   * Get the size of the scalar type in bytes.
   */
  virtual int GetScalarSize(svtkInformation* meta_data);
  virtual int GetScalarSize();
  //@}

  //@{
  /**
   * Different ways to get the increments for moving around the data.
   * GetIncrements() calls ComputeIncrements() to ensure the increments are
   * up to date.  The first three methods compute the increments based on the
   * active scalar field while the next three, the scalar field is passed in.
   */
  virtual svtkIdType* GetIncrements() SVTK_SIZEHINT(3);
  virtual void GetIncrements(svtkIdType& incX, svtkIdType& incY, svtkIdType& incZ);
  virtual void GetIncrements(svtkIdType inc[3]);
  virtual svtkIdType* GetIncrements(svtkDataArray* scalars) SVTK_SIZEHINT(3);
  virtual void GetIncrements(
    svtkDataArray* scalars, svtkIdType& incX, svtkIdType& incY, svtkIdType& incZ);
  virtual void GetIncrements(svtkDataArray* scalars, svtkIdType inc[3]);
  //@}

  //@{
  /**
   * Different ways to get the increments for moving around the data.
   * incX is always returned with 0.  incY is returned with the
   * increment needed to move from the end of one X scanline of data
   * to the start of the next line.  incZ is filled in with the
   * increment needed to move from the end of one image to the start
   * of the next.  The proper way to use these values is to for a loop
   * over Z, Y, X, C, incrementing the pointer by 1 after each
   * component.  When the end of the component is reached, the pointer
   * is set to the beginning of the next pixel, thus incX is properly set to 0.
   * The first form of GetContinuousIncrements uses the active scalar field
   * while the second form allows the scalar array to be passed in.
   */
  virtual void GetContinuousIncrements(
    int extent[6], svtkIdType& incX, svtkIdType& incY, svtkIdType& incZ);
  virtual void GetContinuousIncrements(
    svtkDataArray* scalars, int extent[6], svtkIdType& incX, svtkIdType& incY, svtkIdType& incZ);
  //@}

  //@{
  /**
   * Access the native pointer for the scalar data
   */
  virtual void* GetScalarPointerForExtent(int extent[6]);
  virtual void* GetScalarPointer(int coordinates[3]);
  virtual void* GetScalarPointer(int x, int y, int z);
  virtual void* GetScalarPointer();
  //@}

  //@{
  /**
   * For access to data from wrappers
   */
  virtual float GetScalarComponentAsFloat(int x, int y, int z, int component);
  virtual void SetScalarComponentFromFloat(int x, int y, int z, int component, float v);
  virtual double GetScalarComponentAsDouble(int x, int y, int z, int component);
  virtual void SetScalarComponentFromDouble(int x, int y, int z, int component, double v);
  //@}

  /**
   * Allocate the point scalars for this dataset. The data type determines
   * the type of the array (SVTK_FLOAT, SVTK_INT etc.) where as numComponents
   * determines its number of components.
   */
  virtual void AllocateScalars(int dataType, int numComponents);

  /**
   * Allocate the point scalars for this dataset. The data type and the
   * number of components of the array is determined by the meta-data in
   * the pipeline information. This is usually produced by a reader/filter
   * upstream in the pipeline.
   */
  virtual void AllocateScalars(svtkInformation* pipeline_info);

  //@{
  /**
   * This method is passed a input and output region, and executes the filter
   * algorithm to fill the output from the input.
   * It just executes a switch statement to call the correct function for
   * the regions data types.
   */
  virtual void CopyAndCastFrom(svtkImageData* inData, int extent[6]);
  virtual void CopyAndCastFrom(svtkImageData* inData, int x0, int x1, int y0, int y1, int z0, int z1)
  {
    int e[6];
    e[0] = x0;
    e[1] = x1;
    e[2] = y0;
    e[3] = y1;
    e[4] = z0;
    e[5] = z1;
    this->CopyAndCastFrom(inData, e);
  }
  //@}

  /**
   * Reallocates and copies to set the Extent to updateExtent.
   * This is used internally when the exact extent is requested,
   * and the source generated more than the update extent.
   */
  void Crop(const int* updateExtent) override;

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
   * Set the spacing (width,height,length) of the cubical cells that
   * compose the data set.
   */
  svtkGetVector3Macro(Spacing, double);
  virtual void SetSpacing(double i, double j, double k);
  virtual void SetSpacing(const double ijk[3]);
  //@}

  //@{
  /**
   * Set/Get the origin of the dataset. The origin is the position in world
   * coordinates of the point of extent (0,0,0). This point does not have to be
   * part of the dataset, in other words, the dataset extent does not have to
   * start at (0,0,0) and the origin can be outside of the dataset bounding
   * box.
   * The origin plus spacing determine the position in space of the points.
   */
  svtkGetVector3Macro(Origin, double);
  virtual void SetOrigin(double i, double j, double k);
  virtual void SetOrigin(const double ijk[3]);
  //@}

  //@{
  /**
   * Set/Get the direction transform of the dataset. The direction is a 3 by 3
   * matrix.
   */
  svtkGetObjectMacro(DirectionMatrix, svtkMatrix3x3);
  virtual void SetDirectionMatrix(svtkMatrix3x3* m);
  virtual void SetDirectionMatrix(const double elements[9]);
  virtual void SetDirectionMatrix(double e00, double e01, double e02, double e10, double e11,
    double e12, double e20, double e21, double e22);
  //@}

  //@{
  /**
   * Get the transformation matrix from the index space to the physical space
   * coordinate system of the dataset. The transform is a 4 by 4 matrix.
   */
  svtkGetObjectMacro(IndexToPhysicalMatrix, svtkMatrix4x4);
  //@}

  //@{
  /**
   * Convert coordinates from index space (ijk) to physical space (xyz)
   */
  virtual void TransformContinuousIndexToPhysicalPoint(double i, double j, double k, double xyz[3]);
  virtual void TransformContinuousIndexToPhysicalPoint(const double ijk[3], double xyz[3]);
  virtual void TransformIndexToPhysicalPoint(int i, int j, int k, double xyz[3]);
  virtual void TransformIndexToPhysicalPoint(const int ijk[3], double xyz[3]);
  static void TransformContinuousIndexToPhysicalPoint(double i, double j, double k,
    double const origin[3], double const spacing[3], double const direction[9], double xyz[3]);
  //@}

  //@{
  /**
   * Get the transformation matrix from the physical space to the index space
   * coordinate system of the dataset. The transform is a 4 by 4 matrix.
   */
  svtkGetObjectMacro(PhysicalToIndexMatrix, svtkMatrix4x4);
  //@}

  //@{
  /**
   * Convert coordinates from physical space (xyz) to index space (ijk)
   */
  virtual void TransformPhysicalPointToContinuousIndex(double x, double y, double z, double ijk[3]);
  virtual void TransformPhysicalPointToContinuousIndex(const double xyz[3], double ijk[3]);
  //@}

  static void ComputeIndexToPhysicalMatrix(
    double const origin[3], double const spacing[3], double const direction[9], double result[16]);

  //@{
  /**
   * Convert normal from physical space (xyz) to index space (ijk)
   */
  virtual void TransformPhysicalNormalToContinuousIndex(const double xyz[3], double ijk[3]);
  //@}

  /**
   * Convert a plane form physical to continuous index
   */
  virtual void TransformPhysicalPlaneToContinuousIndex(double const pplane[4], double iplane[4]);

  static void SetScalarType(int, svtkInformation* meta_data);
  static int GetScalarType(svtkInformation* meta_data);
  static bool HasScalarType(svtkInformation* meta_data);
  int GetScalarType();
  const char* GetScalarTypeAsString() { return svtkImageScalarTypeNameMacro(this->GetScalarType()); }

  //@{
  /**
   * Set/Get the number of scalar components for points. As with the
   * SetScalarType method this is setting pipeline info.
   */
  static void SetNumberOfScalarComponents(int n, svtkInformation* meta_data);
  static int GetNumberOfScalarComponents(svtkInformation* meta_data);
  static bool HasNumberOfScalarComponents(svtkInformation* meta_data);
  int GetNumberOfScalarComponents();
  //@}

  /**
   * Override these to handle origin, spacing, scalar type, and scalar
   * number of components.  See svtkDataObject for details.
   */
  void CopyInformationFromPipeline(svtkInformation* information) override;

  /**
   * Copy information from this data object to the pipeline information.
   * This is used by the svtkTrivialProducer that is created when someone
   * calls SetInputData() to connect the image to a pipeline.
   */
  void CopyInformationToPipeline(svtkInformation* information) override;

  /**
   * make the output data ready for new data to be inserted. For most
   * objects we just call Initialize. But for image data we leave the old
   * data in case the memory can be reused.
   */
  void PrepareForNewData() override;

  //@{
  /**
   * Shallow and Deep copy.
   */
  void ShallowCopy(svtkDataObject* src) override;
  void DeepCopy(svtkDataObject* src) override;
  //@}

  //--------------------------------------------------------------------------
  // Methods that apply to any array (not just scalars).
  // I am starting to experiment with generalizing imaging filters
  // to operate on more than just scalars.

  //@{
  /**
   * These are convenience methods for getting a pointer
   * from any filed array.  It is a start at expanding image filters
   * to process any array (not just scalars).
   */
  void* GetArrayPointerForExtent(svtkDataArray* array, int extent[6]);
  void* GetArrayPointer(svtkDataArray* array, int coordinates[3]);
  //@}

  /**
   * Since various arrays have different number of components,
   * the will have different increments.
   */
  void GetArrayIncrements(svtkDataArray* array, svtkIdType increments[3]);

  /**
   * Given how many pixel are required on a side for bounrary conditions (in
   * bnds), the target extent to traverse, compute the internal extent (the
   * extent for this ImageData that does not suffer from any boundary
   * conditions) and place it in intExt
   */
  void ComputeInternalExtent(int* intExt, int* tgtExt, int* bnds);

  /**
   * The extent type is a 3D extent
   */
  int GetExtentType() override { return SVTK_3D_EXTENT; }

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkImageData* GetData(svtkInformation* info);
  static svtkImageData* GetData(svtkInformationVector* v, int i = 0);
  //@}

protected:
  svtkImageData();
  ~svtkImageData() override;

  // The extent of what is currently in the structured grid.
  // Dimensions is just an array to return a value.
  // Its contents are out of data until GetDimensions is called.
  int Dimensions[3];
  svtkIdType Increments[3];

  // Variables used to define dataset physical orientation
  double Origin[3];
  double Spacing[3];
  svtkMatrix3x3* DirectionMatrix;
  svtkMatrix4x4* IndexToPhysicalMatrix;
  svtkMatrix4x4* PhysicalToIndexMatrix;

  int Extent[6];

  // The first method assumes Active Scalars
  void ComputeIncrements();
  // This one is given the number of components of the
  // scalar field explicitly
  void ComputeIncrements(int numberOfComponents);
  void ComputeIncrements(svtkDataArray* scalars);

  // The first method assumes Acitive Scalars
  void ComputeIncrements(svtkIdType inc[3]);
  // This one is given the number of components of the
  // scalar field explicitly
  void ComputeIncrements(int numberOfComponents, svtkIdType inc[3]);
  void ComputeIncrements(svtkDataArray* scalars, svtkIdType inc[3]);

  // for the index to physical methods
  void ComputeTransforms();

  // Cell utilities
  svtkCell* GetCellTemplateForDataDescription();
  bool GetCellTemplateForDataDescription(svtkGenericCell* cell);
  bool GetIJKMinForCellId(svtkIdType cellId, int ijkMin[3]);
  bool GetIJKMaxForIJKMin(int ijkMin[3], int ijkMax[3]);
  void AddPointsToCellTemplate(svtkCell* cell, int ijkMin[3], int ijkMax[3]);

  svtkTimeStamp ExtentComputeTime;

  void SetDataDescription(int desc);
  int GetDataDescription() { return this->DataDescription; }

private:
  void InternalImageDataCopy(svtkImageData* src);

private:
  friend class svtkUniformGrid;

  // for the GetCell method
  svtkVertex* Vertex;
  svtkLine* Line;
  svtkPixel* Pixel;
  svtkVoxel* Voxel;

  // for the GetPoint method
  double Point[3];

  int DataDescription;

  svtkImageData(const svtkImageData&) = delete;
  void operator=(const svtkImageData&) = delete;
};

//----------------------------------------------------------------------------
inline void svtkImageData::ComputeIncrements()
{
  this->ComputeIncrements(this->Increments);
}

//----------------------------------------------------------------------------
inline void svtkImageData::ComputeIncrements(int numberOfComponents)
{
  this->ComputeIncrements(numberOfComponents, this->Increments);
}

//----------------------------------------------------------------------------
inline void svtkImageData::ComputeIncrements(svtkDataArray* scalars)
{
  this->ComputeIncrements(scalars, this->Increments);
}

//----------------------------------------------------------------------------
inline double* svtkImageData::GetPoint(svtkIdType id)
{
  this->GetPoint(id, this->Point);
  return this->Point;
}

//----------------------------------------------------------------------------
inline svtkIdType svtkImageData::GetNumberOfPoints()
{
  const int* extent = this->Extent;
  svtkIdType dims[3];
  dims[0] = extent[1] - extent[0] + 1;
  dims[1] = extent[3] - extent[2] + 1;
  dims[2] = extent[5] - extent[4] + 1;

  return dims[0] * dims[1] * dims[2];
}

//----------------------------------------------------------------------------
inline int svtkImageData::GetDataDimension()
{
  return svtkStructuredData::GetDataDimension(this->DataDescription);
}

#endif
