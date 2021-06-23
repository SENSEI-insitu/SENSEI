/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPoints.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPoints
 * @brief   represent and manipulate 3D points
 *
 * svtkPoints represents 3D points. The data model for svtkPoints is an
 * array of vx-vy-vz triplets accessible by (point or cell) id.
 */

#ifndef svtkPoints_h
#define svtkPoints_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

#include "svtkDataArray.h" // Needed for inline methods

class svtkIdList;

class SVTKCOMMONCORE_EXPORT svtkPoints : public svtkObject
{
public:
  static svtkPoints* New(int dataType);

  static svtkPoints* New();

  svtkTypeMacro(svtkPoints, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Allocate initial memory size. ext is no longer used.
   */
  virtual svtkTypeBool Allocate(svtkIdType sz, svtkIdType ext = 1000);

  /**
   * Return object to instantiated state.
   */
  virtual void Initialize();

  /**
   * Set/Get the underlying data array. This function must be implemented
   * in a concrete subclass to check for consistency. (The tuple size must
   * match the type of data. For example, 3-tuple data array can be assigned to
   * a vector, normal, or points object, but not a tensor object, which has a
   * tuple dimension of 9. Scalars, on the other hand, can have tuple dimension
   * from 1-4, depending on the type of scalar.)
   */
  virtual void SetData(svtkDataArray*);
  svtkDataArray* GetData() { return this->Data; }

  /**
   * Return the underlying data type. An integer indicating data type is
   * returned as specified in svtkSetGet.h.
   */
  virtual int GetDataType() const;

  /**
   * Specify the underlying data type of the object.
   */
  virtual void SetDataType(int dataType);
  void SetDataTypeToBit() { this->SetDataType(SVTK_BIT); }
  void SetDataTypeToChar() { this->SetDataType(SVTK_CHAR); }
  void SetDataTypeToUnsignedChar() { this->SetDataType(SVTK_UNSIGNED_CHAR); }
  void SetDataTypeToShort() { this->SetDataType(SVTK_SHORT); }
  void SetDataTypeToUnsignedShort() { this->SetDataType(SVTK_UNSIGNED_SHORT); }
  void SetDataTypeToInt() { this->SetDataType(SVTK_INT); }
  void SetDataTypeToUnsignedInt() { this->SetDataType(SVTK_UNSIGNED_INT); }
  void SetDataTypeToLong() { this->SetDataType(SVTK_LONG); }
  void SetDataTypeToUnsignedLong() { this->SetDataType(SVTK_UNSIGNED_LONG); }
  void SetDataTypeToFloat() { this->SetDataType(SVTK_FLOAT); }
  void SetDataTypeToDouble() { this->SetDataType(SVTK_DOUBLE); }

  /**
   * Return a void pointer. For image pipeline interface and other
   * special pointer manipulation.
   */
  void* GetVoidPointer(const int id) { return this->Data->GetVoidPointer(id); }

  /**
   * Reclaim any extra memory.
   */
  virtual void Squeeze() { this->Data->Squeeze(); }

  /**
   * Make object look empty but do not delete memory.
   */
  virtual void Reset();

  //@{
  /**
   * Different ways to copy data. Shallow copy does reference count (i.e.,
   * assigns pointers and updates reference count); deep copy runs through
   * entire data array assigning values.
   */
  virtual void DeepCopy(svtkPoints* ad);
  virtual void ShallowCopy(svtkPoints* ad);
  //@}

  /**
   * Return the memory in kibibytes (1024 bytes) consumed by this attribute data.
   * Used to support streaming and reading/writing data. The value
   * returned is guaranteed to be greater than or equal to the
   * memory required to actually represent the data represented
   * by this object. The information returned is valid only after
   * the pipeline has been updated.
   */
  unsigned long GetActualMemorySize();

  /**
   * Return number of points in array.
   */
  svtkIdType GetNumberOfPoints() { return this->Data->GetNumberOfTuples(); }

  /**
   * Return a pointer to a double point x[3] for a specific id.
   * WARNING: Just don't use this error-prone method, the returned pointer
   * and its values are only valid as long as another method invocation is not
   * performed. Prefer GetPoint() with the return value in argument.
   */
  double* GetPoint(svtkIdType id) SVTK_EXPECTS(0 <= id && id < GetNumberOfPoints()) SVTK_SIZEHINT(3)
  {
    return this->Data->GetTuple(id);
  }

  /**
   * Copy point components into user provided array v[3] for specified
   * id.
   */
  void GetPoint(svtkIdType id, double x[3]) SVTK_EXPECTS(0 <= id && id < GetNumberOfPoints())
    SVTK_SIZEHINT(3)
  {
    this->Data->GetTuple(id, x);
  }

  /**
   * Insert point into object. No range checking performed (fast!).
   * Make sure you use SetNumberOfPoints() to allocate memory prior
   * to using SetPoint(). You should call Modified() finally after
   * changing points using this method as it will not do it itself.
   */
  void SetPoint(svtkIdType id, const float x[3]) SVTK_EXPECTS(0 <= id && id < GetNumberOfPoints())
  {
    this->Data->SetTuple(id, x);
  }
  void SetPoint(svtkIdType id, const double x[3]) SVTK_EXPECTS(0 <= id && id < GetNumberOfPoints())
  {
    this->Data->SetTuple(id, x);
  }
  void SetPoint(svtkIdType id, double x, double y, double z)
    SVTK_EXPECTS(0 <= id && id < GetNumberOfPoints());

  //@{
  /**
   * Insert point into object. Range checking performed and memory
   * allocated as necessary.
   */
  void InsertPoint(svtkIdType id, const float x[3]) SVTK_EXPECTS(0 <= id)
  {
    this->Data->InsertTuple(id, x);
  }
  void InsertPoint(svtkIdType id, const double x[3]) SVTK_EXPECTS(0 <= id)
  {
    this->Data->InsertTuple(id, x);
  }
  void InsertPoint(svtkIdType id, double x, double y, double z) SVTK_EXPECTS(0 <= id);
  //@}

  /**
   * Copy the points indexed in srcIds from the source array to the tuple
   * locations indexed by dstIds in this array.
   * Note that memory allocation is performed as necessary to hold the data.
   */
  void InsertPoints(svtkIdList* dstIds, svtkIdList* srcIds, svtkPoints* source)
  {
    this->Data->InsertTuples(dstIds, srcIds, source->Data);
  }

  /**
   * Copy n consecutive points starting at srcStart from the source array to
   * this array, starting at the dstStart location.
   * Note that memory allocation is performed as necessary to hold the data.
   */
  void InsertPoints(svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkPoints* source)
  {
    this->Data->InsertTuples(dstStart, n, srcStart, source->Data);
  }

  /**
   * Insert point into next available slot. Returns id of slot.
   */
  svtkIdType InsertNextPoint(const float x[3]) { return this->Data->InsertNextTuple(x); }
  svtkIdType InsertNextPoint(const double x[3]) { return this->Data->InsertNextTuple(x); }
  svtkIdType InsertNextPoint(double x, double y, double z);

  /**
   * Specify the number of points for this object to hold. Does an
   * allocation as well as setting the MaxId ivar. Used in conjunction with
   * SetPoint() method for fast insertion.
   */
  void SetNumberOfPoints(svtkIdType numPoints);

  /**
   * Resize the internal array while conserving the data.  Returns 1 if
   * resizing succeeded and 0 otherwise.
   */
  svtkTypeBool Resize(svtkIdType numPoints);

  /**
   * Given a list of pt ids, return an array of points.
   */
  void GetPoints(svtkIdList* ptId, svtkPoints* fp);

  /**
   * Determine (xmin,xmax, ymin,ymax, zmin,zmax) bounds of points.
   */
  virtual void ComputeBounds();

  /**
   * Return the bounds of the points.
   */
  double* GetBounds() SVTK_SIZEHINT(6);

  /**
   * Return the bounds of the points.
   */
  void GetBounds(double bounds[6]);

  /**
   * The modified time of the points.
   */
  svtkMTimeType GetMTime() override;

  /**
   * Update the modification time for this object and its Data.
   * As this object acts as a shell around a DataArray and
   * forwards Set methods it needs to forward Modified as well.
   */
  void Modified() override;

protected:
  svtkPoints(int dataType = SVTK_FLOAT);
  ~svtkPoints() override;

  double Bounds[6];
  svtkTimeStamp ComputeTime; // Time at which bounds computed
  svtkDataArray* Data;       // Array which represents data

private:
  svtkPoints(const svtkPoints&) = delete;
  void operator=(const svtkPoints&) = delete;
};

inline void svtkPoints::Reset()
{
  this->Data->Reset();
  this->Modified();
}

inline void svtkPoints::SetNumberOfPoints(svtkIdType numPoints)
{
  this->Data->SetNumberOfComponents(3);
  this->Data->SetNumberOfTuples(numPoints);
  this->Modified();
}

inline svtkTypeBool svtkPoints::Resize(svtkIdType numPoints)
{
  this->Data->SetNumberOfComponents(3);
  this->Modified();
  return this->Data->Resize(numPoints);
}

inline void svtkPoints::SetPoint(svtkIdType id, double x, double y, double z)
{
  double p[3] = { x, y, z };
  this->Data->SetTuple(id, p);
}

inline void svtkPoints::InsertPoint(svtkIdType id, double x, double y, double z)
{
  double p[3] = { x, y, z };
  this->Data->InsertTuple(id, p);
}

inline svtkIdType svtkPoints::InsertNextPoint(double x, double y, double z)
{
  double p[3] = { x, y, z };
  return this->Data->InsertNextTuple(p);
}

#endif
