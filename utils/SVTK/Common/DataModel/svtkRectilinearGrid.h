/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkRectilinearGrid.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkRectilinearGrid
 * @brief   a dataset that is topologically regular with variable spacing in the three coordinate
 * directions
 *
 * svtkRectilinearGrid is a data object that is a concrete implementation of
 * svtkDataSet. svtkRectilinearGrid represents a geometric structure that is
 * topologically regular with variable spacing in the three coordinate
 * directions x-y-z.
 *
 * To define a svtkRectilinearGrid, you must specify the dimensions of the
 * data and provide three arrays of values specifying the coordinates
 * along the x-y-z axes. The coordinate arrays are specified using three
 * svtkDataArray objects (one for x, one for y, one for z).
 *
 * @warning
 * Make sure that the dimensions of the grid match the number of coordinates
 * in the x-y-z directions. If not, unpredictable results (including
 * program failure) may result. Also, you must supply coordinates in all
 * three directions, even if the dataset topology is 2D, 1D, or 0D.
 */

#ifndef svtkRectilinearGrid_h
#define svtkRectilinearGrid_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataSet.h"
#include "svtkStructuredData.h" // For inline methods

class svtkVertex;
class svtkLine;
class svtkPixel;
class svtkVoxel;
class svtkDataArray;
class svtkPoints;

class SVTKCOMMONDATAMODEL_EXPORT svtkRectilinearGrid : public svtkDataSet
{
public:
  static svtkRectilinearGrid* New();

  svtkTypeMacro(svtkRectilinearGrid, svtkDataSet);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return what type of dataset this is.
   */
  int GetDataObjectType() override { return SVTK_RECTILINEAR_GRID; }

  /**
   * Copy the geometric and topological structure of an input rectilinear grid
   * object.
   */
  void CopyStructure(svtkDataSet* ds) override;

  /**
   * Restore object to initial state. Release memory back to system.
   */
  void Initialize() override;

  //@{
  /**
   * Standard svtkDataSet API methods. See svtkDataSet for more information.
   */
  svtkIdType GetNumberOfCells() override;
  svtkIdType GetNumberOfPoints() override;
  double* GetPoint(svtkIdType ptId) SVTK_SIZEHINT(3) override;
  void GetPoint(svtkIdType id, double x[3]) override;
  svtkCell* GetCell(svtkIdType cellId) override;
  svtkCell* GetCell(int i, int j, int k) override;
  void GetCell(svtkIdType cellId, svtkGenericCell* cell) override;
  void GetCellBounds(svtkIdType cellId, double bounds[6]) override;
  svtkIdType FindPoint(double x, double y, double z) { return this->svtkDataSet::FindPoint(x, y, z); }
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
    svtkStructuredData::GetCellPoints(cellId, ptIds, this->DataDescription, this->Dimensions);
  }
  void GetPointCells(svtkIdType ptId, svtkIdList* cellIds) override
  {
    svtkStructuredData::GetPointCells(ptId, cellIds, this->Dimensions);
  }
  void ComputeBounds() override;
  int GetMaxCellSize() override { return 8; } // voxel is the largest
  void GetCellNeighbors(svtkIdType cellId, svtkIdList* ptIds, svtkIdList* cellIds) override;
  //@}

  /**
   * Given a user-supplied svtkPoints container object, this method fills in all
   * the points of the RectilinearGrid.
   */
  void GetPoints(svtkPoints* pnts);

  //@{
  /**
   * Set dimensions of rectilinear grid dataset.
   * This also sets the extent.
   */
  void SetDimensions(int i, int j, int k);
  void SetDimensions(const int dim[3]);
  //@}

  //@{
  /**
   * Get dimensions of this rectilinear grid dataset.
   */
  svtkGetVectorMacro(Dimensions, int, 3);
  //@}

  /**
   * Return the dimensionality of the data.
   */
  int GetDataDimension();

  /**
   * Convenience function computes the structured coordinates for a point x[3].
   * The cell is specified by the array ijk[3], and the parametric coordinates
   * in the cell are specified with pcoords[3]. The function returns a 0 if the
   * point x is outside of the grid, and a 1 if inside the grid.
   */
  int ComputeStructuredCoordinates(double x[3], int ijk[3], double pcoords[3]);

  /**
   * Given a location in structured coordinates (i-j-k), return the point id.
   */
  svtkIdType ComputePointId(int ijk[3]);

  /**
   * Given a location in structured coordinates (i-j-k), return the cell id.
   */
  svtkIdType ComputeCellId(int ijk[3]);

  /**
   * Given the IJK-coordinates of the point, it returns the corresponding
   * xyz-coordinates. The xyz coordinates are stored in the user-supplied
   * array p.
   */
  void GetPoint(const int i, const int j, const int k, double p[3]);

  //@{
  /**
   * Specify the grid coordinates in the x-direction.
   */
  virtual void SetXCoordinates(svtkDataArray*);
  svtkGetObjectMacro(XCoordinates, svtkDataArray);
  //@}

  //@{
  /**
   * Specify the grid coordinates in the y-direction.
   */
  virtual void SetYCoordinates(svtkDataArray*);
  svtkGetObjectMacro(YCoordinates, svtkDataArray);
  //@}

  //@{
  /**
   * Specify the grid coordinates in the z-direction.
   */
  virtual void SetZCoordinates(svtkDataArray*);
  svtkGetObjectMacro(ZCoordinates, svtkDataArray);
  //@}

  //@{
  /**
   * Different ways to set the extent of the data array.  The extent
   * should be set before the "Scalars" are set or allocated.
   * The Extent is stored in the order (X, Y, Z).
   */
  void SetExtent(int extent[6]);
  void SetExtent(int x1, int x2, int y1, int y2, int z1, int z2);
  svtkGetVector6Macro(Extent, int);
  //@}

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
   * Structured extent. The extent type is a 3D extent
   */
  int GetExtentType() override { return SVTK_3D_EXTENT; }

  /**
   * Reallocates and copies to set the Extent to the UpdateExtent.
   * This is used internally when the exact extent is requested,
   * and the source generated more than the update extent.
   */
  void Crop(const int* updateExtent) override;

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkRectilinearGrid* GetData(svtkInformation* info);
  static svtkRectilinearGrid* GetData(svtkInformationVector* v, int i = 0);
  //@}

protected:
  svtkRectilinearGrid();
  ~svtkRectilinearGrid() override;

  // for the GetCell method
  svtkVertex* Vertex;
  svtkLine* Line;
  svtkPixel* Pixel;
  svtkVoxel* Voxel;

  int Dimensions[3];
  int DataDescription;

  int Extent[6];

  svtkDataArray* XCoordinates;
  svtkDataArray* YCoordinates;
  svtkDataArray* ZCoordinates;

  // Hang on to some space for returning points when GetPoint(id) is called.
  double PointReturn[3];

private:
  void Cleanup();

private:
  svtkRectilinearGrid(const svtkRectilinearGrid&) = delete;
  void operator=(const svtkRectilinearGrid&) = delete;
};

//----------------------------------------------------------------------------
inline svtkIdType svtkRectilinearGrid::GetNumberOfCells()
{
  svtkIdType nCells = 1;
  int i;

  for (i = 0; i < 3; i++)
  {
    if (this->Dimensions[i] <= 0)
    {
      return 0;
    }
    if (this->Dimensions[i] > 1)
    {
      nCells *= (this->Dimensions[i] - 1);
    }
  }

  return nCells;
}

//----------------------------------------------------------------------------
inline svtkIdType svtkRectilinearGrid::GetNumberOfPoints()
{
  return static_cast<svtkIdType>(this->Dimensions[0]) * this->Dimensions[1] * this->Dimensions[2];
}

//----------------------------------------------------------------------------
inline int svtkRectilinearGrid::GetDataDimension()
{
  return svtkStructuredData::GetDataDimension(this->DataDescription);
}

//----------------------------------------------------------------------------
inline svtkIdType svtkRectilinearGrid::ComputePointId(int ijk[3])
{
  return svtkStructuredData::ComputePointId(this->Dimensions, ijk);
}

//----------------------------------------------------------------------------
inline svtkIdType svtkRectilinearGrid::ComputeCellId(int ijk[3])
{
  return svtkStructuredData::ComputeCellId(this->Dimensions, ijk);
}

#endif
