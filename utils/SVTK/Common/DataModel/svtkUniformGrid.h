/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUniformGrid.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkUniformGrid
 * @brief   image data with blanking
 *
 * svtkUniformGrid is a subclass of svtkImageData. In addition to all
 * the image data functionality, it supports blanking.
 */

#ifndef svtkUniformGrid_h
#define svtkUniformGrid_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImageData.h"

class svtkEmptyCell;
class svtkStructuredVisibilityConstraint;
class svtkUnsignedCharArray;
class svtkAMRBox;

class SVTKCOMMONDATAMODEL_EXPORT svtkUniformGrid : public svtkImageData
{
public:
  //@{
  /**
   * Construct an empty uniform grid.
   */
  static svtkUniformGrid* New();
  svtkTypeMacro(svtkUniformGrid, svtkImageData);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Copy the geometric and topological structure of an input image data
   * object.
   */
  void CopyStructure(svtkDataSet* ds) override;

  /**
   * Return what type of dataset this is.
   */
  int GetDataObjectType() override { return SVTK_UNIFORM_GRID; }

  //@{
  /**
   * Standard svtkDataSet API methods. See svtkDataSet for more information.
   */
  svtkCell* GetCell(int i, int j, int k) override;
  svtkCell* GetCell(svtkIdType cellId) override;
  void GetCell(svtkIdType cellId, svtkGenericCell* cell) override;
  svtkIdType FindCell(double x[3], svtkCell* cell, svtkIdType cellId, double tol2, int& subId,
    double pcoords[3], double* weights) override;
  svtkIdType FindCell(double x[3], svtkCell* cell, svtkGenericCell* gencell, svtkIdType cellId,
    double tol2, int& subId, double pcoords[3], double* weights) override;
  svtkCell* FindAndGetCell(double x[3], svtkCell* cell, svtkIdType cellId, double tol2, int& subId,
    double pcoords[3], double* weights) override;
  int GetCellType(svtkIdType cellId) override;
  void GetCellPoints(svtkIdType cellId, svtkIdList* ptIds) override
  {
    svtkStructuredData::GetCellPoints(
      cellId, ptIds, this->GetDataDescription(), this->GetDimensions());
  }
  void GetPointCells(svtkIdType ptId, svtkIdList* cellIds) override
  {
    svtkStructuredData::GetPointCells(ptId, cellIds, this->GetDimensions());
  }
  void Initialize() override;
  int GetMaxCellSize() override { return 8; } // voxel is the largest
  //@}

  /**
   * Returns the data description of this uniform grid instance.
   */
  int GetGridDescription();

  /**
   * Initialize with no ghost cell arrays, from the definition in
   * the given box. The box is expetced to be 3D, if you have 2D
   * data the set the third dimensions 0. eg. (X,X,0)(X,X,0)
   * Returns 0 if the initialization failed.
   */
  int Initialize(const svtkAMRBox* def, double* origin, double* spacing);
  /**
   * Initialize from the definition in the given box, with ghost cell
   * arrays nGhosts cells thick in all directions. The box is expetced
   * to be 3D, if you have 2D data the set the third dimensions 0.
   * eg. (X,X,0)(X,X,0)
   * Returns 0 if the initialization failed.
   */
  int Initialize(const svtkAMRBox* def, double* origin, double* spacing, int nGhosts);

  /**
   * Initialize from the definition in the given box, with ghost cell
   * arrays of the thickness given in each direction by "nGhosts" array.
   * The box and ghost array are expected to be 3D, if you have 2D data
   * the set the third dimensions 0. eg. (X,X,0)(X,X,0)
   * Returns 0 if the initialization failed.
   */
  int Initialize(const svtkAMRBox* def, double* origin, double* spacing, const int nGhosts[3]);
  /**
   * Construct a uniform grid, from the definition in the given box
   * "def", with ghost cell arrays of the thickness given in each
   * direction by "nGhosts*". The box and ghost array are expected
   * to be 3D, if you have 2D data the set the third dimensions 0. eg.
   * (X,X,0)(X,X,0)
   * Returns 0 if the initialization failed.
   */
  int Initialize(const svtkAMRBox* def, double* origin, double* spacing, int nGhostsI, int nGhostsJ,
    int nGhostsK);

  //@{
  /**
   * Methods for supporting blanking of cells. Blanking turns on or off
   * points in the structured grid, and hence the cells connected to them.
   * These methods should be called only after the dimensions of the
   * grid are set.
   */
  virtual void BlankPoint(svtkIdType ptId);
  virtual void UnBlankPoint(svtkIdType ptId);
  virtual void BlankPoint(const int i, const int j, const int k);
  virtual void UnBlankPoint(const int i, const int j, const int k);
  //@}

  //@{
  /**
   * Methods for supporting blanking of cells. Blanking turns on or off
   * cells in the structured grid.
   * These methods should be called only after the dimensions of the
   * grid are set.
   */
  virtual void BlankCell(svtkIdType ptId);
  virtual void UnBlankCell(svtkIdType ptId);
  virtual void BlankCell(const int i, const int j, const int k);
  virtual void UnBlankCell(const int i, const int j, const int k);
  //@}

  /**
   * Returns 1 if there is any visibility constraint on the cells,
   * 0 otherwise.
   */
  bool HasAnyBlankCells() override;
  /**
   * Returns 1 if there is any visibility constraint on the points,
   * 0 otherwise.
   */
  bool HasAnyBlankPoints() override;

  /**
   * Return non-zero value if specified point is visible.
   * These methods should be called only after the dimensions of the
   * grid are set.
   */
  virtual unsigned char IsPointVisible(svtkIdType ptId);

  /**
   * Return non-zero value if specified cell is visible.
   * These methods should be called only after the dimensions of the
   * grid are set.
   */
  virtual unsigned char IsCellVisible(svtkIdType cellId);

  virtual svtkImageData* NewImageDataCopy();

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkUniformGrid* GetData(svtkInformation* info);
  static svtkUniformGrid* GetData(svtkInformationVector* v, int i = 0);
  //@}

protected:
  svtkUniformGrid();
  ~svtkUniformGrid() override;

  /**
   * Returns the cell dimensions for this svtkUniformGrid instance.
   */
  void GetCellDims(int cellDims[3]);

  /**
   * Override this method because of blanking.
   */
  void ComputeScalarRange() override;

  svtkEmptyCell* GetEmptyCell();

private:
  svtkUniformGrid(const svtkUniformGrid&) = delete;
  void operator=(const svtkUniformGrid&) = delete;

  svtkEmptyCell* EmptyCell;

  static unsigned char MASKED_CELL_VALUE;
};

#endif
