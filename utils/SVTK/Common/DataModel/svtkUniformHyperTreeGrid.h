/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUniformHyperTreeGrid.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkUniformHyperTreeGrid
 * @brief   A specifalized type of svtkHyperTreeGrid for the case
 * when root cells have uniform sizes in each direction
 *
 * @sa
 * svtkHyperTree svtkHyperTreeGrid svtkRectilinearGrid
 *
 * @par Thanks:
 * This class was written by Philippe Pebay, NexGen Analytics 2017
 * JB modify for introduce Scales by Jacques-Bernard Lekien, CEA 2018.
 * This work was supported by Commissariat a l'Energie Atomique
 * CEA, DAM, DIF, F-91297 Arpajon, France.
 */

#ifndef svtkUniformHyperTreeGrid_h
#define svtkUniformHyperTreeGrid_h

#include "limits.h" // UINT_MAX

#include <memory> // std::shared_ptr

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkHyperTreeGrid.h"

class svtkDoubleArray;
class svtkHyperTreeGridScales;

class SVTKCOMMONDATAMODEL_EXPORT svtkUniformHyperTreeGrid : public svtkHyperTreeGrid
{
public:
  static svtkUniformHyperTreeGrid* New();
  svtkTypeMacro(svtkUniformHyperTreeGrid, svtkHyperTreeGrid);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return what type of dataset this is.
   */
  int GetDataObjectType() override { return SVTK_UNIFORM_HYPER_TREE_GRID; }

  /**
   * Copy the internal geometric and topological structure of a
   * svtkUniformHyperTreeGrid object.
   */
  void CopyStructure(svtkDataObject*) override;

  virtual void Initialize() override;

  //@{
  /**
   * Set/Get origin of grid
   */
  svtkSetVector3Macro(Origin, double);
  svtkGetVector3Macro(Origin, double);
  //@}

  //@{
  /**
   * Set/Get scale of root cells along each direction
   */
  void SetGridScale(double, double, double);
  void SetGridScale(double*);
  svtkGetVector3Macro(GridScale, double);
  //@}

  /**
   * Set all scales at once when root cells are d-cubes
   */
  void SetGridScale(double);

  /**
   * Return a pointer to the geometry bounding box in the form
   * (xmin,xmax, ymin,ymax, zmin,zmax).
   * THIS METHOD IS NOT THREAD SAFE.
   */
  double* GetBounds() SVTK_SIZEHINT(6) override;

  //@{
  /**
   * Set/Get the grid coordinates in the x-direction.
   * NB: Set method deactivated in the case of uniform grids.
   * Use SetSize() instead.
   */
  void SetXCoordinates(svtkDataArray* XCoordinates) override;
  svtkDataArray* GetXCoordinates() override;
  /* JB A faire pour les Get !
  const svtkDataArray* GetXCoordinates() const override {
    throw std::domain_error("Cannot use GetZCoordinates on UniformHyperTreeGrid");
  }
  */
  //@}

  //@{
  /**
   * Set/Get the grid coordinates in the y-direction.
   * NB: Set method deactivated in the case of uniform grids.
   * Use SetSize() instead.
   */
  void SetYCoordinates(svtkDataArray* YCoordinates) override;
  svtkDataArray* GetYCoordinates() override;
  /* JB A faire pour les Get !
  const svtkDataArray* GetYCoordinates() const override {
    throw std::domain_error("Cannot use GetZCoordinates on UniformHyperTreeGrid");
  }
  */
  //@}

  //@{
  /**
   * Set/Get the grid coordinates in the z-direction.
   * NB: Set method deactivated in the case of uniform grids.
   * Use SetSize() instead.
   */
  void SetZCoordinates(svtkDataArray* ZCoordinates) override;
  svtkDataArray* GetZCoordinates() override;
  /* JB A faire pour les Get !
  const svtkDataArray* GetZCoordinates() const override {
    throw std::domain_error("Cannot use GetZCoordinates on UniformHyperTreeGrid");
  }
  */
  // JB A faire pour les autre Get !
  //@}

  //@{
  /**
   * JB Augented services on Coordinates.
   */
  void CopyCoordinates(const svtkHyperTreeGrid* output) override;
  void SetFixedCoordinates(unsigned int axis, double value) override;
  //@}

  /**
   * Convert the global index of a root to its Spacial coordinates origin and size.
   */
  void GetLevelZeroOriginAndSizeFromIndex(svtkIdType, double*, double*) override;

  /**
   * Convert the global index of a root to its Spacial coordinates origin and size.
   */
  void GetLevelZeroOriginFromIndex(svtkIdType, double*) override;

  /**
   * Create shallow copy of hyper tree grid.
   */
  void ShallowCopy(svtkDataObject*) override;

  /**
   * Create deep copy of hyper tree grid.
   */
  void DeepCopy(svtkDataObject*) override;

  /**
   * Return the actual size of the data bytes
   */
  unsigned long GetActualMemorySizeBytes() override;

  /**
   * Return tree located at given index of hyper tree grid
   * NB: This will construct a new HyperTree if grid slot is empty.
   */
  svtkHyperTree* GetTree(svtkIdType, bool create = false) override;

protected:
  /**
   * Constructor
   */
  svtkUniformHyperTreeGrid();

  /**
   * Destructor
   */
  ~svtkUniformHyperTreeGrid() override;

  /**
   * Grid Origin
   */
  double Origin[3];

  /**
   * Element sizes in each direction
   */
  double GridScale[3];

  //@{
  /**
   * Keep track of whether coordinates have been explicitly computed
   */
  bool ComputedXCoordinates;
  bool ComputedYCoordinates;
  bool ComputedZCoordinates;
  //@}

  unsigned int FindDichotomicX(double value) const override
  {
    if (value < this->Origin[0] ||
      value > this->Origin[0] + this->GridScale[0] * (this->GetDimensions()[0] - 1))
    {
      return UINT_MAX;
    }
    return round((value - this->Origin[0]) / this->GridScale[0]);
  }
  unsigned int FindDichotomicY(double value) const override
  {
    if (value < this->Origin[1] ||
      value > this->Origin[1] + this->GridScale[1] * (this->GetDimensions()[1] - 1))
    {
      return UINT_MAX;
    }
    return round((value - this->Origin[1]) / this->GridScale[1]);
  }
  unsigned int FindDichotomicZ(double value) const override
  {
    if (value < this->Origin[2] ||
      value > this->Origin[2] + this->GridScale[2] * (this->GetDimensions()[2] - 1))
    {
      return UINT_MAX;
    }
    return round((value - this->Origin[2]) / this->GridScale[2]);
  }

  /**
   * JB Storage of pre-computed per-level cell scales
   */
  mutable std::shared_ptr<svtkHyperTreeGridScales> Scales;

private:
  svtkUniformHyperTreeGrid(const svtkUniformHyperTreeGrid&) = delete;
  void operator=(const svtkUniformHyperTreeGrid&) = delete;
};

#endif
