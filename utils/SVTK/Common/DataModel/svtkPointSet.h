/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPointSet.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPointSet
 * @brief   abstract class for specifying dataset behavior
 *
 * svtkPointSet is an abstract class that specifies the interface for
 * datasets that explicitly use "point" arrays to represent geometry.
 * For example, svtkPolyData, svtkUnstructuredGrid, and svtkStructuredGrid
 * require point arrays to specify point positions, while svtkImageData
 * represents point positions implicitly (and hence is not a subclass
 * of svtkImageData).
 *
 * Note: The svtkPolyData and svtkUnstructuredGrid datasets (derived classes of
 * svtkPointSet) are often used in geometric computation (e.g.,
 * svtkDelaunay2D).  In most cases during filter execution the output geometry
 * and/or topology is created once and provided as output; however in a very
 * few cases the underlying geometry/topology may be created and then
 * incrementally modified. This has implications on the use of supporting
 * classes like locators and cell links topological structures which may be
 * required to support incremental editing operations. Consequently, there is
 * a flag, Editable, that controls whether the dataset can be incrementally
 * edited after it is initially created. By default, and for performance
 * reasons, svtkPointSet derived classes are created as non-editable.  The few
 * methods that require incremental editing capabilities are documented in
 * derived classes.
 *
 * Another important feature of svtkPointSet classes is the use of an internal
 * locator to speed up certain operations like FindCell(). Depending on the
 * application and desired performance, different locators (either a cell or
 * point locator) of different locator types may be used, along with different
 * strategies for using the locators to perform various operations. See
 * the class svtkFindCellStrategy for more information
 *
 * @sa
 * svtkPolyData svtkStructuredGrid svtkUnstructuredGrid svtkFindCellStrategy
 */

#ifndef svtkPointSet_h
#define svtkPointSet_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataSet.h"

#include "svtkPoints.h" // Needed for inline methods

class svtkAbstractPointLocator;
class svtkAbstractCellLocator;

class SVTKCOMMONDATAMODEL_EXPORT svtkPointSet : public svtkDataSet
{
public:
  //@{
  /**
   * Standard methdos for type information and printing.
   */
  svtkTypeMacro(svtkPointSet, svtkDataSet);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Specify whether this dataset is editable after creation. Meaning, once
   * the points and cells are defined, can the dataset be incrementally
   * modified. By default, this dataset is non-editable (i.e., "static")
   * after construction. The reason for this is performance: cell links and
   * locators can be built (and destroyed) much faster is it is known that
   * the data is static (see svtkStaticCellLinks, svtkStaticPointLocator,
   * svtkStaticCellLocator).
   */
  svtkSetMacro(Editable, bool);
  svtkGetMacro(Editable, bool);
  svtkBooleanMacro(Editable, bool);
  //@}

  /**
   * Reset to an empty state and free any memory.
   */
  void Initialize() override;

  /**
   * Copy the geometric structure of an input point set object.
   */
  void CopyStructure(svtkDataSet* pd) override;

  //@{
  /**
   * See svtkDataSet for additional information.
   */
  svtkIdType GetNumberOfPoints() override;
  void GetPoint(svtkIdType ptId, double x[3]) override { this->Points->GetPoint(ptId, x); }
  svtkIdType FindPoint(double x[3]) override;
  svtkIdType FindPoint(double x, double y, double z) { return this->svtkDataSet::FindPoint(x, y, z); }
  svtkIdType FindCell(double x[3], svtkCell* cell, svtkIdType cellId, double tol2, int& subId,
    double pcoords[3], double* weights) override;
  svtkIdType FindCell(double x[3], svtkCell* cell, svtkGenericCell* gencell, svtkIdType cellId,
    double tol2, int& subId, double pcoords[3], double* weights) override;
  //@}

  /**
   * See svtkDataSet for additional information.
   * WARNING: Just don't use this error-prone method, the returned pointer
   * and its values are only valid as long as another method invocation is not
   * performed. Prefer GetPoint() with the return value in argument.
   */
  double* GetPoint(svtkIdType ptId) SVTK_SIZEHINT(3) override { return this->Points->GetPoint(ptId); }

  /**
   * Return an iterator that traverses the cells in this data set.
   */
  svtkCellIterator* NewCellIterator() override;

  //@{
  /**
   * Build the internal point locator . In a multi-threaded environment, call
   * this method in a single thread before using FindCell() or FindPoint().
   */
  void BuildPointLocator();
  void BuildLocator() { this->BuildPointLocator(); }
  //@}

  /**
   * Build the cell locator. In a multi-threaded environment,
   * call this method in a single thread before using FindCell().
   */
  void BuildCellLocator();

  //@{
  /**
   * Set / get an instance of svtkAbstractPointLocator which is used to
   * support the FindPoint() and FindCell() methods. By default a
   * svtkStaticPointLocator is used, unless the class is set as Editable, in
   * which case a svtkPointLocator is used.
   */
  virtual void SetPointLocator(svtkAbstractPointLocator*);
  svtkGetObjectMacro(PointLocator, svtkAbstractPointLocator);
  //@}

  //@{
  /**
   * Set / get an instance of svtkAbstractCellLocator which may be used
   * when a svtkCellLocatorStrategy is used during a FindCelloperation.
   */
  virtual void SetCellLocator(svtkAbstractCellLocator*);
  svtkGetObjectMacro(CellLocator, svtkAbstractCellLocator);
  //@}

  /**
   * Get MTime which also considers its svtkPoints MTime.
   */
  svtkMTimeType GetMTime() override;

  /**
   * Compute the (X, Y, Z)  bounds of the data.
   */
  void ComputeBounds() override;

  /**
   * Reclaim any unused memory.
   */
  void Squeeze() override;

  //@{
  /**
   * Specify point array to define point coordinates.
   */
  virtual void SetPoints(svtkPoints*);
  svtkGetObjectMacro(Points, svtkPoints);
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

  //@{
  /**
   * Overwritten to handle the data/locator loop
   */
  void Register(svtkObjectBase* o) override;
  void UnRegister(svtkObjectBase* o) override;
  //@}

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkPointSet* GetData(svtkInformation* info);
  static svtkPointSet* GetData(svtkInformationVector* v, int i = 0);
  //@}

protected:
  svtkPointSet();
  ~svtkPointSet() override;

  bool Editable;
  svtkPoints* Points;
  svtkAbstractPointLocator* PointLocator;
  svtkAbstractCellLocator* CellLocator;

  void ReportReferences(svtkGarbageCollector*) override;

private:
  void Cleanup();

  svtkPointSet(const svtkPointSet&) = delete;
  void operator=(const svtkPointSet&) = delete;
};

inline svtkIdType svtkPointSet::GetNumberOfPoints()
{
  if (this->Points)
  {
    return this->Points->GetNumberOfPoints();
  }
  else
  {
    return 0;
  }
}

#endif
