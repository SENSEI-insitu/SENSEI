/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericCellTessellator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkGenericCellTessellator
 * @brief   helper class to perform cell tessellation
 *
 * svtkGenericCellTessellator is a helper class to perform adaptive tessellation
 * of particular cell topologies. The major purpose for this class is to
 * transform higher-order cell types (e.g., higher-order finite elements)
 * into linear cells that can then be easily visualized by SVTK. This class
 * works in conjunction with the svtkGenericDataSet and svtkGenericAdaptorCell
 * classes.
 *
 * This algorithm is based on edge subdivision. An error metric along each
 * edge is evaluated, and if the error is greater than some tolerance, the
 * edge is subdivided (as well as all connected 2D and 3D cells). The process
 * repeats until the error metric is satisfied.
 *
 * A significant issue addressed by this algorithm is to insure face
 * compatibility across neighboring cells. That is, diagonals due to face
 * triangulation must match to insure that the mesh is compatible. The
 * algorithm employs a precomputed table to accelerate the tessellation
 * process. The table was generated with the help of svtkOrderedTriangulator;
 * the basic idea is that the choice of diagonal is made by considering the
 * relative value of the point ids.
 */

#ifndef svtkGenericCellTessellator_h
#define svtkGenericCellTessellator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkCellArray;
class svtkDoubleArray;
class svtkCollection;
class svtkGenericAttributeCollection;
class svtkGenericAdaptorCell;
class svtkGenericCellIterator;
class svtkPointData;
class svtkGenericDataSet;

//-----------------------------------------------------------------------------
//
// The tessellation object
class SVTKCOMMONDATAMODEL_EXPORT svtkGenericCellTessellator : public svtkObject
{
public:
  svtkTypeMacro(svtkGenericCellTessellator, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Tessellate a face of a 3D `cell'. The face is specified by the
   * index value.
   * The result is a set of smaller linear triangles in `cellArray' with
   * `points' and point data `internalPd'.
   * \pre cell_exists: cell!=0
   * \pre valid_dimension: cell->GetDimension()==3
   * \pre valid_index_range: (index>=0) && (index<cell->GetNumberOfBoundaries(2))
   * \pre att_exists: att!=0
   * \pre points_exists: points!=0
   * \pre cellArray_exists: cellArray!=0
   * \pre internalPd_exists: internalPd!=0
   */
  virtual void TessellateFace(svtkGenericAdaptorCell* cell, svtkGenericAttributeCollection* att,
    svtkIdType index, svtkDoubleArray* points, svtkCellArray* cellArray, svtkPointData* internalPd) = 0;

  /**
   * Tessellate a 3D `cell'. The result is a set of smaller linear
   * tetrahedra in `cellArray' with `points' and point data `internalPd'.
   * \pre cell_exists: cell!=0
   * \pre valid_dimension: cell->GetDimension()==3
   * \pre att_exists: att!=0
   * \pre points_exists: points!=0
   * \pre cellArray_exists: cellArray!=0
   * \pre internalPd_exists: internalPd!=0
   */
  virtual void Tessellate(svtkGenericAdaptorCell* cell, svtkGenericAttributeCollection* att,
    svtkDoubleArray* points, svtkCellArray* cellArray, svtkPointData* internalPd) = 0;

  /**
   * Triangulate a 2D `cell'. The result is a set of smaller linear triangles
   * in `cellArray' with `points' and point data `internalPd'.
   * \pre cell_exists: cell!=0
   * \pre valid_dimension: cell->GetDimension()==2
   * \pre att_exists: att!=0
   * \pre points_exists: points!=0
   * \pre cellArray_exists: cellArray!=0
   * \pre internalPd_exists: internalPd!=0
   */
  virtual void Triangulate(svtkGenericAdaptorCell* cell, svtkGenericAttributeCollection* att,
    svtkDoubleArray* points, svtkCellArray* cellArray, svtkPointData* internalPd) = 0;

  //@{
  /**
   * Specify the list of error metrics used to decide if an edge has to be
   * split or not. It is a collection of svtkGenericSubdivisionErrorMetric-s.
   */
  virtual void SetErrorMetrics(svtkCollection* someErrorMetrics);
  svtkGetObjectMacro(ErrorMetrics, svtkCollection);
  //@}

  /**
   * Initialize the tessellator with a data set `ds'.
   */
  virtual void Initialize(svtkGenericDataSet* ds) = 0;

  /**
   * Init the error metric with the dataset. Should be called in each filter
   * before any tessellation of any cell.
   */
  void InitErrorMetrics(svtkGenericDataSet* ds);

  //@{
  /**
   * If true, measure the quality of the fixed subdivision.
   */
  svtkGetMacro(Measurement, int);
  svtkSetMacro(Measurement, int);
  //@}

  /**
   * Get the maximum error measured after the fixed subdivision.
   * \pre errors_exists: errors!=0
   * \pre valid_size: sizeof(errors)==GetErrorMetrics()->GetNumberOfItems()
   */
  void GetMaxErrors(double* errors);

protected:
  svtkGenericCellTessellator();
  ~svtkGenericCellTessellator() override;

  /**
   * Does the edge need to be subdivided according to at least one error
   * metric? The edge is defined by its `leftPoint' and its `rightPoint'.
   * `leftPoint', `midPoint' and `rightPoint' have to be initialized before
   * calling RequiresEdgeSubdivision().
   * Their format is global coordinates, parametric coordinates and
   * point centered attributes: xyx rst abc de...
   * `alpha' is the normalized abscissa of the midpoint along the edge.
   * (close to 0 means close to the left point, close to 1 means close to the
   * right point)
   * \pre leftPoint_exists: leftPoint!=0
   * \pre midPoint_exists: midPoint!=0
   * \pre rightPoint_exists: rightPoint!=0
   * \pre clamped_alpha: alpha>0 && alpha<1
   * \pre valid_size: sizeof(leftPoint)=sizeof(midPoint)=sizeof(rightPoint)
   * =GetAttributeCollection()->GetNumberOfPointCenteredComponents()+6
   */
  int RequiresEdgeSubdivision(double* left, double* mid, double* right, double alpha);

  /**
   * Update the max error of each error metric according to the error at the
   * mid-point. The type of error depends on the state
   * of the concrete error metric. For instance, it can return an absolute
   * or relative error metric.
   * See RequiresEdgeSubdivision() for a description of the arguments.
   * \pre leftPoint_exists: leftPoint!=0
   * \pre midPoint_exists: midPoint!=0
   * \pre rightPoint_exists: rightPoint!=0
   * \pre clamped_alpha: alpha>0 && alpha<1
   * \pre valid_size: sizeof(leftPoint)=sizeof(midPoint)=sizeof(rightPoint)
   * =GetAttributeCollection()->GetNumberOfPointCenteredComponents()+6
   */
  virtual void UpdateMaxError(
    double* leftPoint, double* midPoint, double* rightPoint, double alpha);

  /**
   * Reset the maximal error of each error metric. The purpose of the maximal
   * error is to measure the quality of a fixed subdivision.
   */
  void ResetMaxErrors();

  /**
   * List of error metrics. Collection of svtkGenericSubdivisionErrorMetric
   */
  svtkCollection* ErrorMetrics;

  /**
   * Send the current cell to error metrics. Should be called at the beginning
   * of the implementation of Tessellate(), Triangulate()
   * or TessellateFace()
   * \pre cell_exists: cell!=0
   */
  void SetGenericCell(svtkGenericAdaptorCell* cell);

  /**
   * Dataset to be tessellated.
   */
  svtkGenericDataSet* DataSet;

  int Measurement;   // if true, measure the quality of the fixed subdivision.
  double* MaxErrors; // max error for each error metric, for measuring the
  // quality of a fixed subdivision.
  int MaxErrorsCapacity;

private:
  svtkGenericCellTessellator(const svtkGenericCellTessellator&) = delete;
  void operator=(const svtkGenericCellTessellator&) = delete;
};

#endif
