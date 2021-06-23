/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAttributesErrorMetric.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkAttributesErrorMetric
 * @brief    Objects that compute
 * attribute-based error during cell tessellation.
 *
 *
 * It is a concrete error metric, based on an attribute criterium:
 * the variation of the active attribute/component value from a linear ramp
 *
 * @sa
 * svtkGenericCellTessellator svtkGenericSubdivisionErrorMetric
 */

#ifndef svtkAttributesErrorMetric_h
#define svtkAttributesErrorMetric_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkGenericSubdivisionErrorMetric.h"

class svtkGenericAttributeCollection;
class svtkGenericDataSet;

class SVTKCOMMONDATAMODEL_EXPORT svtkAttributesErrorMetric : public svtkGenericSubdivisionErrorMetric
{
public:
  /**
   * Construct the error metric with a default relative attribute accuracy
   * equal to 0.1.
   */
  static svtkAttributesErrorMetric* New();

  //@{
  /**
   * Standard SVTK type and error macros.
   */
  svtkTypeMacro(svtkAttributesErrorMetric, svtkGenericSubdivisionErrorMetric);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  //@{
  /**
   * Absolute tolerance of the active scalar (attribute+component).
   * Subdivision is required if the square distance between the real attribute
   * at the mid point on the edge and the interpolated attribute is greater
   * than AbsoluteAttributeTolerance.
   * This is the attribute accuracy.
   * 0.01 will give better result than 0.1.
   */
  svtkGetMacro(AbsoluteAttributeTolerance, double);
  //@}

  /**
   * Set the absolute attribute accuracy to `value'. See
   * GetAbsoluteAttributeTolerance() for details.
   * It is particularly useful when some concrete implementation of
   * svtkGenericAttribute does not support GetRange() request, called
   * internally in SetAttributeTolerance(). It may happen when the
   * implementation support higher order attributes but
   * cannot compute the range.
   * \pre valid_range_value: value>0
   */
  void SetAbsoluteAttributeTolerance(double value);

  //@{
  /**
   * Relative tolerance of the active scalar (attribute+component).
   * Subdivision is required if the square distance between the real attribute
   * at the mid point on the edge and the interpolated attribute is greater
   * than AttributeTolerance.
   * This is the attribute accuracy.
   * 0.01 will give better result than 0.1.
   */
  svtkGetMacro(AttributeTolerance, double);
  //@}

  /**
   * Set the relative attribute accuracy to `value'. See
   * GetAttributeTolerance() for details.
   * \pre valid_range_value: value>0 && value<1
   */
  void SetAttributeTolerance(double value);

  /**
   * Does the edge need to be subdivided according to the distance between
   * the value of the active attribute/component at the midpoint and the mean
   * value between the endpoints?
   * The edge is defined by its `leftPoint' and its `rightPoint'.
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
  int RequiresEdgeSubdivision(
    double* leftPoint, double* midPoint, double* rightPoint, double alpha) override;

  /**
   * Return the error at the mid-point. The type of error depends on the state
   * of the concrete error metric. For instance, it can return an absolute
   * or relative error metric.
   * See RequiresEdgeSubdivision() for a description of the arguments.
   * \pre leftPoint_exists: leftPoint!=0
   * \pre midPoint_exists: midPoint!=0
   * \pre rightPoint_exists: rightPoint!=0
   * \pre clamped_alpha: alpha>0 && alpha<1
   * \pre valid_size: sizeof(leftPoint)=sizeof(midPoint)=sizeof(rightPoint)
   * =GetAttributeCollection()->GetNumberOfPointCenteredComponents()+6
   * \post positive_result: result>=0
   */
  double GetError(double* leftPoint, double* midPoint, double* rightPoint, double alpha) override;

protected:
  svtkAttributesErrorMetric();
  ~svtkAttributesErrorMetric() override;

  /**
   * Compute the square absolute attribute tolerance, only if the cached value
   * is obsolete.
   */
  void ComputeSquareAbsoluteAttributeTolerance();

  double AttributeTolerance;

  double SquareAbsoluteAttributeTolerance; // cached value computed from
  // AttributeTolerance and active attribute/component

  double AbsoluteAttributeTolerance;
  int DefinedByAbsolute;

  svtkTimeStamp SquareAbsoluteAttributeToleranceComputeTime;

  double Range; // cached value computed from active attribute/component

  svtkGenericAttributeCollection* AttributeCollection;

private:
  svtkAttributesErrorMetric(const svtkAttributesErrorMetric&) = delete;
  void operator=(const svtkAttributesErrorMetric&) = delete;
};

#endif
