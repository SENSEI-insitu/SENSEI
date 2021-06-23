/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSmoothErrorMetric.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkSmoothErrorMetric
 * @brief   Objects that compute
 * geometry-based error during cell tessellation according to
 * some max angle.
 *
 *
 * It is a concrete error metric, based on a geometric criterium:
 * a max angle between the chord passing through the midpoint and one of the
 * endpoints and the other chord passing through the midpoint and the other
 * endpoint of the edge. It is related to the flatness of an edge.
 *
 * @sa
 * svtkGenericCellTessellator svtkGenericSubdivisionErrorMetric
 */

#ifndef svtkSmoothErrorMetric_h
#define svtkSmoothErrorMetric_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkGenericSubdivisionErrorMetric.h"

class svtkGenericDataSet;

class SVTKCOMMONDATAMODEL_EXPORT svtkSmoothErrorMetric : public svtkGenericSubdivisionErrorMetric
{
public:
  /**
   * Construct the error metric with a default flatness threshold of 90.1
   * degrees.
   */
  static svtkSmoothErrorMetric* New();

  //@{
  /**
   * Standard SVTK type and error macros.
   */
  svtkTypeMacro(svtkSmoothErrorMetric, svtkGenericSubdivisionErrorMetric);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Return the flatness threshold.
   * \post positive_result: result>90 && result<180
   */
  double GetAngleTolerance();

  /**
   * Set the flatness threshold with an angle in degrees. Internally
   * compute the cosine. value is supposed to be in ]90,180[, if not
   * it is clamped in [90.1,179.9].
   * For instance 178 will give better result than 150.
   */
  void SetAngleTolerance(double value);

  /**
   * Does the edge need to be subdivided according to the cosine between
   * the two chords passing through the mid-point and the endpoints?
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
   * Return the error at the mid-point. It will return an error relative to
   * the bounding box size if GetRelative() is true, a square absolute error
   * otherwise.
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
  svtkSmoothErrorMetric();
  ~svtkSmoothErrorMetric() override;

  double AngleTolerance;
  double CosTolerance;

private:
  svtkSmoothErrorMetric(const svtkSmoothErrorMetric&) = delete;
  void operator=(const svtkSmoothErrorMetric&) = delete;
};

#endif
