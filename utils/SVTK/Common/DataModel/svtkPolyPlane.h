/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPolyPlane.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPolyPlane
 * @brief   Implicit function that is generated by extrusion of a polyline along the Z axis
 *
 * svtkPolyPlane is, as the name suggests, an extrusion of a svtkPolyLine.
 * The extrusion direction is assumed to be the Z vector. It can be used in
 * combination with a svtkCutter to cut a dataset with a polyplane.
 * svtkPolyPlane is a concrete implementation of the abstract class
 * svtkImplicitFunction.
 *
 * @todo
 * Generalize to extrusions along arbitrary directions.
 */

#ifndef svtkPolyPlane_h
#define svtkPolyPlane_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImplicitFunction.h"

class svtkPolyLine;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkPolyPlane : public svtkImplicitFunction
{
public:
  /**
   * Construct plane passing through origin and normal to z-axis.
   */
  static svtkPolyPlane* New();

  svtkTypeMacro(svtkPolyPlane, svtkImplicitFunction);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Evaluate plane equation for point x[3].
   */
  using svtkImplicitFunction::EvaluateFunction;
  double EvaluateFunction(double x[3]) override;
  //@}

  /**
   * Evaluate function gradient at point x[3].
   */
  void EvaluateGradient(double x[3], double g[3]) override;

  //@{
  /**
   * Set/get point through which plane passes. Plane is defined by point
   * and normal.
   */
  virtual void SetPolyLine(svtkPolyLine*);
  svtkGetObjectMacro(PolyLine, svtkPolyLine);
  //@}

  /**
   * Override GetMTime to include the polyline
   */
  svtkMTimeType GetMTime() override;

protected:
  svtkPolyPlane();
  ~svtkPolyPlane() override;

  void ComputeNormals();

  double ExtrusionDirection[3];
  svtkPolyLine* PolyLine;
  svtkTimeStamp NormalComputeTime;
  svtkDoubleArray* Normals;
  svtkIdType ClosestPlaneIdx;

private:
  svtkPolyPlane(const svtkPolyPlane&) = delete;
  void operator=(const svtkPolyPlane&) = delete;
};

#endif
