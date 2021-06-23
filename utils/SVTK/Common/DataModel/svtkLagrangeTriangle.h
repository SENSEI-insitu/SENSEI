/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLagrangeTriangle.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkLagrangeTriangle
 * @brief   A 2D cell that represents an arbitrary order Lagrange triangle
 *
 * svtkLagrangeTriangle is a concrete implementation of svtkCell to represent a
 * 2D triangle using Lagrange shape functions of user specified order.
 *
 * The number of points in a Lagrange cell determines the order over which they
 * are iterated relative to the parametric coordinate system of the cell. The
 * first points that are reported are vertices. They appear in the same order in
 * which they would appear in linear cells. Mid-edge points are reported next.
 * They are reported in sequence. For two- and three-dimensional (3D) cells, the
 * following set of points to be reported are face points. Finally, 3D cells
 * report points interior to their volume.
 */

#ifndef svtkLagrangeTriangle_h
#define svtkLagrangeTriangle_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkHigherOrderTriangle.h"

#include <vector> // For caching

class svtkDoubleArray;
class svtkLagrangeCurve;
class svtkTriangle;

class SVTKCOMMONDATAMODEL_EXPORT svtkLagrangeTriangle : public svtkHigherOrderTriangle
{
public:
  static svtkLagrangeTriangle* New();
  svtkTypeMacro(svtkLagrangeTriangle, svtkHigherOrderTriangle);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  int GetCellType() override { return SVTK_LAGRANGE_TRIANGLE; }

  svtkCell* GetEdge(int edgeId) override;
  void InterpolateFunctions(const double pcoords[3], double* weights) override;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override;

  virtual svtkHigherOrderCurve* getEdgeCell() override;

protected:
  svtkLagrangeTriangle();
  ~svtkLagrangeTriangle() override;

  svtkIdType GetNumberOfSubtriangles() const { return this->NumberOfSubtriangles; }
  svtkIdType ComputeNumberOfSubtriangles();
  svtkNew<svtkLagrangeCurve> EdgeCell;

private:
  svtkLagrangeTriangle(const svtkLagrangeTriangle&) = delete;
  void operator=(const svtkLagrangeTriangle&) = delete;
};

#endif
