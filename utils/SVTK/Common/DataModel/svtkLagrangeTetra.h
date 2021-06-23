/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLagrangeTetra.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkLagrangeTetra
 * @brief   A 3D cell that represents an arbitrary order Lagrange tetrahedron
 *
 * svtkLagrangeTetra is a concrete implementation of svtkCell to represent a
 * 3D tetrahedron using Lagrange shape functions of user specified order.
 *
 * The number of points in a Lagrange cell determines the order over which they
 * are iterated relative to the parametric coordinate system of the cell. The
 * first points that are reported are vertices. They appear in the same order in
 * which they would appear in linear cells. Mid-edge points are reported next.
 * They are reported in sequence. For two- and three-dimensional (3D) cells, the
 * following set of points to be reported are face points. Finally, 3D cells
 * report points interior to their volume.
 */

#ifndef svtkLagrangeTetra_h
#define svtkLagrangeTetra_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkHigherOrderTetra.h"

#include <vector> // For caching

class svtkTetra;
class svtkLagrangeCurve;
class svtkLagrangeTriangle;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkLagrangeTetra : public svtkHigherOrderTetra
{
public:
  static svtkLagrangeTetra* New();
  svtkTypeMacro(svtkLagrangeTetra, svtkHigherOrderTetra);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  int GetCellType() override { return SVTK_LAGRANGE_TETRAHEDRON; }

  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int faceId) override;
  void InterpolateFunctions(const double pcoords[3], double* weights) override;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override;
  virtual svtkHigherOrderCurve* getEdgeCell() override;
  virtual svtkHigherOrderTriangle* getFaceCell() override;

protected:
  svtkLagrangeTetra();
  ~svtkLagrangeTetra() override;

  svtkNew<svtkLagrangeCurve> EdgeCell;
  svtkNew<svtkLagrangeTriangle> FaceCell;

private:
  svtkLagrangeTetra(const svtkLagrangeTetra&) = delete;
  void operator=(const svtkLagrangeTetra&) = delete;
};

#endif
