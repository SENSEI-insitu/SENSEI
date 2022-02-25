/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLagrangeWedge.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkLagrangeWedge
 * @brief   A 3D cell that represents an arbitrary order Lagrange wedge
 *
 * svtkLagrangeWedge is a concrete implementation of svtkCell to represent a
 * 3D wedge using Lagrange shape functions of user specified order.
 * A wedge consists of two triangular and three quadrilateral faces.
 * The first six points of the wedge (0-5) are the "corner" points
 * where the first three points are the base of the wedge. This wedge
 * point ordering is opposite the svtkWedge ordering though in that
 * the base of the wedge defined by the first three points (0,1,2) form
 * a triangle whose normal points inward (toward the triangular face (3,4,5)).
 * While this is opposite the svtkWedge convention it is consistent with
 * every other cell type in SVTK. The first 2 parametric coordinates of the
 * Lagrange wedge or for the triangular base and vary between 0 and 1. The
 * third parametric coordinate is between the two triangular faces and goes
 * from 0 to 1 as well.
 */

#ifndef svtkLagrangeWedge_h
#define svtkLagrangeWedge_h

#include "svtkCellType.h"              // For GetCellType.
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkHigherOrderWedge.h"
#include "svtkNew.h"          // For member variable.
#include "svtkSmartPointer.h" // For member variable.

class svtkCellData;
class svtkDoubleArray;
class svtkWedge;
class svtkIdList;
class svtkPointData;
class svtkPoints;
class svtkVector3d;
class svtkVector3i;
class svtkLagrangeCurve;
class svtkLagrangeInterpolation;
class svtkLagrangeQuadrilateral;
class svtkLagrangeTriangle;

class SVTKCOMMONDATAMODEL_EXPORT svtkLagrangeWedge : public svtkHigherOrderWedge
{
public:
  static svtkLagrangeWedge* New();
  svtkTypeMacro(svtkLagrangeWedge, svtkHigherOrderWedge);

  void PrintSelf(ostream& os, svtkIndent indent) override;
  int GetCellType() override { return SVTK_LAGRANGE_WEDGE; }
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int faceId) override;
  void InterpolateFunctions(const double pcoords[3], double* weights) override;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override;

  virtual svtkHigherOrderQuadrilateral* getBdyQuad() override;
  virtual svtkHigherOrderTriangle* getBdyTri() override;
  virtual svtkHigherOrderCurve* getEdgeCell() override;
  virtual svtkHigherOrderInterpolation* getInterp() override;

protected:
  svtkLagrangeWedge();
  ~svtkLagrangeWedge() override;

  svtkNew<svtkLagrangeQuadrilateral> BdyQuad;
  svtkNew<svtkLagrangeTriangle> BdyTri;
  svtkNew<svtkLagrangeCurve> BdyEdge;
  svtkNew<svtkLagrangeInterpolation> Interp;
  svtkNew<svtkLagrangeCurve> EdgeCell;

private:
  svtkLagrangeWedge(const svtkLagrangeWedge&) = delete;
  void operator=(const svtkLagrangeWedge&) = delete;
};

#endif // svtkLagrangeWedge_h
