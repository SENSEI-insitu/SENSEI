/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBezierWedge.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkBezierWedge
 * @brief   A 3D cell that represents an arbitrary order Bezier wedge
 *
 * svtkBezierWedge is a concrete implementation of svtkCell to represent a
 * 3D wedge using Bezier shape functions of user specified order.
 * A wedge consists of two triangular and three quadrilateral faces.
 * The first six points of the wedge (0-5) are the "corner" points
 * where the first three points are the base of the wedge. This wedge
 * point ordering is opposite the svtkWedge ordering though in that
 * the base of the wedge defined by the first three points (0,1,2) form
 * a triangle whose normal points inward (toward the triangular face (3,4,5)).
 * While this is opposite the svtkWedge convention it is consistent with
 * every other cell type in SVTK. The first 2 parametric coordinates of the
 * Bezier wedge or for the triangular base and vary between 0 and 1. The
 * third parametric coordinate is between the two triangular faces and goes
 * from 0 to 1 as well.
 */

#ifndef svtkBezierWedge_h
#define svtkBezierWedge_h

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
class svtkBezierCurve;
class svtkBezierInterpolation;
class svtkBezierQuadrilateral;
class svtkBezierTriangle;
class svtkDataSet;

class SVTKCOMMONDATAMODEL_EXPORT svtkBezierWedge : public svtkHigherOrderWedge
{
public:
  static svtkBezierWedge* New();
  svtkTypeMacro(svtkBezierWedge, svtkHigherOrderWedge);

  void PrintSelf(ostream& os, svtkIndent indent) override;
  int GetCellType() override { return SVTK_BEZIER_WEDGE; }
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int faceId) override;
  void EvaluateLocationProjectedNode(
    int& subId, const svtkIdType point_id, double x[3], double* weights);
  void InterpolateFunctions(const double pcoords[3], double* weights) override;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override;

  void SetRationalWeightsFromPointData(svtkPointData* point_data, const svtkIdType numPts);

  virtual svtkHigherOrderQuadrilateral* getBdyQuad() override;
  virtual svtkHigherOrderTriangle* getBdyTri() override;
  virtual svtkHigherOrderCurve* getEdgeCell() override;
  virtual svtkHigherOrderInterpolation* getInterp() override;

  svtkDoubleArray* GetRationalWeights();

protected:
  svtkBezierWedge();
  ~svtkBezierWedge() override;

  svtkNew<svtkDoubleArray> RationalWeights;
  svtkNew<svtkBezierQuadrilateral> BdyQuad;
  svtkNew<svtkBezierTriangle> BdyTri;
  svtkNew<svtkBezierCurve> BdyEdge;
  svtkNew<svtkBezierInterpolation> Interp;
  svtkNew<svtkBezierCurve> EdgeCell;

private:
  svtkBezierWedge(const svtkBezierWedge&) = delete;
  void operator=(const svtkBezierWedge&) = delete;
};

#endif // svtkBezierWedge_h
