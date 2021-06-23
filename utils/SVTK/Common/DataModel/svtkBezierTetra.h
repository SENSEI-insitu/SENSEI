/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBezierTetra.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkBezierTetra
 * @brief   A 3D cell that represents an arbitrary order Bezier tetrahedron
 *
 * svtkBezierTetra is a concrete implementation of svtkCell to represent a
 * 3D tetrahedron using Bezier shape functions of user specified order.
 *
 * The number of points in a Bezier cell determines the order over which they
 * are iterated relative to the parametric coordinate system of the cell. The
 * first points that are reported are vertices. They appear in the same order in
 * which they would appear in linear cells. Mid-edge points are reported next.
 * They are reported in sequence. For two- and three-dimensional (3D) cells, the
 * following set of points to be reported are face points. Finally, 3D cells
 * report points interior to their volume.
 */

#ifndef svtkBezierTetra_h
#define svtkBezierTetra_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkHigherOrderTetra.h"

class svtkTetra;
class svtkBezierCurve;
class svtkBezierTriangle;
class svtkDoubleArray;
class svtkDataSet;

class SVTKCOMMONDATAMODEL_EXPORT svtkBezierTetra : public svtkHigherOrderTetra
{
public:
  static svtkBezierTetra* New();
  svtkTypeMacro(svtkBezierTetra, svtkHigherOrderTetra);

  void PrintSelf(ostream& os, svtkIndent indent) override;
  int GetCellType() override { return SVTK_BEZIER_TETRAHEDRON; }
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int faceId) override;
  void EvaluateLocationProjectedNode(
    int& subId, const svtkIdType point_id, double x[3], double* weights);
  void SetRationalWeightsFromPointData(svtkPointData* point_data, const svtkIdType numPts);
  void InterpolateFunctions(const double pcoords[3], double* weights) override;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override;

  virtual svtkHigherOrderCurve* getEdgeCell() override;
  virtual svtkHigherOrderTriangle* getFaceCell() override;

  svtkDoubleArray* GetRationalWeights();

protected:
  svtkBezierTetra();
  ~svtkBezierTetra() override;
  svtkNew<svtkBezierCurve> EdgeCell;
  svtkNew<svtkBezierTriangle> FaceCell;
  svtkNew<svtkDoubleArray> RationalWeights;

private:
  svtkBezierTetra(const svtkBezierTetra&) = delete;
  void operator=(const svtkBezierTetra&) = delete;
};

#endif
