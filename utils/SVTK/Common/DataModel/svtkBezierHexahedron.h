/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBezierHexahedron.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkBezierHexahedron
 * @brief   A 3D cell that represents an arbitrary order Bezier hex
 *
 * svtkBezierHexahedron is a concrete implementation of svtkCell to represent a
 * 3D hexahedron using Bezier shape functions of user specified order.
 *
 * @sa
 * svtkHexahedron
 */

#ifndef svtkBezierHexahedron_h
#define svtkBezierHexahedron_h

#include "svtkCellType.h"              // For GetCellType.
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkHigherOrderHexahedron.h"
#include "svtkNew.h"          // For member variable.
#include "svtkSmartPointer.h" // For member variable.

class svtkCellData;
class svtkDoubleArray;
class svtkHexahedron;
class svtkIdList;
class svtkBezierCurve;
class svtkBezierInterpolation;
class svtkBezierQuadrilateral;
class svtkPointData;
class svtkPoints;
class svtkVector3d;
class svtkVector3i;
class svtkDataSet;

class SVTKCOMMONDATAMODEL_EXPORT svtkBezierHexahedron : public svtkHigherOrderHexahedron
{
public:
  static svtkBezierHexahedron* New();
  svtkTypeMacro(svtkBezierHexahedron, svtkHigherOrderHexahedron);

  void PrintSelf(ostream& os, svtkIndent indent) override;
  int GetCellType() override { return SVTK_BEZIER_HEXAHEDRON; }
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int faceId) override;
  void EvaluateLocationProjectedNode(
    int& subId, const svtkIdType point_id, double x[3], double* weights);
  void InterpolateFunctions(const double pcoords[3], double* weights) override;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override;

  void SetRationalWeightsFromPointData(svtkPointData* point_data, const svtkIdType numPts);

  svtkDoubleArray* GetRationalWeights();
  virtual svtkHigherOrderCurve* getEdgeCell() override;
  virtual svtkHigherOrderQuadrilateral* getFaceCell() override;
  virtual svtkHigherOrderInterpolation* getInterp() override;

protected:
  svtkHexahedron* GetApproximateHex(
    int subId, svtkDataArray* scalarsIn = nullptr, svtkDataArray* scalarsOut = nullptr) override;
  svtkBezierHexahedron();
  ~svtkBezierHexahedron() override;

  svtkNew<svtkDoubleArray> RationalWeights;
  svtkNew<svtkBezierQuadrilateral> FaceCell;
  svtkNew<svtkBezierCurve> EdgeCell;
  svtkNew<svtkBezierInterpolation> Interp;

private:
  svtkBezierHexahedron(const svtkBezierHexahedron&) = delete;
  void operator=(const svtkBezierHexahedron&) = delete;
};

#endif // svtkBezierHexahedron_h
