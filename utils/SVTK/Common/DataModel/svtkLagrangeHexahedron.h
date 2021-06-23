/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLagrangeHexahedron.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkLagrangeHexahedron
 * @brief   A 3D cell that represents an arbitrary order Lagrange hex
 *
 * svtkLagrangeHexahedron is a concrete implementation of svtkCell to represent a
 * 3D hexahedron using Lagrange shape functions of user specified order.
 *
 * @sa
 * svtkHexahedron
 */

#ifndef svtkLagrangeHexahedron_h
#define svtkLagrangeHexahedron_h

#include "svtkCellType.h"              // For GetCellType.
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkHigherOrderHexahedron.h"
#include "svtkNew.h"          // For member variable.
#include "svtkSmartPointer.h" // For member variable.

class svtkCellData;
class svtkDoubleArray;
class svtkHexahedron;
class svtkIdList;
class svtkLagrangeCurve;
class svtkLagrangeInterpolation;
class svtkLagrangeQuadrilateral;
class svtkPointData;
class svtkPoints;
class svtkVector3d;
class svtkVector3i;

class SVTKCOMMONDATAMODEL_EXPORT svtkLagrangeHexahedron : public svtkHigherOrderHexahedron
{
public:
  static svtkLagrangeHexahedron* New();
  svtkTypeMacro(svtkLagrangeHexahedron, svtkHigherOrderHexahedron);

  void PrintSelf(ostream& os, svtkIndent indent) override;
  int GetCellType() override { return SVTK_LAGRANGE_HEXAHEDRON; }
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int faceId) override;
  void InterpolateFunctions(const double pcoords[3], double* weights) override;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override;
  virtual svtkHigherOrderCurve* getEdgeCell() override;
  virtual svtkHigherOrderQuadrilateral* getFaceCell() override;
  virtual svtkHigherOrderInterpolation* getInterp() override;

protected:
  svtkHexahedron* GetApproximateHex(
    int subId, svtkDataArray* scalarsIn = nullptr, svtkDataArray* scalarsOut = nullptr) override;
  svtkLagrangeHexahedron();
  ~svtkLagrangeHexahedron() override;

  svtkNew<svtkLagrangeQuadrilateral> FaceCell;
  svtkNew<svtkLagrangeCurve> EdgeCell;
  svtkNew<svtkLagrangeInterpolation> Interp;

private:
  svtkLagrangeHexahedron(const svtkLagrangeHexahedron&) = delete;
  void operator=(const svtkLagrangeHexahedron&) = delete;
};

#endif // svtkLagrangeHexahedron_h
