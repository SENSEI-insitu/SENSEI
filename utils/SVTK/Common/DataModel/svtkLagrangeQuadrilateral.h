/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLagrangeQuadrilateral.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME svtkLagrangeQuadrilateral
// .SECTION Description
// .SECTION See Also

#ifndef svtkLagrangeQuadrilateral_h
#define svtkLagrangeQuadrilateral_h

#include "svtkCellType.h"              // For GetCellType.
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkHigherOrderQuadrilateral.h"
#include "svtkNew.h"          // For member variable.
#include "svtkSmartPointer.h" // For member variable.

class svtkCellData;
class svtkDoubleArray;
class svtkIdList;
class svtkLagrangeCurve;
class svtkPointData;
class svtkPoints;
class svtkQuad;
class svtkVector3d;
class svtkVector3i;

class SVTKCOMMONDATAMODEL_EXPORT svtkLagrangeQuadrilateral : public svtkHigherOrderQuadrilateral
{
public:
  static svtkLagrangeQuadrilateral* New();
  svtkTypeMacro(svtkLagrangeQuadrilateral, svtkHigherOrderQuadrilateral);

  void PrintSelf(ostream& os, svtkIndent indent) override;
  int GetCellType() override { return SVTK_LAGRANGE_QUADRILATERAL; }

  svtkCell* GetEdge(int edgeId) override;
  void InterpolateFunctions(const double pcoords[3], double* weights) override;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override;
  virtual svtkHigherOrderCurve* getEdgeCell() override;

protected:
  // The verion of GetApproximateQuad between Lagrange and Bezier is different because Bezier is
  // non-interpolatory
  svtkQuad* GetApproximateQuad(
    int subId, svtkDataArray* scalarsIn = nullptr, svtkDataArray* scalarsOut = nullptr) override;

  svtkLagrangeQuadrilateral();
  ~svtkLagrangeQuadrilateral() override;

  svtkNew<svtkLagrangeCurve> EdgeCell;

private:
  svtkLagrangeQuadrilateral(const svtkLagrangeQuadrilateral&) = delete;
  void operator=(const svtkLagrangeQuadrilateral&) = delete;
};

#endif // svtkLagrangeQuadrilateral_h
