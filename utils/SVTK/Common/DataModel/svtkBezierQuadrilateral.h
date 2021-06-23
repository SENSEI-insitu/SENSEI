/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBezierQuadrilateral.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME svtkBezierQuadrilateral
// .SECTION Description
// .SECTION See Also

#ifndef svtkBezierQuadrilateral_h
#define svtkBezierQuadrilateral_h

#include "svtkCellType.h"              // For GetCellType.
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkHigherOrderQuadrilateral.h"
#include "svtkNew.h"          // For member variable.
#include "svtkSmartPointer.h" // For member variable.

class svtkCellData;
class svtkDoubleArray;
class svtkIdList;
class svtkBezierCurve;
class svtkPointData;
class svtkPoints;
class svtkQuad;
class svtkVector3d;
class svtkVector3i;
class svtkDataSet;

class SVTKCOMMONDATAMODEL_EXPORT svtkBezierQuadrilateral : public svtkHigherOrderQuadrilateral
{
public:
  static svtkBezierQuadrilateral* New();
  svtkTypeMacro(svtkBezierQuadrilateral, svtkHigherOrderQuadrilateral);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  int GetCellType() override { return SVTK_BEZIER_QUADRILATERAL; }

  svtkCell* GetEdge(int edgeId) override;
  void EvaluateLocationProjectedNode(
    int& subId, const svtkIdType point_id, double x[3], double* weights);
  void InterpolateFunctions(const double pcoords[3], double* weights) override;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override;

  void SetRationalWeightsFromPointData(svtkPointData* point_data, const svtkIdType numPts);
  svtkDoubleArray* GetRationalWeights();
  virtual svtkHigherOrderCurve* getEdgeCell() override;

protected:
  // The verion of GetApproximateQuad between Lagrange and Bezier is different because Bezier is
  // non-interpolatory
  svtkQuad* GetApproximateQuad(
    int subId, svtkDataArray* scalarsIn = nullptr, svtkDataArray* scalarsOut = nullptr) override;

  svtkBezierQuadrilateral();
  ~svtkBezierQuadrilateral() override;

  svtkNew<svtkDoubleArray> RationalWeights;
  svtkNew<svtkBezierCurve> EdgeCell;

private:
  svtkBezierQuadrilateral(const svtkBezierQuadrilateral&) = delete;
  void operator=(const svtkBezierQuadrilateral&) = delete;
};

#endif // svtkBezierQuadrilateral_h
