/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBezierCurve.h

  Copyright (c) Kevin Tew
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME svtkBezierCurve
// .SECTION Description
// .SECTION See Also

#ifndef svtkBezierCurve_h
#define svtkBezierCurve_h

#include "svtkCellType.h"              // For GetCellType.
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkHigherOrderCurve.h"
#include "svtkNew.h"          // For member variable.
#include "svtkSmartPointer.h" // For member variable.

class svtkCellData;
class svtkDoubleArray;
class svtkIdList;
class svtkLine;
class svtkPointData;
class svtkPoints;
class svtkVector3d;
class svtkVector3i;
class svtkDataSet;

class SVTKCOMMONDATAMODEL_EXPORT svtkBezierCurve : public svtkHigherOrderCurve
{
public:
  static svtkBezierCurve* New();
  svtkTypeMacro(svtkBezierCurve, svtkHigherOrderCurve);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  int GetCellType() override { return SVTK_BEZIER_CURVE; }
  void EvaluateLocationProjectedNode(
    int& subId, const svtkIdType point_id, double x[3], double* weights);
  void SetRationalWeightsFromPointData(svtkPointData* point_data, const svtkIdType numPts);
  void InterpolateFunctions(const double pcoords[3], double* weights) override;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override;

  svtkDoubleArray* GetRationalWeights();

protected:
  svtkLine* GetApproximateLine(
    int subId, svtkDataArray* scalarsIn = nullptr, svtkDataArray* scalarsOut = nullptr) override;
  svtkBezierCurve();
  ~svtkBezierCurve() override;

  svtkNew<svtkDoubleArray> RationalWeights;

private:
  svtkBezierCurve(const svtkBezierCurve&) = delete;
  void operator=(const svtkBezierCurve&) = delete;
};

#endif // svtkBezierCurve_h
