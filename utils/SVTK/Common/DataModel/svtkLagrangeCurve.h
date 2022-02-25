/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLagrangeCurve.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME svtkLagrangeCurve
// .SECTION Description
// .SECTION See Also

#ifndef svtkLagrangeCurve_h
#define svtkLagrangeCurve_h

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

class SVTKCOMMONDATAMODEL_EXPORT svtkLagrangeCurve : public svtkHigherOrderCurve
{
public:
  static svtkLagrangeCurve* New();
  svtkTypeMacro(svtkLagrangeCurve, svtkHigherOrderCurve);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  int GetCellType() override { return SVTK_LAGRANGE_CURVE; }

  void InterpolateFunctions(const double pcoords[3], double* weights) override;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override;

protected:
  svtkLine* GetApproximateLine(
    int subId, svtkDataArray* scalarsIn = nullptr, svtkDataArray* scalarsOut = nullptr) override;
  svtkLagrangeCurve();
  ~svtkLagrangeCurve() override;

private:
  svtkLagrangeCurve(const svtkLagrangeCurve&) = delete;
  void operator=(const svtkLagrangeCurve&) = delete;
};

#endif // svtkLagrangeCurve_h
