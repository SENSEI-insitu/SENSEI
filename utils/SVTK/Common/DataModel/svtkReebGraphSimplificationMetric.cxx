/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkReebGraphSimplificationMetric.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkReebGraphSimplificationMetric.h"

#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkReebGraphSimplificationMetric);

//----------------------------------------------------------------------------
svtkReebGraphSimplificationMetric::svtkReebGraphSimplificationMetric()
{
  this->LowerBound = 0;
  this->UpperBound = 1;
}

//----------------------------------------------------------------------------
svtkReebGraphSimplificationMetric::~svtkReebGraphSimplificationMetric() = default;

//----------------------------------------------------------------------------
void svtkReebGraphSimplificationMetric::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Upper Bound: " << this->UpperBound << "\n";
  os << indent << "Lower Bound: " << this->LowerBound << "\n";
}

//----------------------------------------------------------------------------
double svtkReebGraphSimplificationMetric::ComputeMetric(svtkDataSet* svtkNotUsed(mesh),
  svtkDataArray* svtkNotUsed(scalarField), svtkIdType svtkNotUsed(startCriticalPoint),
  svtkAbstractArray* svtkNotUsed(vertexList), svtkIdType svtkNotUsed(endCriticalPoint))
{
  printf("too bad, wrong code\n");
  return 0;
}
