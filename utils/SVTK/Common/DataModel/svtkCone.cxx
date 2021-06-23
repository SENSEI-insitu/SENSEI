/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCone.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkCone.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkCone);

// Construct cone with angle of 45 degrees.
svtkCone::svtkCone()
{
  this->Angle = 45.0;
}

// Evaluate cone equation.
double svtkCone::EvaluateFunction(double x[3])
{
  double tanTheta = tan(svtkMath::RadiansFromDegrees(this->Angle));
  return x[1] * x[1] + x[2] * x[2] - x[0] * x[0] * tanTheta * tanTheta;
}

// Evaluate cone normal.
void svtkCone::EvaluateGradient(double x[3], double g[3])
{
  double tanTheta = tan(svtkMath::RadiansFromDegrees(this->Angle));
  g[0] = -2.0 * x[0] * tanTheta * tanTheta;
  g[1] = 2.0 * x[1];
  g[2] = 2.0 * x[2];
}

void svtkCone::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Angle: " << this->Angle << "\n";
}
