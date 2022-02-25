/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestMatrix3x3.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkMathUtilities.h"
#include "svtkMatrix3x3.h"
#include "svtkNew.h"
#include "svtkPoints2D.h"
#include "svtkTransform2D.h"

int TestMatrix3x3(int, char*[])
{
  // Instantiate a svtkMatrix3x3 and test out the functions.
  svtkNew<svtkMatrix3x3> matrix;
  cout << "Testing svtkMatrix3x3..." << endl;
  if (!matrix->IsIdentity())
  {
    svtkGenericWarningMacro("Matrix should be initialized to identity.");
    return 1;
  }
  matrix->Invert();
  if (!matrix->IsIdentity())
  {
    svtkGenericWarningMacro("Inverse of identity should be identity.");
    return 1;
  }
  // Check copying and comparison
  svtkNew<svtkMatrix3x3> matrix2;
  matrix2->DeepCopy(matrix);
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
      if (matrix->GetElement(i, j) != matrix2->GetElement(i, j))
      {
        svtkGenericWarningMacro("DeepCopy of svtkMatrix3x3 failed.");
        return 1;
      }
  }
  matrix2->SetElement(0, 0, 5.0);
  if (!svtkMathUtilities::FuzzyCompare(matrix2->GetElement(0, 0), 5.0))
  {
    svtkGenericWarningMacro("Value not stored in matrix properly.");
    return 1;
  }
  matrix2->SetElement(1, 2, 42.0);
  if (!svtkMathUtilities::FuzzyCompare(matrix2->GetElement(1, 2), 42.0))
  {
    svtkGenericWarningMacro("Value not stored in matrix properly.");
    return 1;
  }

  // Test matrix transpose
  matrix2->Transpose();
  if (!svtkMathUtilities::FuzzyCompare(matrix2->GetElement(0, 0), 5.0) ||
    !svtkMathUtilities::FuzzyCompare(matrix2->GetElement(2, 1), 42.0))
  {
    svtkGenericWarningMacro("svtkMatrix::Transpose failed.");
    return 1;
  }

  matrix2->Invert();
  if (!svtkMathUtilities::FuzzyCompare(matrix2->GetElement(0, 0), 0.2) ||
    !svtkMathUtilities::FuzzyCompare(matrix2->GetElement(2, 1), -42.0))
  {
    svtkGenericWarningMacro("svtkMatrix::Invert failed.");
    return 1;
  }

  // Multiply a coordinate by this matrix.
  double inout[3] = { 12.3, 45.6, 78.9 };
  matrix2->MultiplyPoint(inout, inout);

  if (!svtkMathUtilities::FuzzyCompare(inout[0], 2.46, 1e-5) ||
    !svtkMathUtilities::FuzzyCompare(inout[1], 45.6, 1e-5) ||
    !svtkMathUtilities::FuzzyCompare(inout[2], -1836.3, 1e-5))
  {
    svtkGenericWarningMacro("svtkMatrix::MultiplyPoint failed.");
    return 1;
  }

  // Not test the 2D transform with some 2D points
  svtkNew<svtkTransform2D> transform;
  svtkNew<svtkPoints2D> points;
  svtkNew<svtkPoints2D> points2;
  points->SetNumberOfPoints(3);
  points->SetPoint(0, 0.0, 0.0);
  points->SetPoint(1, 3.0, 4.9);
  points->SetPoint(2, 42.0, 69.0);

  transform->TransformPoints(points, points2);
  for (int i = 0; i < 3; ++i)
  {
    double p1[2], p2[2];
    points->GetPoint(i, p1);
    points2->GetPoint(i, p2);
    if (!svtkMathUtilities::FuzzyCompare(p1[0], p2[0], 1e-5) ||
      !svtkMathUtilities::FuzzyCompare(p1[1], p2[1], 1e-5))
    {
      svtkGenericWarningMacro("Identity transform moved points."
        << " Delta: " << p1[0] - (p2[0] - 2.0) << ", " << p1[1] - (p2[1] - 6.9));
      return 1;
    }
  }
  transform->Translate(2.0, 6.9);
  transform->TransformPoints(points, points2);
  for (int i = 0; i < 3; ++i)
  {
    double p1[2], p2[2];
    points->GetPoint(i, p1);
    points2->GetPoint(i, p2);
    if (!svtkMathUtilities::FuzzyCompare(p1[0], p2[0] - 2.0, 1e-5) ||
      !svtkMathUtilities::FuzzyCompare(p1[1], p2[1] - 6.9, 1e-5))
    {
      svtkGenericWarningMacro("Translation transform failed. Delta: " << p1[0] -
          (p2[0] - 2.0) << ", " << p1[1] - (p2[1] - 6.9));

      return 1;
    }
  }
  transform->InverseTransformPoints(points2, points2);
  for (int i = 0; i < 3; ++i)
  {
    double p1[2], p2[2];
    points->GetPoint(i, p1);
    points2->GetPoint(i, p2);
    if (!svtkMathUtilities::FuzzyCompare(p1[0], p2[0], 1e-5) ||
      !svtkMathUtilities::FuzzyCompare(p1[1], p2[1], 1e-5))
    {
      svtkGenericWarningMacro("Inverse transform did not return original points."
        << " Delta: " << p1[0] - (p2[0] - 2.0) << ", " << p1[1] - (p2[1] - 6.9));
      return 1;
    }
  }

  // Zero out the matrix.
  matrix->Zero();
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      if (matrix->GetElement(i, j) != 0.0)
      {
        svtkGenericWarningMacro("svtkMatrix::Zero failed.");
        return 1;
      }
    }
  }

  return 0;
}
