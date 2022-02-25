/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestImageDataOrientation.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// .NAME Test direction API for image data
// .SECTION Description
// This program tests the direction API of the image data.

#include "svtkCell.h"
#include "svtkDebugLeaks.h"
#include "svtkImageData.h"
#include "svtkMath.h"
#include "svtkMathUtilities.h"
#include "svtkMatrix4x4.h"
#include "svtkNew.h"
#include "svtkPoints.h"

inline int DoOrientationTest(
  int extent[6], double origin[3], double spacing[3], double direction[9])
{
  double tol = 10e-15;

  // Create image
  svtkNew<svtkImageData> image;
  image->SetExtent(extent);
  image->SetOrigin(origin);
  image->SetSpacing(spacing);
  image->SetDirectionMatrix(direction);
  image->AllocateScalars(SVTK_DOUBLE, 1);

  // Check some values in index to physical matrix
  svtkMatrix4x4* m4 = image->GetIndexToPhysicalMatrix();
  if (m4->GetElement(0, 3) != origin[0] || m4->GetElement(1, 3) != origin[1] ||
    m4->GetElement(2, 3) != origin[2] || m4->GetElement(3, 3) != 1)
  {

    svtkGenericWarningMacro(
      "IndexToPhysical matrix of the image data is missing the translation information");
    return EXIT_FAILURE;
  }

  // Go from min IJK to XYZ coordinates
  int i, j, k;
  i = extent[0];
  j = extent[2];
  k = extent[4];
  double xyz[3];
  image->TransformIndexToPhysicalPoint(i, j, k, xyz);

  // Test FindCell and ensure it finds the first cell (since we used IJK min)
  double pcoords[3];
  int subId = 0;
  svtkIdType cellId = image->FindCell(xyz, nullptr, 0, 0, subId, pcoords, nullptr);
  if (cellId != 0)
  {
    svtkGenericWarningMacro("FindCell returns " << cellId << ", expected 0");
    return EXIT_FAILURE;
  }
  if (!svtkMathUtilities::FuzzyCompare(pcoords[0], 0.0, tol) ||
    !svtkMathUtilities::FuzzyCompare(pcoords[1], 0.0, tol) ||
    !svtkMathUtilities::FuzzyCompare(pcoords[2], 0.0, tol))
  {
    svtkGenericWarningMacro(
      "FindCell returns the proper cell (0), but pcoords isn't equal to {0,0,0}");
    return EXIT_FAILURE;
  }

  // Test GetCell and ensure it returns the same value as XYZ above
  svtkCell* cell = image->GetCell(cellId);
  double pt[3];
  cell->GetPoints()->GetPoint(0, pt);
  if (!svtkMathUtilities::FuzzyCompare(pt[0], xyz[0], tol) ||
    !svtkMathUtilities::FuzzyCompare(pt[1], xyz[1], tol) ||
    !svtkMathUtilities::FuzzyCompare(pt[2], xyz[2], tol))
  {
    svtkGenericWarningMacro(
      "GetCell result for cell " << cellId << " does not match expected values.");
    return EXIT_FAILURE;
  }
  // Go from physical coordinate to index coordinate and ensure
  // it matches with ijk
  double index[3];
  image->TransformPhysicalPointToContinuousIndex(pt, index);
  if (!svtkMathUtilities::FuzzyCompare(index[0], (double)i, tol) ||
    !svtkMathUtilities::FuzzyCompare(index[1], (double)j, tol) ||
    !svtkMathUtilities::FuzzyCompare(index[2], (double)k, tol))
  {
    svtkGenericWarningMacro("Applying the PhysicalToIndex matrix does not return expected indices.");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int TestImageDataOrientation(int, char*[])
{
  const double pi = svtkMath::Pi();

  // test 0D, 1D, 2D, 3D data with various extents, spacings, origins, directions
  static int dims[4][3] = {
    { 1, 1, 1 },
    { 3, 1, 1 },
    { 3, 3, 1 },
    { 3, 3, 3 },
  };
  static int starts[4][3] = {
    { 0, 0, 0 },
    { -1, 0, -1 },
    { 2, 3, 6 },
    { -10, 0, 5 },
  };
  static double spacings[4][3] = {
    { 1, 1, 1 },
    { 1.0 / 7, 1, 1 },
    { 1, -1, 1 },
    { -1, 1, -1 / 13.0 },
  };
  static double origins[4][3] = {
    { 0, 0, 0 },
    { 1.0 / 13, 0, 0 },
    { 0, -1, 0 },
    { -1, 0, -1 / 7.0 },
  };
  static double directions[7][9] = {
    {
      1, 0, 0, //
      0, 1, 0, //
      0, 0, 1  //
    },
    {
      -1, 0, 0, //
      0, -1, 0, //
      0, 0, 1   //
    },
    {
      1, 0, 0, //
      0, 0, 1, //
      0, 1, 0  //
    },
    {
      0, -1, 0, //
      1, 0, 0,  //
      0, 0, 1   //
    },
    {
      1, 0, 0,                     //
      0, cos(pi / 4), sin(pi / 4), //
      0, -sin(pi / 4), cos(pi / 4) //
    },
    {
      cos(-pi / 5), sin(-pi / 5), 0,  //
      -sin(-pi / 5), cos(-pi / 5), 0, //
      0, 0, 1                         //
    },
    {
      cos(pi / 0.8), 0, sin(pi / 0.8),  //
      0, 1, 0,                          //
      -sin(pi / 0.8), 0, cos(pi / 0.8), //
    },
  };

  int extent[6];
  double* spacing;
  double* origin;
  double* direction;

  int failed = 0;

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      for (int k = 0; k < 4; k++)
      {
        spacing = spacings[k];
        for (int l = 0; l < 4; l++)
        {
          origin = origins[l];
          for (int ii = 0; ii < 3; ii++)
          {
            extent[2 * ii] = starts[i][ii];
            extent[2 * ii + 1] = starts[i][ii] + dims[j][ii] - 1;
          }

          for (int jj = 0; jj < 4; jj++)
          {
            direction = directions[jj];
            if (DoOrientationTest(extent, origin, spacing, direction) == EXIT_FAILURE)
            {
              failed = EXIT_FAILURE;
            }
          }
        }
      }
    }
  }

  return failed;
}
