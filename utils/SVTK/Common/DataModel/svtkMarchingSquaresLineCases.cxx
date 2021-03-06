/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMarchingSquaresLineCases.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkMarchingSquaresLineCases.h"

// Note: the following code is placed here to deal with cross-library
// symbol export and import on Microsoft compilers.
static svtkMarchingSquaresLineCases SVTK_MARCHING_SQUARES_LINECASES[] = {
  { { -1, -1, -1, -1, -1 } },
  { { 0, 3, -1, -1, -1 } },
  { { 1, 0, -1, -1, -1 } },
  { { 1, 3, -1, -1, -1 } },
  { { 2, 1, -1, -1, -1 } },
  { { 0, 3, 2, 1, -1 } },
  { { 2, 0, -1, -1, -1 } },
  { { 2, 3, -1, -1, -1 } },
  { { 3, 2, -1, -1, -1 } },
  { { 0, 2, -1, -1, -1 } },
  { { 1, 0, 3, 2, -1 } },
  { { 1, 2, -1, -1, -1 } },
  { { 3, 1, -1, -1, -1 } },
  { { 0, 1, -1, -1, -1 } },
  { { 3, 0, -1, -1, -1 } },
  { { -1, -1, -1, -1, -1 } },
};

svtkMarchingSquaresLineCases* svtkMarchingSquaresLineCases::GetCases()
{
  return SVTK_MARCHING_SQUARES_LINECASES;
}
