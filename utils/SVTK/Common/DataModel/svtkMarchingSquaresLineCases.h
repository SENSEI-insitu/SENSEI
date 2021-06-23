/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMarchingSquaresLineCases.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef svtkMarchingSquaresLineCases_h
#define svtkMarchingSquaresLineCases_h
//
// Marching squares cases for generating isolines.
//
#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkSystemIncludes.h"

typedef int EDGE_LIST;
struct SVTKCOMMONDATAMODEL_EXPORT svtkMarchingSquaresLineCases
{
  EDGE_LIST edges[5];
  static svtkMarchingSquaresLineCases* GetCases();
};

#endif
// SVTK-HeaderTest-Exclude: svtkMarchingSquaresLineCases.h
