/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCellType.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCellType
 * @brief   define types of cells
 *
 * svtkCellType defines the allowable cell types in the visualization
 * library (svtk). In svtk, datasets consist of collections of cells.
 * Different datasets consist of different cell types. The cells may be
 * explicitly represented (as in svtkPolyData), or may be implicit to the
 * data type (as in svtkStructuredPoints).
 */

#ifndef svtkCellType_h
#define svtkCellType_h

// To add a new cell type, define a new integer type flag here, then
// create a subclass of svtkCell to implement the proper behavior. You
// may have to modify the following methods: svtkDataSet (and subclasses)
// GetCell() and svtkGenericCell::SetCellType(). Also, to do the job right,
// you'll also have to modify some filters (svtkGeometryFilter...) and
// regression tests (example scripts) to reflect the new cell addition.
// Also, make sure to update svtkCellTypesStrings in svtkCellTypes.cxx
// and the svtkCellTypes::IsLinear method in svtkCellTypes.h.

// .SECTION Caveats
// An unstructured grid stores the types of its cells as a
// unsigned char array. Therefore, the maximum encoding number for a cell type
// is 255.

typedef enum
{
  // Linear cells
  SVTK_EMPTY_CELL = 0,
  SVTK_VERTEX = 1,
  SVTK_POLY_VERTEX = 2,
  SVTK_LINE = 3,
  SVTK_POLY_LINE = 4,
  SVTK_TRIANGLE = 5,
  SVTK_TRIANGLE_STRIP = 6,
  SVTK_POLYGON = 7,
  SVTK_PIXEL = 8,
  SVTK_QUAD = 9,
  SVTK_TETRA = 10,
  SVTK_VOXEL = 11,
  SVTK_HEXAHEDRON = 12,
  SVTK_WEDGE = 13,
  SVTK_PYRAMID = 14,
  SVTK_PENTAGONAL_PRISM = 15,
  SVTK_HEXAGONAL_PRISM = 16,

  // Quadratic, isoparametric cells
  SVTK_QUADRATIC_EDGE = 21,
  SVTK_QUADRATIC_TRIANGLE = 22,
  SVTK_QUADRATIC_QUAD = 23,
  SVTK_QUADRATIC_POLYGON = 36,
  SVTK_QUADRATIC_TETRA = 24,
  SVTK_QUADRATIC_HEXAHEDRON = 25,
  SVTK_QUADRATIC_WEDGE = 26,
  SVTK_QUADRATIC_PYRAMID = 27,
  SVTK_BIQUADRATIC_QUAD = 28,
  SVTK_TRIQUADRATIC_HEXAHEDRON = 29,
  SVTK_QUADRATIC_LINEAR_QUAD = 30,
  SVTK_QUADRATIC_LINEAR_WEDGE = 31,
  SVTK_BIQUADRATIC_QUADRATIC_WEDGE = 32,
  SVTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33,
  SVTK_BIQUADRATIC_TRIANGLE = 34,

  // Cubic, isoparametric cell
  SVTK_CUBIC_LINE = 35,

  // Special class of cells formed by convex group of points
  SVTK_CONVEX_POINT_SET = 41,

  // Polyhedron cell (consisting of polygonal faces)
  SVTK_POLYHEDRON = 42,

  // Higher order cells in parametric form
  SVTK_PARAMETRIC_CURVE = 51,
  SVTK_PARAMETRIC_SURFACE = 52,
  SVTK_PARAMETRIC_TRI_SURFACE = 53,
  SVTK_PARAMETRIC_QUAD_SURFACE = 54,
  SVTK_PARAMETRIC_TETRA_REGION = 55,
  SVTK_PARAMETRIC_HEX_REGION = 56,

  // Higher order cells
  SVTK_HIGHER_ORDER_EDGE = 60,
  SVTK_HIGHER_ORDER_TRIANGLE = 61,
  SVTK_HIGHER_ORDER_QUAD = 62,
  SVTK_HIGHER_ORDER_POLYGON = 63,
  SVTK_HIGHER_ORDER_TETRAHEDRON = 64,
  SVTK_HIGHER_ORDER_WEDGE = 65,
  SVTK_HIGHER_ORDER_PYRAMID = 66,
  SVTK_HIGHER_ORDER_HEXAHEDRON = 67,

  // Arbitrary order Lagrange elements (formulated separated from generic higher order cells)
  SVTK_LAGRANGE_CURVE = 68,
  SVTK_LAGRANGE_TRIANGLE = 69,
  SVTK_LAGRANGE_QUADRILATERAL = 70,
  SVTK_LAGRANGE_TETRAHEDRON = 71,
  SVTK_LAGRANGE_HEXAHEDRON = 72,
  SVTK_LAGRANGE_WEDGE = 73,
  SVTK_LAGRANGE_PYRAMID = 74,

  // Arbitrary order Bezier elements (formulated separated from generic higher order cells)
  SVTK_BEZIER_CURVE = 75,
  SVTK_BEZIER_TRIANGLE = 76,
  SVTK_BEZIER_QUADRILATERAL = 77,
  SVTK_BEZIER_TETRAHEDRON = 78,
  SVTK_BEZIER_HEXAHEDRON = 79,
  SVTK_BEZIER_WEDGE = 80,
  SVTK_BEZIER_PYRAMID = 81,

  SVTK_NUMBER_OF_CELL_TYPES
} SVTKCellType;

#endif
// SVTK-HeaderTest-Exclude: svtkCellType.h
