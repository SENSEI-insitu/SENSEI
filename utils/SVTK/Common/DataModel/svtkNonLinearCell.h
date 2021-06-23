/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkNonLinearCell.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkNonLinearCell
 * @brief   abstract superclass for non-linear cells
 *
 * svtkNonLinearCell is an abstract superclass for non-linear cell types.
 * Cells that are a direct subclass of svtkCell or svtkCell3D are linear;
 * cells that are a subclass of svtkNonLinearCell have non-linear interpolation
 * functions. Non-linear cells require special treatment when tessellating
 * or converting to graphics primitives. Note that the linearity of the cell
 * is a function of whether the cell needs tessellation, which does not
 * strictly correlate with interpolation order (e.g., svtkHexahedron has
 * non-linear interpolation functions (a product of three linear functions
 * in r-s-t) even thought svtkHexahedron is considered linear.)
 */

#ifndef svtkNonLinearCell_h
#define svtkNonLinearCell_h

#include "svtkCell.h"
#include "svtkCommonDataModelModule.h" // For export macro

class SVTKCOMMONDATAMODEL_EXPORT svtkNonLinearCell : public svtkCell
{
public:
  svtkTypeMacro(svtkNonLinearCell, svtkCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Non-linear cells require special treatment (tessellation) when
   * converting to graphics primitives (during mapping). The svtkCell
   * API IsLinear() is modified to indicate this requirement.
   */
  int IsLinear() override { return 0; }

protected:
  svtkNonLinearCell();
  ~svtkNonLinearCell() override {}

private:
  svtkNonLinearCell(const svtkNonLinearCell&) = delete;
  void operator=(const svtkNonLinearCell&) = delete;
};

#endif
