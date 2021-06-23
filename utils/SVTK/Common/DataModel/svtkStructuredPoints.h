/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStructuredPoints.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkStructuredPoints
 * @brief   A subclass of ImageData.
 *
 * StructuredPoints is a subclass of ImageData that requires the data extent
 * to exactly match the update extent. Normall image data allows that the
 * data extent may be larger than the update extent.
 * StructuredPoints also defines the origin differently that svtkImageData.
 * For structured points the origin is the location of first point.
 * Whereas images define the origin as the location of point 0, 0, 0.
 * Image Origin is stored in ivar, and structured points
 * have special methods for setting/getting the origin/extents.
 */

#ifndef svtkStructuredPoints_h
#define svtkStructuredPoints_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImageData.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkStructuredPoints : public svtkImageData
{
public:
  static svtkStructuredPoints* New();
  svtkTypeMacro(svtkStructuredPoints, svtkImageData);

  /**
   * To simplify filter superclasses,
   */
  int GetDataObjectType() override { return SVTK_STRUCTURED_POINTS; }

protected:
  svtkStructuredPoints();
  ~svtkStructuredPoints() override {}

private:
  svtkStructuredPoints(const svtkStructuredPoints&) = delete;
  void operator=(const svtkStructuredPoints&) = delete;
};

#endif

// SVTK-HeaderTest-Exclude: svtkStructuredPoints.h
