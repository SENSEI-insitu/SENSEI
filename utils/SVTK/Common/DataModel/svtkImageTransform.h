/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImageTransform.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkImageTransform
 * @brief   helper class to transform output of non-axis-aligned images
 *
 * svtkImageTransform is a helper class to transform the output of
 * image filters (i.e., filter that input svtkImageData) by applying the
 * Index to Physical transformation frmo the input image, which can
 * include origin, spacing, direction. The transformation process is
 * threaded with svtkSMPTools for performance.
 *
 * Typically in application the single method TransformPointSet() is
 * invoked to transform the output of an image algorithm (assuming
 * that the image's direction/orientation matrix is
 * non-identity). Note that svtkPointSets encompass svtkPolyData as well
 * as svtkUnstructuredGrids. In the future other output types may be
 * added. Note that specific methods for transforming points, normals,
 * and vectors is also provided by this class in case additional
 * output data arrays need to be transformed (since
 * TransformPointSet() only processes data arrays labeled as points,
 * normals, and vectors).
 *
 * @warning
 * This class has been threaded with svtkSMPTools. Using TBB or other
 * non-sequential type (set in the CMake variable
 * SVTK_SMP_IMPLEMENTATION_TYPE) may improve performance significantly.
 *
 */

#ifndef svtkImageTransform_h
#define svtkImageTransform_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkDataArray;
class svtkImageData;
class svtkMatrix3x3;
class svtkMatrix4x4;
class svtkPointSet;

class SVTKCOMMONDATAMODEL_EXPORT svtkImageTransform : public svtkObject
{
public:
  //@{
  /**
   * Standard methods for construction, type information, printing.
   */
  static svtkImageTransform* New();
  svtkTypeMacro(svtkImageTransform, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Given a svtkImageData (and hence its associated orientation
   * matrix), and an instance of svtkPointSet, transform its points, as
   * well as any normals and vectors, associated with the
   * svtkPointSet. This is a convenience function, internally it calls
   * TranslatePoints(), TransformPoints(), TransformNormals(), and/or
   * TransformVectors() as appropriate. Note that both the normals and
   * vectors associated with the point and cell data are transformed.
   */
  static void TransformPointSet(svtkImageData* im, svtkPointSet* ps);

  /**
   * Given x-y-z points represented by a svtkDataArray,
   * translate the points using the image origin. This
   * method is useful if there is no orientation or
   * spacing to apply.
   */
  static void TranslatePoints(double* t, svtkDataArray* da);

  /**
   * Given x-y-z points represented by a svtkDataArray,
   * transform the points using the matrix provided.
   */
  static void TransformPoints(svtkMatrix4x4* m4, svtkDataArray* da);

  /**
   * Given three-component normals represented by a svtkDataArray,
   * transform the normals using the matrix provided.
   */
  static void TransformNormals(svtkMatrix3x3* m3, double spacing[3], svtkDataArray* da);

  /**
   * Given three-component vectors represented by a svtkDataArray,
   * transform the vectors using the matrix provided.
   */
  static void TransformVectors(svtkMatrix3x3* m3, double spacing[3], svtkDataArray* da);

protected:
  svtkImageTransform() {}
  ~svtkImageTransform() override {}

private:
  svtkImageTransform(const svtkImageTransform&) = delete;
  void operator=(const svtkImageTransform&) = delete;
};

#endif
