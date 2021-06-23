/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkMeanValueCoordinatesInterpolator.h

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkMeanValueCoordinatesInterpolator
 * @brief   compute interpolation computes
 * for closed triangular mesh
 *
 * svtkMeanValueCoordinatesInterpolator computes interpolation weights for a
 * closed, manifold polyhedron mesh.  Once computed, the interpolation
 * weights can be used to interpolate data anywhere interior or exterior to
 * the mesh. This work implements two MVC algorithms. The first one is for
 * triangular meshes which is documented in the Siggraph 2005 paper by Tao Ju,
 * Scot Schaefer and Joe Warren from Rice University "Mean Value Coordinates
 * for Closed Triangular Meshes". The second one is for general polyhedron
 * mesh which is documented in the Eurographics Symposium on Geometry Processing
 * 2006 paper by Torsten Langer, Alexander Belyaev and Hans-Peter Seidel from
 * MPI Informatik "Spherical Barycentric Coordinates".
 * The filter will automatically choose which algorithm to use based on whether
 * the input mesh is triangulated or not.
 *
 * In SVTK this class was initially created to interpolate data across
 * polyhedral cells. In addition, the class can be used to interpolate
 * data values from a polyhedron mesh, and to smoothly deform a mesh from
 * an associated control mesh.
 *
 * @sa
 * svtkPolyhedralCell
 */

#ifndef svtkMeanValueCoordinatesInterpolator_h
#define svtkMeanValueCoordinatesInterpolator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkPoints;
class svtkIdList;
class svtkCellArray;
class svtkDataArray;

// Special internal class for iterating over data
class svtkMVCTriIterator;
class svtkMVCPolyIterator;

class SVTKCOMMONDATAMODEL_EXPORT svtkMeanValueCoordinatesInterpolator : public svtkObject
{
public:
  //@{
  /**
   * Standard instantiable class methods.
   */
  static svtkMeanValueCoordinatesInterpolator* New();
  svtkTypeMacro(svtkMeanValueCoordinatesInterpolator, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Method to generate interpolation weights for a point x[3] from a list of
   * triangles.  In this version of the method, the triangles are defined by
   * a svtkPoints array plus a svtkIdList, where the svtkIdList is organized
   * such that three ids in order define a triangle.  Note that number of weights
   * must equal the number of points.
   */
  static void ComputeInterpolationWeights(
    const double x[3], svtkPoints* pts, svtkIdList* tris, double* weights);

  /**
   * Method to generate interpolation weights for a point x[3] from a list of
   * polygonal faces.  In this version of the method, the faces are defined by
   * a svtkPoints array plus a svtkCellArray, where the svtkCellArray contains all
   * faces and is of format [nFace0Pts, pid1, pid2, pid3,..., nFace1Pts, pid1,
   * pid2, pid3,...].  Note: the number of weights must equal the number of points.
   */
  static void ComputeInterpolationWeights(
    const double x[3], svtkPoints* pts, svtkCellArray* tris, double* weights);

protected:
  svtkMeanValueCoordinatesInterpolator();
  ~svtkMeanValueCoordinatesInterpolator() override;

  /**
   * Internal method that sets up the processing of triangular meshes.
   */
  static void ComputeInterpolationWeightsForTriangleMesh(
    const double x[3], svtkPoints* pts, svtkMVCTriIterator& iter, double* weights);

  /**
   * Internal method that sets up the processing of general polyhedron meshes.
   */
  static void ComputeInterpolationWeightsForPolygonMesh(
    const double x[3], svtkPoints* pts, svtkMVCPolyIterator& iter, double* weights);

private:
  svtkMeanValueCoordinatesInterpolator(const svtkMeanValueCoordinatesInterpolator&) = delete;
  void operator=(const svtkMeanValueCoordinatesInterpolator&) = delete;
};

#endif
