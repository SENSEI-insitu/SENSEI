/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPentagonalPrism.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPentagonalPrism
 * @brief   a 3D cell that represents a convex prism with
 * pentagonal base
 *
 * svtkPentagonalPrism is a concrete implementation of svtkCell to represent a
 * linear convex 3D prism with pentagonal base. Such prism is defined by the
 * ten points (0-9), where (0,1,2,3,4) is the base of the prism which, using
 * the right hand rule, forms a pentagon whose normal points is in the direction
 * of the opposite face (5,6,7,8,9).
 *
 * @par Thanks:
 * Thanks to Philippe Guerville who developed this class.
 * Thanks to Charles Pignerol (CEA-DAM, France) who ported this class under
 * SVTK 4. <br>
 * Thanks to Jean Favre (CSCS, Switzerland) who contributed to integrate this
 * class in SVTK. <br>
 * Please address all comments to Jean Favre (jfavre at cscs.ch).
 *
 * @par Thanks:
 * The Interpolation functions and derivatives were changed in June
 * 2015 by Bill Lorensen. These changes follow the formulation in:
 * http://dilbert.engr.ucdavis.edu/~suku/nem/papers/polyelas.pdf
 * NOTE: An additional copy of this paper is located at:
 * http://www.svtk.org/Wiki/File:ApplicationOfPolygonalFiniteElementsInLinearElasticity.pdf
 */

#ifndef svtkPentagonalPrism_h
#define svtkPentagonalPrism_h

#include "svtkCell3D.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkLine;
class svtkPolygon;
class svtkQuad;
class svtkTriangle;

class SVTKCOMMONDATAMODEL_EXPORT svtkPentagonalPrism : public svtkCell3D
{
public:
  static svtkPentagonalPrism* New();
  svtkTypeMacro(svtkPentagonalPrism, svtkCell3D);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * See svtkCell3D API for description of these methods.
   */
  void GetEdgePoints(svtkIdType edgeId, const svtkIdType*& pts) override;
  // @deprecated Replaced by GetEdgePoints(svtkIdType, const svtkIdType*&) as of SVTK 9.0
  SVTK_LEGACY(virtual void GetEdgePoints(int edgeId, int*& pts) override);
  svtkIdType GetFacePoints(svtkIdType faceId, const svtkIdType*& pts) override;
  // @deprecated Replaced by GetFacePoints(svtkIdType, const svtkIdType*&) as of SVTK 9.0
  SVTK_LEGACY(virtual void GetFacePoints(int faceId, int*& pts) override);
  void GetEdgeToAdjacentFaces(svtkIdType edgeId, const svtkIdType*& pts) override;
  svtkIdType GetFaceToAdjacentFaces(svtkIdType faceId, const svtkIdType*& faceIds) override;
  svtkIdType GetPointToIncidentEdges(svtkIdType pointId, const svtkIdType*& edgeIds) override;
  svtkIdType GetPointToIncidentFaces(svtkIdType pointId, const svtkIdType*& faceIds) override;
  svtkIdType GetPointToOneRingPoints(svtkIdType pointId, const svtkIdType*& pts) override;
  bool GetCentroid(double centroid[3]) const override;
  bool IsInsideOut() override;
  //@}

  /**
   * static constexpr handle on the number of points.
   */
  static constexpr svtkIdType NumberOfPoints = 10;

  /**
   * static contexpr handle on the number of faces.
   */
  static constexpr svtkIdType NumberOfEdges = 15;

  /**
   * static contexpr handle on the number of edges.
   */
  static constexpr svtkIdType NumberOfFaces = 7;

  /**
   * static contexpr handle on the maximum face size. It can also be used
   * to know the number of faces adjacent to one face.
   */
  static constexpr svtkIdType MaximumFaceSize = 5;

  /**
   * static constexpr handle on the maximum valence of this cell.
   * The valence of a vertex is the number of incident edges (or equivalently faces)
   * to this vertex. It is also equal to the size of a one ring neighborhood of a vertex.
   */
  static constexpr svtkIdType MaximumValence = 3;

  //@{
  /**
   * See the svtkCell3D API for descriptions of these methods.
   */
  int GetCellType() override { return SVTK_PENTAGONAL_PRISM; }
  int GetCellDimension() override { return 3; }
  int GetNumberOfEdges() override { return 15; }
  int GetNumberOfFaces() override { return 7; }
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int faceId) override;
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;
  //@}

  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  double* GetParametricCoords() override;

  /**
   * Return the center of the wedge in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  static void InterpolationFunctions(const double pcoords[3], double weights[10]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[30]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[10]) override
  {
    svtkPentagonalPrism::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[30]) override
  {
    svtkPentagonalPrism::InterpolationDerivs(pcoords, derivs);
  }
  //@}

  //@{
  /**
   * Return the ids of the vertices defining edge/face (`edgeId`/`faceId').
   * Ids are related to the cell, not to the dataset.
   *
   * @note The return type changed. It used to be int*, it is now const svtkIdType*.
   * This is so ids are unified between svtkCell and svtkPoints, and so svtkCell ids
   * can be used as inputs in algorithms such as svtkPolygon::ComputeNormal.
   */
  static const svtkIdType* GetEdgeArray(svtkIdType edgeId);
  static const svtkIdType* GetFaceArray(svtkIdType faceId);
  //@}

  /**
   * Static method version of GetEdgeToAdjacentFaces.
   */
  static const svtkIdType* GetEdgeToAdjacentFacesArray(svtkIdType edgeId) SVTK_SIZEHINT(2);

  /**
   * Static method version of GetFaceToAdjacentFaces.
   */
  static const svtkIdType* GetFaceToAdjacentFacesArray(svtkIdType faceId) SVTK_SIZEHINT(5);

  /**
   * Static method version of GetPointToIncidentEdgesArray.
   */
  static const svtkIdType* GetPointToIncidentEdgesArray(svtkIdType pointId) SVTK_SIZEHINT(3);

  /**
   * Static method version of GetPointToIncidentFacesArray.
   */
  static const svtkIdType* GetPointToIncidentFacesArray(svtkIdType pointId) SVTK_SIZEHINT(3);

  /**
   * Static method version of GetPointToOneRingPoints.
   */
  static const svtkIdType* GetPointToOneRingPointsArray(svtkIdType pointId) SVTK_SIZEHINT(3);

  /**
   * Static method version of GetCentroid.
   */
  static bool ComputeCentroid(svtkPoints* points, const svtkIdType* pointIds, double centroid[3]);

  /**
   * Given parametric coordinates compute inverse Jacobian transformation
   * matrix. Returns 9 elements of 3x3 inverse Jacobian plus interpolation
   * function derivatives.
   */
  void JacobianInverse(const double pcoords[3], double** inverse, double derivs[24]);

protected:
  svtkPentagonalPrism();
  ~svtkPentagonalPrism() override;

  svtkLine* Line;
  svtkQuad* Quad;
  svtkPolygon* Polygon;
  svtkTriangle* Triangle;

private:
  svtkPentagonalPrism(const svtkPentagonalPrism&) = delete;
  void operator=(const svtkPentagonalPrism&) = delete;
};

//----------------------------------------------------------------------------
inline int svtkPentagonalPrism::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = 0.5;
  pcoords[2] = 0.5;

  return 0;
}
#endif
