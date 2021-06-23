/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkWedge.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkWedge
 * @brief   a 3D cell that represents a linear wedge
 *
 * svtkWedge is a concrete implementation of svtkCell to represent a linear 3D
 * wedge. A wedge consists of two triangular and three quadrilateral faces
 * and is defined by the six points (0-5). svtkWedge uses the standard
 * isoparametric shape functions for a linear wedge. The wedge is defined
 * by the six points (0-5) where (0,1,2) is the base of the wedge which,
 * using the right hand rule, forms a triangle whose normal points outward
 * (away from the triangular face (3,4,5)).
 *
 * @sa
 * svtkConvexPointSet svtkHexahedron svtkPyramid svtkTetra svtkVoxel
 */

#ifndef svtkWedge_h
#define svtkWedge_h

#include "svtkCell3D.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkLine;
class svtkTriangle;
class svtkQuad;
class svtkUnstructuredGrid;
class svtkIncrementalPointLocator;

class SVTKCOMMONDATAMODEL_EXPORT svtkWedge : public svtkCell3D
{
public:
  static svtkWedge* New();
  svtkTypeMacro(svtkWedge, svtkCell3D);
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
  static constexpr svtkIdType NumberOfPoints = 6;

  /**
   * static contexpr handle on the number of faces.
   */
  static constexpr svtkIdType NumberOfEdges = 9;

  /**
   * static contexpr handle on the number of edges.
   */
  static constexpr svtkIdType NumberOfFaces = 5;

  /**
   * static contexpr handle on the maximum face size. It can also be used
   * to know the number of faces adjacent to one face.
   */
  static constexpr svtkIdType MaximumFaceSize = 4;

  /**
   * static constexpr handle on the maximum valence of this cell.
   * The valence of a vertex is the number of incident edges (or equivalently faces)
   * to this vertex. It is also equal to the size of a one ring neighborhood of a vertex.
   */
  static constexpr svtkIdType MaximumValence = 3;

  //@{
  /**
   * See the svtkCell API for descriptions of these methods.
   */
  int GetCellType() override { return SVTK_WEDGE; }
  int GetCellDimension() override { return 3; }
  int GetNumberOfEdges() override { return static_cast<int>(NumberOfEdges); }
  int GetNumberOfFaces() override { return static_cast<int>(NumberOfFaces); }
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int faceId) override;
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;
  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;
  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;
  double* GetParametricCoords() override;
  //@}

  /**
   * Return the case table for table-based isocontouring (aka marching cubes
   * style implementations). A linear 3D cell with N vertices will have 2**N
   * cases. The returned case array lists three edges in order to produce one
   * output triangle which may be repeated to generate multiple triangles. The
   * list of cases terminates with a -1 entry.
   */
  static int* GetTriangleCases(int caseId);

  /**
   * Return the center of the wedge in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  static void InterpolationFunctions(const double pcoords[3], double weights[6]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[18]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[6]) override
  {
    svtkWedge::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[18]) override
  {
    svtkWedge::InterpolationDerivs(pcoords, derivs);
  }
  int JacobianInverse(const double pcoords[3], double** inverse, double derivs[18]);
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
  static const svtkIdType* GetEdgeArray(svtkIdType edgeId) SVTK_SIZEHINT(2);
  static const svtkIdType* GetFaceArray(svtkIdType faceId) SVTK_SIZEHINT(MaximumFaceSize + 1);
  //@}

  /**
   * Static method version of GetEdgeToAdjacentFaces.
   */
  static const svtkIdType* GetEdgeToAdjacentFacesArray(svtkIdType edgeId) SVTK_SIZEHINT(2);

  /**
   * Static method version of GetFaceToAdjacentFaces.
   */
  static const svtkIdType* GetFaceToAdjacentFacesArray(svtkIdType faceId) SVTK_SIZEHINT(4);

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

protected:
  svtkWedge();
  ~svtkWedge() override;

  svtkLine* Line;
  svtkTriangle* Triangle;
  svtkQuad* Quad;

private:
  svtkWedge(const svtkWedge&) = delete;
  void operator=(const svtkWedge&) = delete;
};

inline int svtkWedge::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = 0.333333;
  pcoords[2] = 0.5;
  return 0;
}

#endif
