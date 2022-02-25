/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTetra.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkTetra
 * @brief   a 3D cell that represents a tetrahedron
 *
 * svtkTetra is a concrete implementation of svtkCell to represent a 3D
 * tetrahedron. svtkTetra uses the standard isoparametric shape functions
 * for a linear tetrahedron. The tetrahedron is defined by the four points
 * (0-3); where (0,1,2) is the base of the tetrahedron which, using the
 * right hand rule, forms a triangle whose normal points in the direction
 * of the fourth point.
 *
 * @sa
 * svtkConvexPointSet svtkHexahedron svtkPyramid svtkVoxel svtkWedge
 */

#ifndef svtkTetra_h
#define svtkTetra_h

#include "svtkCell3D.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkLine;
class svtkTriangle;
class svtkUnstructuredGrid;
class svtkIncrementalPointLocator;

class SVTKCOMMONDATAMODEL_EXPORT svtkTetra : public svtkCell3D
{
public:
  static svtkTetra* New();
  svtkTypeMacro(svtkTetra, svtkCell3D);
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
  static constexpr svtkIdType NumberOfPoints = 4;

  /**
   * static contexpr handle on the number of faces.
   */
  static constexpr svtkIdType NumberOfEdges = 6;

  /**
   * static contexpr handle on the number of edges.
   */
  static constexpr svtkIdType NumberOfFaces = 4;

  /**
   * static contexpr handle on the maximum face size. It can also be used
   * to know the number of faces adjacent to one face.
   */
  static constexpr svtkIdType MaximumFaceSize = 3;

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
  int GetCellType() override { return SVTK_TETRA; }
  int GetNumberOfEdges() override { return 6; }
  int GetNumberOfFaces() override { return 4; }
  svtkCell* GetEdge(int edgeId) override;
  svtkCell* GetFace(int faceId) override;
  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* connectivity, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;
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
   * Returns the set of points that are on the boundary of the tetrahedron that
   * are closest parametrically to the point specified. This may include faces,
   * edges, or vertices.
   */
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;

  /**
   * Return the center of the tetrahedron in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  /**
   * Return the distance of the parametric coordinate provided to the
   * cell. If inside the cell, a distance of zero is returned.
   */
  double GetParametricDistance(const double pcoords[3]) override;

  /**
   * Compute the center of the tetrahedron,
   */
  static void TetraCenter(double p1[3], double p2[3], double p3[3], double p4[3], double center[3]);

  /**
   * Compute the circumcenter (center[3]) and radius squared (method
   * return value) of a tetrahedron defined by the four points x1, x2,
   * x3, and x4.
   */
  static double Circumsphere(
    double p1[3], double p2[3], double p3[3], double p4[3], double center[3]);

  /**
   * Compute the center (center[3]) and radius (method return value) of
   * a sphere that just fits inside the faces of a tetrahedron defined
   * by the four points x1, x2, x3, and x4.
   */
  static double Insphere(double p1[3], double p2[3], double p3[3], double p4[3], double center[3]);

  /**
   * Given a 3D point x[3], determine the barycentric coordinates of the point.
   * Barycentric coordinates are a natural coordinate system for simplices that
   * express a position as a linear combination of the vertices. For a
   * tetrahedron, there are four barycentric coordinates (because there are
   * four vertices), and the sum of the coordinates must equal 1. If a
   * point x is inside a simplex, then all four coordinates will be strictly
   * positive.  If three coordinates are zero (so the fourth =1), then the
   * point x is on a vertex. If two coordinates are zero, the point x is on an
   * edge (and so on). In this method, you must specify the vertex coordinates
   * x1->x4. Returns 0 if tetrahedron is degenerate.
   */
  static int BarycentricCoords(
    double x[3], double x1[3], double x2[3], double x3[3], double x4[3], double bcoords[4]);

  /**
   * Compute the volume of a tetrahedron defined by the four points
   * p1, p2, p3, and p4.
   */
  static double ComputeVolume(double p1[3], double p2[3], double p3[3], double p4[3]);

  /**
   * Given parametric coordinates compute inverse Jacobian transformation
   * matrix. Returns 9 elements of 3x3 inverse Jacobian plus interpolation
   * function derivatives. Returns 0 if no inverse exists.
   */
  int JacobianInverse(double** inverse, double derivs[12]);

  static void InterpolationFunctions(const double pcoords[3], double weights[4]);
  static void InterpolationDerivs(const double pcoords[3], double derivs[12]);
  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double weights[4]) override
  {
    svtkTetra::InterpolationFunctions(pcoords, weights);
  }
  void InterpolateDerivs(const double pcoords[3], double derivs[12]) override
  {
    svtkTetra::InterpolationDerivs(pcoords, derivs);
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
  static const svtkIdType* GetEdgeArray(svtkIdType edgeId) SVTK_SIZEHINT(2);
  static const svtkIdType* GetFaceArray(svtkIdType faceId) SVTK_SIZEHINT(3);
  //@}

  /**
   * Static method version of GetEdgeToAdjacentFaces.
   */
  static const svtkIdType* GetEdgeToAdjacentFacesArray(svtkIdType edgeId) SVTK_SIZEHINT(2);

  /**
   * Static method version of GetFaceToAdjacentFaces.
   */
  static const svtkIdType* GetFaceToAdjacentFacesArray(svtkIdType faceId) SVTK_SIZEHINT(3);

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
  svtkTetra();
  ~svtkTetra() override;

  svtkLine* Line;
  svtkTriangle* Triangle;

private:
  svtkTetra(const svtkTetra&) = delete;
  void operator=(const svtkTetra&) = delete;
};

inline int svtkTetra::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = pcoords[2] = 0.25;
  return 0;
}

#endif
