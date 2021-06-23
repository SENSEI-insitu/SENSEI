/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPolyhedron.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPolyhedron
 * @brief   a 3D cell defined by a set of polygonal faces
 *
 * svtkPolyhedron is a concrete implementation that represents a 3D cell
 * defined by a set of polygonal faces. The polyhedron should be watertight,
 * non-self-intersecting and manifold (each edge is used twice).
 *
 * Interpolation functions and weights are defined / computed using the
 * method of Mean Value Coordinates (MVC). See the SVTK class
 * svtkMeanValueCoordinatesInterpolator for more information.
 *
 * The class does not require the polyhedron to be convex. However, the
 * polygonal faces must be planar. Non-planar polygonal faces will
 * definitely cause problems, especially in severely warped situations.
 *
 * @sa
 * svtkCell3D svtkConvecPointSet svtkMeanValueCoordinatesInterpolator
 */

#ifndef svtkPolyhedron_h
#define svtkPolyhedron_h

#include "svtkCell3D.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkIdTypeArray;
class svtkCellArray;
class svtkTriangle;
class svtkQuad;
class svtkTetra;
class svtkPolygon;
class svtkLine;
class svtkPointIdMap;
class svtkIdToIdVectorMapType;
class svtkIdToIdMapType;
class svtkEdgeTable;
class svtkPolyData;
class svtkCellLocator;
class svtkGenericCell;
class svtkPointLocator;

class SVTKCOMMONDATAMODEL_EXPORT svtkPolyhedron : public svtkCell3D
{
public:
  //@{
  /**
   * Standard new methods.
   */
  static svtkPolyhedron* New();
  svtkTypeMacro(svtkPolyhedron, svtkCell3D);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  //@{
  /**
   * See svtkCell3D API for description of these methods.
   * @warning These method are unimplemented in svtkPolyhedron
   */
  void GetEdgePoints(svtkIdType svtkNotUsed(edgeId), const svtkIdType*& svtkNotUsed(pts)) override
  {
    svtkWarningMacro(<< "svtkPolyhedron::GetEdgePoints Not Implemented");
  }
  // @deprecated Replaced by GetEdgePoints(svtkIdType, const svtkIdType*&) as of SVTK 9.0
  SVTK_LEGACY(void GetEdgePoints(int svtkNotUsed(edgeId), int*& svtkNotUsed(pts)) override {
    svtkWarningMacro(<< "svtkPolyhedron::GetEdgePoints Not Implemented. "
                    << "Also note that this signature is deprecated. "
                    << "Please use GetEdgePoints(svtkIdType, const svtkIdType*& instead");
  });
  svtkIdType GetFacePoints(svtkIdType svtkNotUsed(faceId), const svtkIdType*& svtkNotUsed(pts)) override
  {
    svtkWarningMacro(<< "svtkPolyhedron::GetFacePoints Not Implemented");
    return 0;
  }
  // @deprecated Replaced by GetFacePoints(svtkIdType, const svtkIdType*&) as of SVTK 9.0
  SVTK_LEGACY(void GetFacePoints(int svtkNotUsed(faceId), int*& svtkNotUsed(pts)) override {
    svtkWarningMacro(<< "svtkPolyhedron::GetFacePoints Not Implemented. "
                    << "Also note that this signature is deprecated. "
                    << "Please use GetFacePoints(svtkIdType, const svtkIdType*& instead");
  });
  void GetEdgeToAdjacentFaces(
    svtkIdType svtkNotUsed(edgeId), const svtkIdType*& svtkNotUsed(pts)) override
  {
    svtkWarningMacro(<< "svtkPolyhedron::GetEdgeToAdjacentFaces Not Implemented");
  }
  svtkIdType GetFaceToAdjacentFaces(
    svtkIdType svtkNotUsed(faceId), const svtkIdType*& svtkNotUsed(faceIds)) override
  {
    svtkWarningMacro(<< "svtkPolyhedron::GetFaceToAdjacentFaces Not Implemented");
    return 0;
  }
  svtkIdType GetPointToIncidentEdges(
    svtkIdType svtkNotUsed(pointId), const svtkIdType*& svtkNotUsed(edgeIds)) override
  {
    svtkWarningMacro(<< "svtkPolyhedron::GetPointToIncidentEdges Not Implemented");
    return 0;
  }
  svtkIdType GetPointToIncidentFaces(
    svtkIdType svtkNotUsed(pointId), const svtkIdType*& svtkNotUsed(faceIds)) override
  {
    svtkWarningMacro(<< "svtkPolyhedron::GetPointToIncidentFaces Not Implemented");
    return 0;
  }
  svtkIdType GetPointToOneRingPoints(
    svtkIdType svtkNotUsed(pointId), const svtkIdType*& svtkNotUsed(pts)) override
  {
    svtkWarningMacro(<< "svtkPolyhedron::GetPointToOneRingPoints Not Implemented");
    return 0;
  }
  bool GetCentroid(double svtkNotUsed(centroid)[3]) const override
  {
    svtkWarningMacro(<< "svtkPolyhedron::GetCentroid Not Implemented");
    return 0;
  }
  //@}

  /**
   * See svtkCell3D API for description of this method.
   */
  double* GetParametricCoords() override;

  /**
   * See the svtkCell API for descriptions of these methods.
   */
  int GetCellType() override { return SVTK_POLYHEDRON; }

  /**
   * This cell requires that it be initialized prior to access.
   */
  int RequiresInitialization() override { return 1; }
  void Initialize() override;

  //@{
  /**
   * A polyhedron is represented internally by a set of polygonal faces.
   * These faces can be processed to explicitly determine edges.
   */
  int GetNumberOfEdges() override;
  svtkCell* GetEdge(int) override;
  int GetNumberOfFaces() override;
  svtkCell* GetFace(int faceId) override;
  //@}

  /**
   * Satisfy the svtkCell API. This method contours the input polyhedron and outputs
   * a polygon. When the result polygon is not planar, it will be triangulated.
   * The current implementation assumes water-tight polyhedron cells.
   */
  void Contour(double value, svtkDataArray* scalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;

  /**
   * Satisfy the svtkCell API. This method clips the input polyhedron and outputs
   * a new polyhedron. The face information of the output polyhedron is encoded
   * in the output svtkCellArray using a special format:
   * CellLength [nCellFaces, nFace0Pts, i, j, k, nFace1Pts, i, j, k, ...].
   * Use the static method svtkUnstructuredGrid::DecomposePolyhedronCellArray
   * to convert it into a standard format. Note: the algorithm assumes water-tight
   * polyhedron cells.
   */
  void Clip(double value, svtkDataArray* scalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* connectivity, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;

  /**
   * Satisfy the svtkCell API. The subId is ignored and zero is always
   * returned. The parametric coordinates pcoords are normalized values in
   * the bounding box of the polyhedron. The weights are determined by
   * evaluating the MVC coordinates. The dist is always zero if the point x[3]
   * is inside the polyhedron; otherwise it's the distance to the surface.
   */
  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;

  /**
   * The inverse of EvaluatePosition. Note the weights should be the MVC
   * weights.
   */
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;

  /**
   * Intersect the line (p1,p2) with a given tolerance tol to determine a
   * point of intersection x[3] with parametric coordinate t along the
   * line. The parametric coordinates are returned as well (subId can be
   * ignored). Returns true if the line intersects a face.
   */
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;

  /**
   * Use svtkOrderedTriangulator to tetrahedralize the polyhedron mesh. This
   * method works well for a convex polyhedron but may return wrong result
   * in a concave case.
   * Once triangulation has been performed, the results are saved in ptIds and
   * pts. The ptIds is a svtkIdList with 4xn number of ids (n is the number of
   * result tetrahedrons). The first 4 represent the point ids of the first
   * tetrahedron, the second 4 represents the point ids of the second tetrahedron
   * and so on. The point ids represent global dataset ids.
   * The points of result tetrahedons are stored in pts. Note that there are
   * 4xm output points (m is the number of points in the original polyhedron).
   * A point may be stored multiple times when it is shared by more than one
   * tetrahedrons. The points stored in pts are ordered the same as they are
   * listed in ptIds.
   */
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;

  /**
   * Computes derivatives at the point specified by the parameter coordinate.
   * Current implementation uses all vertices and subId is not used.
   * To accelerate the speed, the future implementation can triangulate and
   * extract the local tetrahedron from subId and pcoords, then evaluate
   * derivatives on the local tetrahedron.
   */
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;

  /**
   * Find the boundary face closest to the point defined by the pcoords[3]
   * and subId of the cell (subId can be ignored).
   */
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;

  /**
   * Return the center of the cell in parametric coordinates. In this cell,
   * the center of the bounding box is returned.
   */
  int GetParametricCenter(double pcoords[3]) override;

  /**
   * A polyhedron is a full-fledged primary cell.
   */
  int IsPrimaryCell() override { return 1; }

  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives). Here we use the MVC calculation
   * process to compute the interpolation functions.
   */
  void InterpolateFunctions(const double x[3], double* sf) override;
  void InterpolateDerivs(const double x[3], double* derivs) override;
  //@}

  //@{
  /**
   * Methods supporting the definition of faces. Note that the GetFaces()
   * returns a list of faces in svtkCellArray form; use the method
   * GetNumberOfFaces() to determine the number of faces in the list.
   * The SetFaces() method is also in svtkCellArray form, except that it
   * begins with a leading count indicating the total number of faces in
   * the list.
   */
  int RequiresExplicitFaceRepresentation() override { return 1; }
  void SetFaces(svtkIdType* faces) override;
  svtkIdType* GetFaces() override;
  //@}

  /**
   * A method particular to svtkPolyhedron. It determines whether a point x[3]
   * is inside the polyhedron or not (returns 1 is the point is inside, 0
   * otherwise). The tolerance is expressed in normalized space; i.e., a
   * fraction of the size of the bounding box.
   */
  int IsInside(const double x[3], double tolerance);

  /**
   * Determine whether or not a polyhedron is convex. This method is adapted
   * from Devillers et al., "Checking the Convexity of Polytopes and the
   * Planarity of Subdivisions", Computational Geometry, Volume 11, Issues
   * 3 - 4, December 1998, Pages 187 - 208.
   */
  bool IsConvex();

  /**
   * Construct polydata if no one exist, then return this->PolyData
   */
  svtkPolyData* GetPolyData();

protected:
  svtkPolyhedron();
  ~svtkPolyhedron() override;

  // Internal classes for supporting operations on this cell
  svtkLine* Line;
  svtkTriangle* Triangle;
  svtkQuad* Quad;
  svtkPolygon* Polygon;
  svtkTetra* Tetra;
  svtkIdTypeArray* GlobalFaces; // these are numbered in global id space
  svtkIdTypeArray* FaceLocations;

  // svtkCell has the data members Points (x,y,z coordinates) and PointIds
  // (global cell ids corresponding to cell canonical numbering (0,1,2,....)).
  // These data members are implicitly organized in canonical space, i.e., where
  // the cell point ids are (0,1,...,npts-1). The PointIdMap maps global point id
  // back to these canonoical point ids.
  svtkPointIdMap* PointIdMap;

  // If edges are needed. Note that the edge numbering is in
  // canonical space.
  int EdgesGenerated;        // true/false
  svtkEdgeTable* EdgeTable;   // keep track of all edges
  svtkIdTypeArray* Edges;     // edge pairs kept in this list, in canonical id space
  svtkIdTypeArray* EdgeFaces; // face pairs that comprise each edge, with the
                             // same ordering as EdgeTable
  int GenerateEdges();       // method populates the edge table and edge array

  // If faces need renumbering into canonical numbering space these members
  // are used. When initiallly loaded, the face numbering uses global dataset
  // ids. Once renumbered, they are converted to canonical space.
  svtkIdTypeArray* Faces; // these are numbered in canonical id space
  int FacesGenerated;
  void GenerateFaces();

  // Bounds management
  int BoundsComputed;
  void ComputeBounds();
  void ComputeParametricCoordinate(const double x[3], double pc[3]);
  void ComputePositionFromParametricCoordinate(const double pc[3], double x[3]);

  // Members for supporting geometric operations
  int PolyDataConstructed;
  svtkPolyData* PolyData;
  svtkCellArray* Polys;
  void ConstructPolyData();
  int LocatorConstructed;
  svtkCellLocator* CellLocator;
  void ConstructLocator();
  svtkIdList* CellIds;
  svtkGenericCell* Cell;

private:
  svtkPolyhedron(const svtkPolyhedron&) = delete;
  void operator=(const svtkPolyhedron&) = delete;
};

//----------------------------------------------------------------------------
inline int svtkPolyhedron::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = pcoords[2] = 0.5;
  return 0;
}

#endif
