/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkConvexPointSet.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkConvexPointSet
 * @brief   a 3D cell defined by a set of convex points
 *
 * svtkConvexPointSet is a concrete implementation that represents a 3D cell
 * defined by a convex set of points. An example of such a cell is an octant
 * (from an octree). svtkConvexPointSet uses the ordered triangulations
 * approach (svtkOrderedTriangulator) to create triangulations guaranteed to
 * be compatible across shared faces. This allows a general approach to
 * processing complex, convex cell types.
 *
 * @sa
 * svtkHexahedron svtkPyramid svtkTetra svtkVoxel svtkWedge
 */

#ifndef svtkConvexPointSet_h
#define svtkConvexPointSet_h

#include "svtkCell3D.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkUnstructuredGrid;
class svtkCellArray;
class svtkTriangle;
class svtkTetra;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkConvexPointSet : public svtkCell3D
{
public:
  static svtkConvexPointSet* New();
  svtkTypeMacro(svtkConvexPointSet, svtkCell3D);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * See svtkCell3D API for description of this method.
   */
  virtual int HasFixedTopology() { return 0; }

  //@{
  /**
   * See svtkCell3D API for description of these methods.
   * @warning These method are unimplemented in svtkConvexPointSet
   */
  void GetEdgePoints(svtkIdType svtkNotUsed(edgeId), const svtkIdType*& svtkNotUsed(pts)) override
  {
    svtkWarningMacro(<< "svtkConvexPointSet::GetEdgePoints Not Implemented");
  }
  // @deprecated Replaced by GetEdgePoints(svtkIdType, const svtkIdType*&) as of SVTK 9.0
  SVTK_LEGACY(void GetEdgePoints(int svtkNotUsed(edgeId), int*& svtkNotUsed(pts)) override {
    svtkWarningMacro(<< "svtkConvexPointSet::GetEdgePoints Not Implemented. "
                    << "Also note that this signature is deprecated. "
                    << "Please use GetEdgePoints(svtkIdType, const svtkIdType*& instead");
  });
  svtkIdType GetFacePoints(svtkIdType svtkNotUsed(faceId), const svtkIdType*& svtkNotUsed(pts)) override
  {
    svtkWarningMacro(<< "svtkConvexPointSet::GetFacePoints Not Implemented");
    return 0;
  }
  // @deprecated Replaced by GetFacePoints(svtkIdType, const svtkIdType*&) as of SVTK 9.0
  SVTK_LEGACY(void GetFacePoints(int svtkNotUsed(faceId), int*& svtkNotUsed(pts)) override {
    svtkWarningMacro(<< "svtkConvexPointSet::GetFacePoints Not Implemented. "
                    << "Also note that this signature is deprecated. "
                    << "Please use GetFacePoints(svtkIdType, const svtkIdType*& instead");
  });
  void GetEdgeToAdjacentFaces(
    svtkIdType svtkNotUsed(edgeId), const svtkIdType*& svtkNotUsed(pts)) override
  {
    svtkWarningMacro(<< "svtkConvexPointSet::GetEdgeToAdjacentFaces Not Implemented");
  }
  svtkIdType GetFaceToAdjacentFaces(
    svtkIdType svtkNotUsed(faceId), const svtkIdType*& svtkNotUsed(faceIds)) override
  {
    svtkWarningMacro(<< "svtkConvexPointSet::GetFaceToAdjacentFaces Not Implemented");
    return 0;
  }
  svtkIdType GetPointToIncidentEdges(
    svtkIdType svtkNotUsed(pointId), const svtkIdType*& svtkNotUsed(edgeIds)) override
  {
    svtkWarningMacro(<< "svtkConvexPointSet::GetPointToIncidentEdges Not Implemented");
    return 0;
  }
  svtkIdType GetPointToIncidentFaces(
    svtkIdType svtkNotUsed(pointId), const svtkIdType*& svtkNotUsed(faceIds)) override
  {
    svtkWarningMacro(<< "svtkConvexPointSet::GetPointToIncidentFaces Not Implemented");
    return 0;
  }
  svtkIdType GetPointToOneRingPoints(
    svtkIdType svtkNotUsed(pointId), const svtkIdType*& svtkNotUsed(pts)) override
  {
    svtkWarningMacro(<< "svtkConvexPointSet::GetPointToOneRingPoints Not Implemented");
    return 0;
  }
  bool GetCentroid(double svtkNotUsed(centroid)[3]) const override
  {
    svtkWarningMacro(<< "svtkConvexPointSet::GetCentroid Not Implemented");
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
  int GetCellType() override { return SVTK_CONVEX_POINT_SET; }

  /**
   * This cell requires that it be initialized prior to access.
   */
  int RequiresInitialization() override { return 1; }
  void Initialize() override;

  //@{
  /**
   * A convex point set has no explicit cell edge or faces; however
   * implicitly (after triangulation) it does. Currently the method
   * GetNumberOfEdges() always returns 0 while the GetNumberOfFaces() returns
   * the number of boundary triangles of the triangulation of the convex
   * point set. The method GetNumberOfFaces() triggers a triangulation of the
   * convex point set; repeated calls to GetFace() then return the boundary
   * faces. (Note: GetNumberOfEdges() currently returns 0 because it is a
   * rarely used method and hard to implement. It can be changed in the future.
   */
  int GetNumberOfEdges() override { return 0; }
  svtkCell* GetEdge(int) override { return nullptr; }
  int GetNumberOfFaces() override;
  svtkCell* GetFace(int faceId) override;
  //@}

  /**
   * Satisfy the svtkCell API. This method contours by triangulating the
   * cell and then contouring the resulting tetrahedra.
   */
  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;

  /**
   * Satisfy the svtkCell API. This method contours by triangulating the
   * cell and then adding clip-edge intersection points into the
   * triangulation; extracting the clipped region.
   */
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* connectivity, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;

  /**
   * Satisfy the svtkCell API. This method determines the subId, pcoords,
   * and weights by triangulating the convex point set, and then
   * determining which tetrahedron the point lies in.
   */
  int EvaluatePosition(const double x[3], double closestPoint[3], int& subId, double pcoords[3],
    double& dist2, double weights[]) override;

  /**
   * The inverse of EvaluatePosition.
   */
  void EvaluateLocation(int& subId, const double pcoords[3], double x[3], double* weights) override;

  /**
   * Triangulates the cells and then intersects them to determine the
   * intersection point.
   */
  int IntersectWithLine(const double p1[3], const double p2[3], double tol, double& t, double x[3],
    double pcoords[3], int& subId) override;

  /**
   * Triangulate using methods of svtkOrderedTriangulator.
   */
  int Triangulate(int index, svtkIdList* ptIds, svtkPoints* pts) override;

  /**
   * Computes derivatives by triangulating and from subId and pcoords,
   * evaluating derivatives on the resulting tetrahedron.
   */
  void Derivatives(
    int subId, const double pcoords[3], const double* values, int dim, double* derivs) override;

  /**
   * Returns the set of points forming a face of the triangulation of these
   * points that are on the boundary of the cell that are closest
   * parametrically to the point specified.
   */
  int CellBoundary(int subId, const double pcoords[3], svtkIdList* pts) override;

  /**
   * Return the center of the cell in parametric coordinates.
   */
  int GetParametricCenter(double pcoords[3]) override;

  /**
   * A convex point set is triangulated prior to any operations on it so
   * it is not a primary cell, it is a composite cell.
   */
  int IsPrimaryCell() override { return 0; }

  //@{
  /**
   * Compute the interpolation functions/derivatives
   * (aka shape functions/derivatives)
   */
  void InterpolateFunctions(const double pcoords[3], double* sf) override;
  void InterpolateDerivs(const double pcoords[3], double* derivs) override;
  //@}

protected:
  svtkConvexPointSet();
  ~svtkConvexPointSet() override;

  svtkTetra* Tetra;
  svtkIdList* TetraIds;
  svtkPoints* TetraPoints;
  svtkDoubleArray* TetraScalars;

  svtkCellArray* BoundaryTris;
  svtkTriangle* Triangle;
  svtkDoubleArray* ParametricCoords;

private:
  svtkConvexPointSet(const svtkConvexPointSet&) = delete;
  void operator=(const svtkConvexPointSet&) = delete;
};

//----------------------------------------------------------------------------
inline int svtkConvexPointSet::GetParametricCenter(double pcoords[3])
{
  pcoords[0] = pcoords[1] = pcoords[2] = 0.5;
  return 0;
}

#endif
