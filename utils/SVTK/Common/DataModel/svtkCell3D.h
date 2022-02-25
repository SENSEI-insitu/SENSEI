/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCell3D.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCell3D
 * @brief   abstract class to specify 3D cell interface
 *
 * svtkCell3D is an abstract class that extends the interfaces for 3D data
 * cells, and implements methods needed to satisfy the svtkCell API. The
 * 3D cells include hexehedra, tetrahedra, wedge, pyramid, and voxel.
 *
 * @sa
 * svtkTetra svtkHexahedron svtkVoxel svtkWedge svtkPyramid
 */

#ifndef svtkCell3D_h
#define svtkCell3D_h

#include "svtkCell.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkOrderedTriangulator;
class svtkTetra;
class svtkCellArray;
class svtkDoubleArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkCell3D : public svtkCell
{
public:
  svtkTypeMacro(svtkCell3D, svtkCell);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Get the pair of vertices that define an edge. The method returns the
   * number of vertices, along with an array of vertices. Note that the
   * vertices are 0-offset; that is, they refer to the ids of the cell, not
   * the point ids of the mesh that the cell belongs to. The edgeId must
   * range between 0<=edgeId<this->GetNumberOfEdges().
   */
  virtual void GetEdgePoints(svtkIdType edgeId, const svtkIdType*& pts) = 0;
  // @deprecated Replaced by GetEdgePoints(svtkIdType, const svtkIdType*&) as of SVTK 9.0
  SVTK_LEGACY(virtual void GetEdgePoints(int edgeId, int*& pts) = 0;);

  /**
   * Get the list of vertices that define a face. The list is terminated
   * with a negative number. Note that the vertices are 0-offset; that is,
   * they refer to the ids of the cell, not the point ids of the mesh that
   * the cell belongs to. The faceId must range between
   * 0<=faceId<this->GetNumberOfFaces().
   *
   * @return The number of points in face faceId
   */
  virtual svtkIdType GetFacePoints(svtkIdType faceId, const svtkIdType*& pts) = 0;
  // @deprecated Replaced by GetFacePoints(svtkIdType, const svtkIdType*&) as of SVTK 9.0
  SVTK_LEGACY(virtual void GetFacePoints(int faceId, int*& pts) = 0;);

  /**
   * Get the ids of the two adjacent faces to edge of id edgeId.
   * The output face ids are sorted from id of lowest rank to highest.
   * Note that the faces are 0-offset; that is, they refer to the ids of the cells,
   * not the face ids of the mesh that the cell belongs to. The edgeId must range
   * between 0<=edgeId<this->GetNumberOfEdges().
   */
  virtual void GetEdgeToAdjacentFaces(svtkIdType edgeId, const svtkIdType*& faceIds) = 0;

  /**
   * Get the ids of the adjacent faces to face of id faceId. The order of
   * faces is consistent. They are always ordered in counter clockwise w.r.t.
   * normal orientation.
   * The first id faces[0] corresponds to the face sharing point of id pts[0]
   * where pts is obtained from this->GetFacePoints(faceId, pts), being
   * the "most counter clockwise" oriented w.r.t. face faceId.
   * Note that the faces are 0-offset; that is, they
   * refer to the ids of the cell, not the face ids of the mesh that the cell belongs to.
   * The faceId must be between 0<=faceId<this->GetNumberOfFaces();
   *
   * @warning If the svtkCell3D is "inside out", i.e. normals point inside the cell, the order is
   * inverted.
   * @return The number of adjacent faces to faceId.
   */
  virtual svtkIdType GetFaceToAdjacentFaces(svtkIdType faceId, const svtkIdType*& faceIds) = 0;

  /**
   * Get the ids of the incident edges to point of id pointId. Edges are
   * sorted in counter clockwise order w.r.t. bisectrix pointing outside the cell
   * at point of id pointId.
   * The first edge corresponds to the edge containing point of id pts[0], where
   * pts is obtained from this->GetPointToOnRingVertices(pointId, pts).
   * Note that the edges are 0-offset; that is, they refer to the ids of the cell,
   * not the edge ids of the mesh that the cell belongs to.
   * The edgeId must be between 0<=edgeId<this->GetNumberOfEdges();
   *
   * @warning If the svtkCell3D is "inside out", i.e. normals point inside the cell, the order is
   * inverted.
   * @return The valence of point pointId.
   */
  virtual svtkIdType GetPointToIncidentEdges(svtkIdType pointId, const svtkIdType*& edgeIds) = 0;

  /**
   * Get the ids of the incident faces point of id pointId. Faces are
   * sorted in counter clockwise order w.r.t. bisectrix pointing outside the cell
   * at point of id pointId.
   * The first face corresponds to the face containing edge of id edges[0],
   * where edges is obtained from this->GetPointToIncidentEdges(pointId, edges),
   * such that face faces[0] is the "most counterclockwise" face incident to
   * point pointId containing edges[0].
   * Note that the faces are 0-offset; that is, they refer to the ids of the cell,
   * not the face ids of the mesh that the cell belongs to.
   * The pointId must be between 0<=pointId<this->GetNumberOfPoints().
   *
   * @warning If the svtkCell3D is "inside out", i.e. normals point inside the cell, the order is
   * inverted.
   * @return The valence of point pointId.
   */
  virtual svtkIdType GetPointToIncidentFaces(svtkIdType pointId, const svtkIdType*& faceIds) = 0;

  /**
   * Get the ids of a one-ring surrounding point of id pointId. Points are
   * sorted in counter clockwise order w.r.t. bisectrix pointing outside the cell
   * at point of id pointId.
   * The first point corresponds to the point contained in edges[0], where
   * edges is obtained from this->GetPointToIncidentEdges(pointId, edges).
   * Note that the points are 0-offset; that is, they refer to the ids of the cell,
   * not the point ids of the mesh that the cell belongs to.
   * The pointId must be between 0<pointId<this->GetNumberOfPoints().
   * @return The valence of point pointId.
   */
  virtual svtkIdType GetPointToOneRingPoints(svtkIdType pointId, const svtkIdType*& pts) = 0;

  /**
   * Returns true if the normals of the svtkCell3D point inside the cell.
   *
   * @warning This flag is not precomputed. It is advised for the return result of
   * this method to be stored in a local boolean by the user if needed multiple times.
   */
  virtual bool IsInsideOut();

  /**
   * Computes the centroid of the cell.
   */
  virtual bool GetCentroid(double centroid[3]) const = 0;

  void Contour(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* verts, svtkCellArray* lines, svtkCellArray* polys, svtkPointData* inPd,
    svtkPointData* outPd, svtkCellData* inCd, svtkIdType cellId, svtkCellData* outCd) override;

  /**
   * Cut (or clip) the cell based on the input cellScalars and the specified
   * value. The output of the clip operation will be one or more cells of the
   * same topological dimension as the original cell.  The flag insideOut
   * controls what part of the cell is considered inside - normally cell
   * points whose scalar value is greater than "value" are considered
   * inside. If insideOut is on, this is reversed. Also, if the output cell
   * data is non-nullptr, the cell data from the clipped cell is passed to the
   * generated contouring primitives. (Note: the CopyAllocate() method must
   * be invoked on both the output cell and point data. The cellId refers to
   * the cell from which the cell data is copied.)  (Satisfies svtkCell API.)
   */
  void Clip(double value, svtkDataArray* cellScalars, svtkIncrementalPointLocator* locator,
    svtkCellArray* connectivity, svtkPointData* inPd, svtkPointData* outPd, svtkCellData* inCd,
    svtkIdType cellId, svtkCellData* outCd, int insideOut) override;

  /**
   * The topological dimension of the cell. (Satisfies svtkCell API.)
   */
  int GetCellDimension() override { return 3; }

  //@{
  /**
   * Set the tolerance for merging clip intersection points that are near
   * the vertices of cells. This tolerance is used to prevent the generation
   * of degenerate tetrahedra during clipping.
   */
  svtkSetClampMacro(MergeTolerance, double, 0.0001, 0.25);
  svtkGetMacro(MergeTolerance, double);
  //@}

protected:
  svtkCell3D();
  ~svtkCell3D() override;

  svtkOrderedTriangulator* Triangulator;
  double MergeTolerance;

  // used to support clipping
  svtkTetra* ClipTetra;
  svtkDoubleArray* ClipScalars;

private:
  svtkCell3D(const svtkCell3D&) = delete;
  void operator=(const svtkCell3D&) = delete;
};

#endif
