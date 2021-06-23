/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDistributedGraphHelper.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*
 * Copyright (C) 2008 The Trustees of Indiana University.
 * Use, modification and distribution is subject to the Boost Software
 * License, Version 1.0. (See http://www.boost.org/LICENSE_1_0.txt)
 */

/**
 * @class   svtkDistributedGraphHelper
 * @brief   helper for the svtkGraph class
 * that allows the graph to be distributed across multiple memory spaces.
 *
 *
 * A distributed graph helper can be attached to an empty svtkGraph
 * object to turn the svtkGraph into a distributed graph, whose
 * vertices and edges are distributed across several different
 * processors. svtkDistributedGraphHelper is an abstract class. Use a
 * subclass of svtkDistributedGraphHelper, such as
 * svtkPBGLDistributedGraphHelper, to build distributed graphs.
 *
 * The distributed graph helper provides facilities used by svtkGraph
 * to communicate with other processors that store other parts of the
 * same distributed graph. The only user-level functionality provided
 * by svtkDistributedGraphHelper involves this communication among
 * processors and the ability to map between "distributed" vertex and
 * edge IDs and their component parts (processor and local index). For
 * example, the Synchronize() method provides a barrier that allows
 * all processors to catch up to the same point in the code before any
 * processor can leave that Synchronize() call. For example, one would
 * call Synchronize() after adding many edges to a distributed graph,
 * so that all processors can handle the addition of inter-processor
 * edges and continue, after the Synchronize() call, with a consistent
 * view of the distributed graph. For more information about
 * manipulating (distributed) graphs, see the svtkGraph documentation.
 *
 * @sa
 * svtkGraph
 */

#ifndef svtkDistributedGraphHelper_h
#define svtkDistributedGraphHelper_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkDistributedGraphHelperInternals;
struct svtkEdgeType;
class svtkGraph;
class svtkVariant;
class svtkVariantArray;
class svtkInformationIntegerKey;

// .NAME svtkVertexPedigreeIdDistributionFunction - The type of a
// function used to determine how to distribute vertex pedigree IDs
// across processors in a svtkGraph. The pedigree ID distribution
// function takes the pedigree ID of the vertex and a user-supplied
// void pointer and returns a hash value V. A vertex with that
// pedigree ID will reside on processor V % P, where P is the number
// of processors. This type is used in conjunction with the
// svtkDistributedGraphHelper class.
typedef svtkIdType (*svtkVertexPedigreeIdDistribution)(const svtkVariant& pedigreeId, void* userData);

class SVTKCOMMONDATAMODEL_EXPORT svtkDistributedGraphHelper : public svtkObject
{
public:
  svtkTypeMacro(svtkDistributedGraphHelper, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Returns owner of vertex v, by extracting top ceil(log2 P) bits of v.
   */
  svtkIdType GetVertexOwner(svtkIdType v) const;

  /**
   * Returns local index of vertex v, by masking off top ceil(log2 P) bits of v.
   */
  svtkIdType GetVertexIndex(svtkIdType v) const;

  /**
   * Returns owner of edge with ID e_id, by extracting top ceil(log2 P) bits of e_id.
   */
  svtkIdType GetEdgeOwner(svtkIdType e_id) const;

  /**
   * Returns local index of edge with ID e_id, by masking off top ceil(log2 P)
   * bits of e_id.
   */
  svtkIdType GetEdgeIndex(svtkIdType e_id) const;

  /**
   * Builds a distributed ID consisting of the given owner and the local ID.
   */
  svtkIdType MakeDistributedId(int owner, svtkIdType local);

  /**
   * Set the pedigreeId -> processor distribution function that determines
   * how vertices are distributed when they are associated with
   * pedigree ID, which must be a unique label such as a URL or IP
   * address. If a nullptr function pointer is provided, the default
   * hashed distribution will be used.
   */
  void SetVertexPedigreeIdDistribution(svtkVertexPedigreeIdDistribution Func, void* userData);

  /**
   * Determine which processor owns the vertex with the given pedigree ID.
   */
  svtkIdType GetVertexOwnerByPedigreeId(const svtkVariant& pedigreeId);

  /**
   * Synchronizes all of the processors involved in this distributed
   * graph, so that all processors have a consistent view of the
   * distributed graph for the computation that follows. This routine
   * should be invoked after adding new edges into the distributed
   * graph, so that other processors will see those edges (or their
   * corresponding back-edges).
   */
  virtual void Synchronize() = 0;

  /**
   * Clones the distributed graph helper, returning another
   * distributed graph helper of the same kind that can be used in
   * another svtkGraph.
   */
  virtual svtkDistributedGraphHelper* Clone() = 0;

  //@{
  /**
   * Information Keys that distributed graphs can append to attribute arrays
   * to flag them as containing distributed IDs.  These can be used to let
   * routines that migrate vertices (either repartitioning or collecting graphs
   * to single nodes) to also modify the ids contained in the attribute arrays
   * to maintain consistency.
   */
  static svtkInformationIntegerKey* DISTRIBUTEDVERTEXIDS();
  static svtkInformationIntegerKey* DISTRIBUTEDEDGEIDS();
  //@}

protected:
  svtkDistributedGraphHelper();
  ~svtkDistributedGraphHelper() override;

  /**
   * Add a vertex, optionally with properties, to the distributed graph.
   * If vertex is non-nullptr, it will be set
   * to the newly-added (or found) vertex. Note that if propertyArr is
   * non-nullptr and the vertex data contains pedigree IDs, a vertex will
   * only be added if there is no vertex with that pedigree ID.
   */
  virtual void AddVertexInternal(svtkVariantArray* propertyArr, svtkIdType* vertex) = 0;

  /**
   * Add a vertex with the given pedigreeId to the distributed graph. If
   * vertex is non-nullptr, it will receive the newly-created vertex.
   */
  virtual void AddVertexInternal(const svtkVariant& pedigreeId, svtkIdType* vertex) = 0;

  /**
   * Add an edge (u, v) to the distributed graph. The edge may be directed
   * undirected. If edge is non-null, it will receive the newly-created edge.
   * If propertyArr is non-null, it specifies the properties that will be
   * attached to the newly-created edge.
   */
  virtual void AddEdgeInternal(
    svtkIdType u, svtkIdType v, bool directed, svtkVariantArray* propertyArr, svtkEdgeType* edge) = 0;

  /**
   * Adds an edge (u, v) and returns the new edge. The graph edge may
   * or may not be directed, depending on the given flag. If edge is
   * non-null, it will receive the newly-created edge. uPedigreeId is
   * the pedigree ID of vertex u, which will be added if no vertex by
   * that pedigree ID exists. If propertyArr is non-null, it specifies
   * the properties that will be attached to the newly-created edge.
   */
  virtual void AddEdgeInternal(const svtkVariant& uPedigreeId, svtkIdType v, bool directed,
    svtkVariantArray* propertyArr, svtkEdgeType* edge) = 0;

  /**
   * Adds an edge (u, v) and returns the new edge. The graph edge may
   * or may not be directed, depending on the given flag. If edge is
   * non-null, it will receive the newly-created edge. vPedigreeId is
   * the pedigree ID of vertex u, which will be added if no vertex
   * with that pedigree ID exists. If propertyArr is non-null, it specifies
   * the properties that will be attached to the newly-created edge.
   */
  virtual void AddEdgeInternal(svtkIdType u, const svtkVariant& vPedigreeId, bool directed,
    svtkVariantArray* propertyArr, svtkEdgeType* edge) = 0;

  /**
   * Adds an edge (u, v) and returns the new edge. The graph edge may
   * or may not be directed, depending on the given flag. If edge is
   * non-null, it will receive the newly-created edge. uPedigreeId is
   * the pedigree ID of vertex u and vPedigreeId is the pedigree ID of
   * vertex u, each of which will be added if no vertex by that
   * pedigree ID exists. If propertyArr is non-null, it specifies
   * the properties that will be attached to the newly-created edge.
   */
  virtual void AddEdgeInternal(const svtkVariant& uPedigreeId, const svtkVariant& vPedigreeId,
    bool directed, svtkVariantArray* propertyArr, svtkEdgeType* edge) = 0;

  /**
   * Try to find the vertex with the given pedigree ID. Returns the
   * vertex ID if the vertex is found, or -1 if there is no vertex
   * with that pedigree ID.
   */
  virtual svtkIdType FindVertex(const svtkVariant& pedigreeId) = 0;

  /**
   * Determine the source and target of the edge with the given
   * ID. Used internally by svtkGraph::GetSourceVertex and
   * svtkGraph::GetTargetVertex.
   */
  virtual void FindEdgeSourceAndTarget(svtkIdType id, svtkIdType* source, svtkIdType* target) = 0;

  /**
   * Attach this distributed graph helper to the given graph. This will
   * be called as part of svtkGraph::SetDistributedGraphHelper.
   */
  virtual void AttachToGraph(svtkGraph* graph);

  /**
   * The graph to which this distributed graph helper is already attached.
   */
  svtkGraph* Graph;

  /**
   * The distribution function used to map a pedigree ID to a processor.
   */
  svtkVertexPedigreeIdDistribution VertexDistribution;

  /**
   * Extra, user-specified data to be passed into the distribution function.
   */
  void* VertexDistributionUserData;

  /**
   * Bit mask to speed up decoding graph info {owner,index}
   */
  svtkIdType signBitMask;

  /**
   * Bit mask to speed up decoding graph info {owner,index}
   */
  svtkIdType highBitShiftMask;

  /**
   * Number of bits required to represent # of processors (owner)
   */
  int procBits;

  /**
   * Number of bits required to represent {vertex,edge} index
   */
  int indexBits;

private:
  svtkDistributedGraphHelper(const svtkDistributedGraphHelper&) = delete;
  void operator=(const svtkDistributedGraphHelper&) = delete;

  friend class svtkGraph;
};

#endif // svtkDistributedGraphHelper_h
