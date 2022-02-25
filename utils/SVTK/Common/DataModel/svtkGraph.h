/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGraph.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------*/
/**
 * @class   svtkGraph
 * @brief   Base class for graph data types.
 *
 *
 * svtkGraph is the abstract base class that provides all read-only API for graph
 * data types. A graph consists of a collection of vertices and a
 * collection of edges connecting pairs of vertices. The svtkDirectedGraph
 * subclass represents a graph whose edges have inherent order from source
 * vertex to target vertex, while svtkUndirectedGraph is a graph whose edges
 * have no inherent ordering.
 *
 * Graph vertices may be traversed in two ways. In the current implementation,
 * all vertices are assigned consecutive ids starting at zero, so they may
 * be traversed in a simple for loop from 0 to graph->GetNumberOfVertices() - 1.
 * You may alternately create a svtkVertexListIterator and call graph->GetVertices(it).
 * it->Next() will return the id of the next vertex, while it->HasNext() indicates
 * whether there are more vertices in the graph.
 * This is the preferred method, since in the future graphs may support filtering
 * or subsetting where the vertex ids may not be contiguous.
 *
 * Graph edges must be traversed through iterators. To traverse all edges
 * in a graph, create an instance of svtkEdgeListIterator and call graph->GetEdges(it).
 * it->Next() returns lightweight svtkEdgeType structures, which contain the public
 * fields Id, Source and Target. Id is the identifier for the edge, which may
 * be used to look up values in assiciated edge data arrays. Source and Target
 * store the ids of the source and target vertices of the edge. Note that the
 * edge list iterator DOES NOT necessarily iterate over edges in order of ascending
 * id. To traverse edges from wrapper code (Python, Java), use
 * it->NextGraphEdge() instead of it->Next().  This will return a heavyweight,
 * wrappable svtkGraphEdge object, which has the same fields as svtkEdgeType
 * accessible through getter methods.
 *
 * To traverse all edges outgoing from a vertex, create a svtkOutEdgeIterator and
 * call graph->GetOutEdges(v, it). it->Next() returns a lightweight svtkOutEdgeType
 * containing the fields Id and Target. The source of the edge is always the
 * vertex that was passed as an argument to GetOutEdges().
 * Incoming edges may be similarly traversed with svtkInEdgeIterator, which returns
 * svtkInEdgeType structures with Id and Source fields.
 * Both svtkOutEdgeIterator and svtkInEdgeIterator also provide the wrapper functions
 * NextGraphEdge() which return svtkGraphEdge objects.
 *
 * An additional iterator, svtkAdjacentVertexIterator can traverse outgoing vertices
 * directly, instead needing to parse through edges. Initialize the iterator by
 * calling graph->GetAdjacentVertices(v, it).
 *
 * svtkGraph has two instances of svtkDataSetAttributes for associated
 * vertex and edge data. It also has a svtkPoints instance which may store
 * x,y,z locations for each vertex. This is populated by filters such as
 * svtkGraphLayout and svtkAssignCoordinates.
 *
 * All graph types share the same implementation, so the structure of one
 * may be shared among multiple graphs, even graphs of different types.
 * Structures from svtkUndirectedGraph and svtkMutableUndirectedGraph may be
 * shared directly.  Structures from svtkDirectedGraph, svtkMutableDirectedGraph,
 * and svtkTree may be shared directly with the exception that setting a
 * structure to a tree requires that a "is a tree" test passes.
 *
 * For graph types that are known to be compatible, calling ShallowCopy()
 * or DeepCopy() will work as expected.  When the outcome of a conversion
 * is unknown (i.e. setting a graph to a tree), CheckedShallowCopy() and
 * CheckedDeepCopy() exist which are identical to ShallowCopy() and DeepCopy(),
 * except that instead of emitting an error for an incompatible structure,
 * the function returns false.  This allows you to programmatically check
 * structure compatibility without causing error messages.
 *
 * To construct a graph, use svtkMutableDirectedGraph or
 * svtkMutableUndirectedGraph. You may then use CheckedShallowCopy
 * to set the contents of a mutable graph type into one of the non-mutable
 * types svtkDirectedGraph, svtkUndirectedGraph.
 * To construct a tree, use svtkMutableDirectedGraph, with directed edges
 * which point from the parent to the child, then use CheckedShallowCopy
 * to set the structure to a svtkTree.
 *
 * @warning
 * All copy operations implement copy-on-write. The structures are initially
 * shared, but if one of the graphs is modified, the structure is copied
 * so that to the user they function as if they were deep copied. This means
 * that care must be taken if different threads are accessing different graph
 * instances that share the same structure. Race conditions may develop if
 * one thread is modifying the graph at the same time that another graph is
 * copying the structure.
 *
 * @par Vertex pedigree IDs:
 * The vertices in a svtkGraph can be associated with pedigree IDs
 * through GetVertexData()->SetPedigreeIds. In this case, there is a
 * 1-1 mapping between pedigree Ids and vertices. One can query the
 * vertex ID based on the pedigree ID using FindVertex, add new
 * vertices by pedigree ID with AddVertex, and add edges based on the
 * pedigree IDs of the source and target vertices. For example,
 * AddEdge("Here", "There") will find (or add) vertices with pedigree
 * ID "Here" and "There" and then introduce an edge from "Here" to
 * "There".
 *
 * @par Vertex pedigree IDs:
 * To configure the svtkGraph with a pedigree ID mapping, create a
 * svtkDataArray that will store the pedigree IDs and set that array as
 * the pedigree ID array for the vertices via
 * GetVertexData()->SetPedigreeIds().
 *
 *
 * @par Distributed graphs:
 * svtkGraph instances can be distributed across multiple machines, to
 * allow the construction and manipulation of graphs larger than a
 * single machine could handle. A distributed graph will typically be
 * distributed across many different nodes within a cluster, using the
 * Message Passing Interface (MPI) to allow those cluster nodes to
 * communicate.
 *
 * @par Distributed graphs:
 * An empty svtkGraph can be made into a distributed graph by attaching
 * an instance of a svtkDistributedGraphHelper via the
 * SetDistributedGraphHelper() method. To determine whether a graph is
 * distributed or not, call GetDistributedGraphHelper() and check
 * whether the result is non-nullptr. For a distributed graph, the number
 * of processors across which the graph is distributed can be
 * retrieved by extracting the value for the DATA_NUMBER_OF_PIECES key
 * in the svtkInformation object (retrieved by GetInformation())
 * associated with the graph. Similarly, the value corresponding to
 * the DATA_PIECE_NUMBER key of the svtkInformation object describes
 * which piece of the data this graph instance provides.
 *
 * @par Distributed graphs:
 * Distributed graphs behave somewhat differently from non-distributed
 * graphs, and will require special care. In a distributed graph, each
 * of the processors will contain a subset of the vertices in the
 * graph. That subset of vertices can be accessed via the
 * svtkVertexListIterator produced by GetVertices().
 * GetNumberOfVertices(), therefore, returns the number of vertices
 * stored locally: it does not account for vertices stored on other
 * processors. A vertex (or edge) is identified by both the rank of
 * its owning processor and by its index within that processor, both
 * of which are encoded within the svtkIdType value that describes that
 * vertex (or edge). The owning processor is a value between 0 and
 * P-1, where P is the number of processors across which the svtkGraph
 * has been distributed. The local index will be a value between 0 and
 * GetNumberOfVertices(), for vertices, or GetNumberOfEdges(), for
 * edges, and can be used to access the local parts of distributed
 * data arrays. When given a svtkIdType identifying a vertex, one can
 * determine the owner of the vertex with
 * svtkDistributedGraphHelper::GetVertexOwner() and the local index
 * with svtkDistributedGraphHelper::GetVertexIndex(). With edges, the
 * appropriate methods are svtkDistributedGraphHelper::GetEdgeOwner()
 * and svtkDistributedGraphHelper::GetEdgeIndex(), respectively. To
 * construct a svtkIdType representing either a vertex or edge given
 * only its owner and local index, use
 * svtkDistributedGraphHelper::MakeDistributedId().
 *
 * @par Distributed graphs:
 * The edges in a distributed graph are always stored on the
 * processors that own the vertices named by the edge. For example,
 * given a directed edge (u, v), the edge will be stored in the
 * out-edges list for vertex u on the processor that owns u, and in
 * the in-edges list for vertex v on the processor that owns v. This
 * "row-wise" decomposition of the graph means that, for any vertex
 * that is local to a processor, that processor can look at all of the
 * incoming and outgoing edges of the graph. Processors cannot,
 * however, access the incoming or outgoing edge lists of vertex owned
 * by other processors. Vertices owned by other processors will not be
 * encountered when traversing the vertex list via GetVertices(), but
 * may be encountered by traversing the in- and out-edge lists of
 * local vertices or the edge list.
 *
 * @par Distributed graphs:
 * Distributed graphs can have pedigree IDs for the vertices in the
 * same way that non-distributed graphs can. In this case, the
 * distribution of the vertices in the graph is based on pedigree
 * ID. For example, a vertex with the pedigree ID "Here" might land on
 * processor 0 while a vertex pedigree ID "There" would end up on
 * processor 3. By default, the pedigree IDs themselves are hashed to
 * give a random (and, hopefully, even) distribution of the
 * vertices. However, one can provide a different vertex distribution
 * function by calling
 * svtkDistributedGraphHelper::SetVertexPedigreeIdDistribution.  Once a
 * distributed graph has pedigree IDs, the no-argument AddVertex()
 * method can no longer be used. Additionally, once a vertex has a
 * pedigree ID, that pedigree ID should not be changed unless the user
 * can guarantee that the vertex distribution will still map that
 * vertex to the same processor where it already resides.
 *
 * @sa
 * svtkDirectedGraph svtkUndirectedGraph svtkMutableDirectedGraph
 * svtkMutableUndirectedGraph svtkTree svtkDistributedGraphHelper
 *
 * @par Thanks:
 * Thanks to Brian Wylie, Timothy Shead, Ken Moreland of Sandia National
 * Laboratories and Douglas Gregor of Indiana University for designing these
 * classes.
 */

#ifndef svtkGraph_h
#define svtkGraph_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataObject.h"

class svtkAdjacentVertexIterator;
class svtkCellArray;
class svtkEdgeListIterator;
class svtkDataSetAttributes;
class svtkDirectedGraph;
class svtkGraphEdge;
class svtkGraphEdgePoints;
class svtkDistributedGraphHelper;
class svtkGraphInternals;
class svtkIdTypeArray;
class svtkInEdgeIterator;
class svtkOutEdgeIterator;
class svtkPoints;
class svtkUndirectedGraph;
class svtkVertexListIterator;
class svtkVariant;
class svtkVariantArray;

// Forward declare some boost stuff even if boost wrappers
// are turned off.
namespace boost
{
class svtk_edge_iterator;
class svtk_out_edge_pointer_iterator;
class svtk_in_edge_pointer_iterator;
}

// Edge structures.
struct svtkEdgeBase
{
  svtkEdgeBase() {}
  svtkEdgeBase(svtkIdType id)
    : Id(id)
  {
  }
  svtkIdType Id;
};

struct svtkOutEdgeType : svtkEdgeBase
{
  svtkOutEdgeType() {}
  svtkOutEdgeType(svtkIdType t, svtkIdType id)
    : svtkEdgeBase(id)
    , Target(t)
  {
  }
  svtkIdType Target;
};

struct svtkInEdgeType : svtkEdgeBase
{
  svtkInEdgeType() {}
  svtkInEdgeType(svtkIdType s, svtkIdType id)
    : svtkEdgeBase(id)
    , Source(s)
  {
  }
  svtkIdType Source;
};

struct svtkEdgeType : svtkEdgeBase
{
  svtkEdgeType() {}
  svtkEdgeType(svtkIdType s, svtkIdType t, svtkIdType id)
    : svtkEdgeBase(id)
    , Source(s)
    , Target(t)
  {
  }
  svtkIdType Source;
  svtkIdType Target;
};

class SVTKCOMMONDATAMODEL_EXPORT svtkGraph : public svtkDataObject
{
public:
  svtkTypeMacro(svtkGraph, svtkDataObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Get the vertex or edge data.
   */
  svtkGetObjectMacro(VertexData, svtkDataSetAttributes);
  svtkGetObjectMacro(EdgeData, svtkDataSetAttributes);
  //@}

  /**
   * Return what type of dataset this is.
   */
  int GetDataObjectType() override { return SVTK_GRAPH; }

  /**
   * Initialize to an empty graph.
   */
  void Initialize() override;

  //@{
  /**
   * These methods return the point (0,0,0) until the points structure
   * is created, when it returns the actual point position. In a
   * distributed graph, only the points for local vertices can be
   * retrieved.
   */
  double* GetPoint(svtkIdType ptId);
  void GetPoint(svtkIdType ptId, double x[3]);
  //@}

  //@{
  /**
   * Returns the points array for this graph.
   * If points is not yet constructed, generates and returns
   * a new points array filled with (0,0,0) coordinates. In a
   * distributed graph, only the points for local vertices can be
   * retrieved or modified.
   */
  svtkPoints* GetPoints();
  virtual void SetPoints(svtkPoints* points);
  //@}

  /**
   * Compute the bounds of the graph. In a distributed graph, this
   * computes the bounds around the local part of the graph.
   */
  void ComputeBounds();

  //@{
  /**
   * Return a pointer to the geometry bounding box in the form
   * (xmin,xmax, ymin,ymax, zmin,zmax). In a distributed graph, this
   * computes the bounds around the local part of the graph.
   */
  double* GetBounds();
  void GetBounds(double bounds[6]);
  //@}

  /**
   * The modified time of the graph.
   */
  svtkMTimeType GetMTime() override;

  /**
   * Initializes the out edge iterator to iterate over
   * all outgoing edges of vertex v.  For an undirected graph,
   * returns all incident edges. In a distributed graph, the vertex
   * v must be local to this processor.
   */
  virtual void GetOutEdges(svtkIdType v, svtkOutEdgeIterator* it);

  /**
   * The total of all incoming and outgoing vertices for vertex v.
   * For undirected graphs, this is simply the number of edges incident
   * to v. In a distributed graph, the vertex v must be local to this
   * processor.
   */
  virtual svtkIdType GetDegree(svtkIdType v);

  /**
   * The number of outgoing edges from vertex v.
   * For undirected graphs, returns the same as GetDegree(). In a
   * distributed graph, the vertex v must be local to this processor.
   */
  virtual svtkIdType GetOutDegree(svtkIdType v);

  /**
   * Random-access method for retrieving outgoing edges from vertex v.
   */
  virtual svtkOutEdgeType GetOutEdge(svtkIdType v, svtkIdType index);

  /**
   * Random-access method for retrieving outgoing edges from vertex v.
   * The method fills the svtkGraphEdge instance with the id, source, and
   * target of the edge. This method is provided for wrappers,
   * GetOutEdge(svtkIdType, svtkIdType) is preferred.
   */
  virtual void GetOutEdge(svtkIdType v, svtkIdType index, svtkGraphEdge* e);

  /**
   * Initializes the in edge iterator to iterate over
   * all incoming edges to vertex v.  For an undirected graph,
   * returns all incident edges. In a distributed graph, the vertex
   * v must be local to this processor.
   */
  virtual void GetInEdges(svtkIdType v, svtkInEdgeIterator* it);

  /**
   * The number of incoming edges to vertex v.
   * For undirected graphs, returns the same as GetDegree(). In a
   * distributed graph, the vertex v must be local to this processor.
   */
  virtual svtkIdType GetInDegree(svtkIdType v);

  /**
   * Random-access method for retrieving incoming edges to vertex v.
   */
  virtual svtkInEdgeType GetInEdge(svtkIdType v, svtkIdType index);

  /**
   * Random-access method for retrieving incoming edges to vertex v.
   * The method fills the svtkGraphEdge instance with the id, source, and
   * target of the edge. This method is provided for wrappers,
   * GetInEdge(svtkIdType, svtkIdType) is preferred.
   */
  virtual void GetInEdge(svtkIdType v, svtkIdType index, svtkGraphEdge* e);

  /**
   * Initializes the adjacent vertex iterator to iterate over
   * all outgoing vertices from vertex v.  For an undirected graph,
   * returns all adjacent vertices. In a distributed graph, the vertex
   * v must be local to this processor.
   */
  virtual void GetAdjacentVertices(svtkIdType v, svtkAdjacentVertexIterator* it);

  /**
   * Initializes the edge list iterator to iterate over all
   * edges in the graph. Edges may not be traversed in order of
   * increasing edge id. In a distributed graph, this returns edges
   * that are stored locally.
   */
  virtual void GetEdges(svtkEdgeListIterator* it);

  /**
   * The number of edges in the graph. In a distributed graph,
   * this returns the number of edges stored locally.
   */
  virtual svtkIdType GetNumberOfEdges();

  /**
   * Initializes the vertex list iterator to iterate over all
   * vertices in the graph. In a distributed graph, the iterator
   * traverses all local vertices.
   */
  virtual void GetVertices(svtkVertexListIterator* it);

  /**
   * The number of vertices in the graph. In a distributed graph,
   * returns the number of local vertices in the graph.
   */
  virtual svtkIdType GetNumberOfVertices();

  /**
   * Sets the distributed graph helper of this graph, turning it into a
   * distributed graph. This operation can only be executed on an empty
   * graph.
   */
  void SetDistributedGraphHelper(svtkDistributedGraphHelper* helper);

  /**
   * Retrieves the distributed graph helper for this graph
   */
  svtkDistributedGraphHelper* GetDistributedGraphHelper();

  /**
   * Retrieve the vertex with the given pedigree ID. If successful,
   * returns the ID of the vertex. Otherwise, either the vertex data
   * does not have a pedigree ID array or there is no vertex with the
   * given pedigree ID, so this function returns -1.
   * If the graph is a distributed graph, this method will return the
   * Distributed-ID of the vertex.
   */
  svtkIdType FindVertex(const svtkVariant& pedigreeID);

  /**
   * Shallow copies the data object into this graph.
   * If it is an incompatible graph, reports an error.
   */
  void ShallowCopy(svtkDataObject* obj) override;

  /**
   * Deep copies the data object into this graph.
   * If it is an incompatible graph, reports an error.
   */
  void DeepCopy(svtkDataObject* obj) override;

  /**
   * Does a shallow copy of the topological information,
   * but not the associated attributes.
   */
  virtual void CopyStructure(svtkGraph* g);

  /**
   * Performs the same operation as ShallowCopy(),
   * but instead of reporting an error for an incompatible graph,
   * returns false.
   */
  virtual bool CheckedShallowCopy(svtkGraph* g);

  /**
   * Performs the same operation as DeepCopy(),
   * but instead of reporting an error for an incompatible graph,
   * returns false.
   */
  virtual bool CheckedDeepCopy(svtkGraph* g);

  /**
   * Reclaim unused memory.
   */
  virtual void Squeeze();

  /**
   * Return the actual size of the data in kibibytes (1024 bytes). This number
   * is valid only after the pipeline has updated. The memory size
   * returned is guaranteed to be greater than or equal to the
   * memory required to represent the data (e.g., extra space in
   * arrays, etc. are not included in the return value).
   */
  unsigned long GetActualMemorySize() override;

  //@{
  /**
   * Retrieve a graph from an information vector.
   */
  static svtkGraph* GetData(svtkInformation* info);
  static svtkGraph* GetData(svtkInformationVector* v, int i = 0);
  //@}

  /**
   * Reorder the outgoing vertices of a vertex.
   * The vertex list must have the same elements as the current out edge
   * list, just in a different order.
   * This method does not change the topology of the graph.
   * In a distributed graph, the vertex v must be local.
   */
  void ReorderOutVertices(svtkIdType v, svtkIdTypeArray* vertices);

  /**
   * Returns true if both graphs point to the same adjacency structure.
   * Can be used to test the copy-on-write feature of the graph.
   */
  bool IsSameStructure(svtkGraph* other);

  //@{
  /**
   * Retrieve the source and target vertices for an edge id.
   * NOTE: The first time this is called, the graph will build
   * a mapping array from edge id to source/target that is the
   * same size as the number of edges in the graph. If you have
   * access to a svtkOutEdgeType, svtkInEdgeType, svtkEdgeType, or
   * svtkGraphEdge, you should directly use these structures
   * to look up the source or target instead of this method.
   */
  svtkIdType GetSourceVertex(svtkIdType e);
  svtkIdType GetTargetVertex(svtkIdType e);
  //@}

  //@{
  /**
   * Get/Set the internal edge control points associated with each edge.
   * The size of the pts array is 3*npts, and holds the x,y,z
   * location of each edge control point.
   */
  void SetEdgePoints(svtkIdType e, svtkIdType npts, const double pts[]) SVTK_SIZEHINT(pts, 3 * npts);
  void GetEdgePoints(svtkIdType e, svtkIdType& npts, double*& pts) SVTK_SIZEHINT(pts, 3 * npts);
  //@}

  /**
   * Get the number of edge points associated with an edge.
   */
  svtkIdType GetNumberOfEdgePoints(svtkIdType e);

  /**
   * Get the x,y,z location of a point along edge e.
   */
  double* GetEdgePoint(svtkIdType e, svtkIdType i) SVTK_SIZEHINT(3);

  /**
   * Clear all points associated with an edge.
   */
  void ClearEdgePoints(svtkIdType e);

  /**
   * Set an x,y,z location of a point along an edge.
   * This assumes there is already a point at location i, and simply
   * overwrites it.
   */
  void SetEdgePoint(svtkIdType e, svtkIdType i, const double x[3]);
  void SetEdgePoint(svtkIdType e, svtkIdType i, double x, double y, double z)
  {
    double p[3] = { x, y, z };
    this->SetEdgePoint(e, i, p);
  }

  /**
   * Adds a point to the end of the list of edge points for a certain edge.
   */
  void AddEdgePoint(svtkIdType e, const double x[3]);
  void AddEdgePoint(svtkIdType e, double x, double y, double z)
  {
    double p[3] = { x, y, z };
    this->AddEdgePoint(e, p);
  }

  //@{
  /**
   * Copy the internal edge point data from another graph into this graph.
   * Both graphs must have the same number of edges.
   */
  void ShallowCopyEdgePoints(svtkGraph* g);
  void DeepCopyEdgePoints(svtkGraph* g);
  //@}

  /**
   * Returns the internal representation of the graph. If modifying is
   * true, then the returned svtkGraphInternals object will be unique to
   * this svtkGraph object.
   */
  svtkGraphInternals* GetGraphInternals(bool modifying);

  /**
   * Fills a list of edge indices with the edges contained in the induced
   * subgraph formed by the vertices in the vertex list.
   */
  void GetInducedEdges(svtkIdTypeArray* verts, svtkIdTypeArray* edges);

  /**
   * Returns the attributes of the data object as a svtkFieldData.
   * This returns non-null values in all the same cases as GetAttributes,
   * in addition to the case of FIELD, which will return the field data
   * for any svtkDataObject subclass.
   */
  svtkFieldData* GetAttributesAsFieldData(int type) override;

  /**
   * Get the number of elements for a specific attribute type (VERTEX, EDGE, etc.).
   */
  svtkIdType GetNumberOfElements(int type) override;

  /**
   * Dump the contents of the graph to standard output.
   */
  void Dump();

  /**
   * Returns the Id of the edge between vertex a and vertex b.
   * This is independent of directionality of the edge, that is,
   * if edge A->B exists or if edge B->A exists, this function will
   * return its Id. If multiple edges exist between a and b, here is no guarantee
   * about which one will be returned.
   * Returns -1 if no edge exists between a and b.
   */
  svtkIdType GetEdgeId(svtkIdType a, svtkIdType b);

  /**
   * Convert the graph to a directed graph.
   */
  bool ToDirectedGraph(svtkDirectedGraph* g);

  /**
   * Convert the graph to an undirected graph.
   */
  bool ToUndirectedGraph(svtkUndirectedGraph* g);

protected:
  svtkGraph();
  ~svtkGraph() override;

  /**
   * Protected method for adding vertices, optionally with properties,
   * used by mutable subclasses. If vertex is non-null, it will be set
   * to the newly-added (or found) vertex. Note that if propertyArr is
   * non-null and the vertex data contains pedigree IDs, a vertex will
   * only be added if there is no vertex with that pedigree ID.
   */
  void AddVertexInternal(svtkVariantArray* propertyArr = nullptr, svtkIdType* vertex = nullptr);

  /**
   * Adds a vertex with the given pedigree ID to the graph. If a vertex with
   * this pedigree ID already exists, no new vertex is added, but the vertex
   * argument is set to the ID of the existing vertex.  Otherwise, a
   * new vertex is added and its ID is provided.
   */
  void AddVertexInternal(const svtkVariant& pedigree, svtkIdType* vertex);

  //@{
  /**
   * Protected method for adding edges of a certain directedness used
   * by mutable subclasses. If propertyArr is non-null, it specifies
   * the properties to be attached to the newly-created edge. If
   * non-null, edge will receive the newly-added edge.
   */
  void AddEdgeInternal(
    svtkIdType u, svtkIdType v, bool directed, svtkVariantArray* propertyArr, svtkEdgeType* edge);
  void AddEdgeInternal(const svtkVariant& uPedigree, svtkIdType v, bool directed,
    svtkVariantArray* propertyArr, svtkEdgeType* edge);
  void AddEdgeInternal(svtkIdType u, const svtkVariant& vPedigree, bool directed,
    svtkVariantArray* propertyArr, svtkEdgeType* edge);
  void AddEdgeInternal(const svtkVariant& uPedigree, const svtkVariant& vPedigree, bool directed,
    svtkVariantArray* propertyArr, svtkEdgeType* edge);
  //@}

  /**
   * Removes a vertex from the graph, along with any adjacent edges.
   * This invalidates the id of the last vertex, since it is reassigned to v.
   */
  void RemoveVertexInternal(svtkIdType v, bool directed);

  /**
   * Removes an edge from the graph.
   * This invalidates the id of the last edge, since it is reassigned to e.
   */
  void RemoveEdgeInternal(svtkIdType e, bool directed);

  /**
   * Removes a collection of vertices from the graph, along with any adjacent edges.
   */
  void RemoveVerticesInternal(svtkIdTypeArray* arr, bool directed);

  /**
   * Removes a collection of edges from the graph.
   */
  void RemoveEdgesInternal(svtkIdTypeArray* arr, bool directed);

  /**
   * Subclasses override this method to accept the structure
   * based on their requirements.
   */
  virtual bool IsStructureValid(svtkGraph* g) = 0;

  /**
   * Copy internal data structure.
   */
  virtual void CopyInternal(svtkGraph* g, bool deep);

  /**
   * The adjacency list internals of this graph.
   */
  svtkGraphInternals* Internals;

  /**
   * The distributed graph helper. Only non-nullptr for distributed graphs.
   */
  svtkDistributedGraphHelper* DistributedHelper;

  /**
   * Private method for setting internals.
   */
  void SetInternals(svtkGraphInternals* internals);

  /**
   * The structure for holding the edge points.
   */
  svtkGraphEdgePoints* EdgePoints;

  /**
   * Private method for setting edge points.
   */
  void SetEdgePoints(svtkGraphEdgePoints* edgePoints);

  /**
   * If this instance does not own its internals, it makes a copy of the
   * internals.  This is called before any write operation.
   */
  void ForceOwnership();

  //@{
  /**
   * Fast access functions for iterators.
   */
  virtual void GetOutEdges(svtkIdType v, const svtkOutEdgeType*& edges, svtkIdType& nedges);
  virtual void GetInEdges(svtkIdType v, const svtkInEdgeType*& edges, svtkIdType& nedges);
  //@}

  /**
   * Builds a mapping from edge id to source/target vertex id.
   */
  void BuildEdgeList();

  //@{
  /**
   * Friend iterator classes.
   */
  friend class svtkAdjacentVertexIterator;
  friend class svtkEdgeListIterator;
  friend class svtkInEdgeIterator;
  friend class svtkOutEdgeIterator;
  friend class boost::svtk_edge_iterator;
  friend class boost::svtk_in_edge_pointer_iterator;
  friend class boost::svtk_out_edge_pointer_iterator;
  //@}

  //@{
  /**
   * The vertex and edge data.
   */
  svtkDataSetAttributes* VertexData;
  svtkDataSetAttributes* EdgeData;
  //@}

  /**
   * (xmin,xmax, ymin,ymax, zmin,zmax) geometric bounds.
   */
  double Bounds[6];

  /**
   * Time at which bounds were computed.
   */
  svtkTimeStamp ComputeTime;

  //@{
  /**
   * The vertex locations.
   */
  svtkPoints* Points;
  static double DefaultPoint[3];
  //@}

  //@{
  /**
   * The optional mapping from edge id to source/target ids.
   */
  svtkGetObjectMacro(EdgeList, svtkIdTypeArray);
  virtual void SetEdgeList(svtkIdTypeArray* list);
  svtkIdTypeArray* EdgeList;
  //@}

private:
  svtkGraph(const svtkGraph&) = delete;
  void operator=(const svtkGraph&) = delete;
};

bool SVTKCOMMONDATAMODEL_EXPORT operator==(svtkEdgeBase e1, svtkEdgeBase e2);
bool SVTKCOMMONDATAMODEL_EXPORT operator!=(svtkEdgeBase e1, svtkEdgeBase e2);
SVTKCOMMONDATAMODEL_EXPORT ostream& operator<<(ostream& out, svtkEdgeBase e);

#endif
