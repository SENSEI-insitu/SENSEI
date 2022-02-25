/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkEdgeTable.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkEdgeTable
 * @brief   keep track of edges (edge is pair of integer id's)
 *
 * svtkEdgeTable is a general object for keeping track of lists of edges. An
 * edge is defined by the pair of point id's (p1,p2). Methods are available
 * to insert edges, check if edges exist, and traverse the list of edges.
 * Also, it's possible to associate attribute information with each edge.
 * The attribute information may take the form of svtkIdType id's, void*
 * pointers, or points. To store attributes, make sure that
 * InitEdgeInsertion() is invoked with the storeAttributes flag set properly.
 * If points are inserted, use the methods InitPointInsertion() and
 * InsertUniquePoint().
 */

#ifndef svtkEdgeTable_h
#define svtkEdgeTable_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkIdList;
class svtkPoints;
class svtkVoidArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkEdgeTable : public svtkObject
{
public:
  /**
   * Instantiate object assuming that 1000 edges are to be inserted.
   */
  static svtkEdgeTable* New();

  svtkTypeMacro(svtkEdgeTable, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Free memory and return to the initially instantiated state.
   */
  void Initialize();

  /**
   * Initialize the edge insertion process. Provide an estimate of the number
   * of points in a dataset (the maximum range value of p1 or p2).  The
   * storeAttributes variable controls whether attributes are to be stored
   * with the edge, and what type of attributes. If storeAttributes==1, then
   * attributes of svtkIdType can be stored. If storeAttributes==2, then
   * attributes of type void* can be stored. In either case, additional
   * memory will be required by the data structure to store attribute data
   * per each edge.  This method is used in conjunction with one of the three
   * InsertEdge() methods described below (don't mix the InsertEdge()
   * methods---make sure that the one used is consistent with the
   * storeAttributes flag set in InitEdgeInsertion()).
   */
  int InitEdgeInsertion(svtkIdType numPoints, int storeAttributes = 0);

  /**
   * Insert the edge (p1,p2) into the table. It is the user's
   * responsibility to check if the edge has already been inserted
   * (use IsEdge()). If the storeAttributes flag in InitEdgeInsertion()
   * has been set, then the method returns a unique integer id (i.e.,
   * the edge id) that can be used to set and get edge
   * attributes. Otherwise, the method will return 1. Do not mix this
   * method with the InsertEdge() method that follows.
   */
  svtkIdType InsertEdge(svtkIdType p1, svtkIdType p2);

  /**
   * Insert the edge (p1,p2) into the table with the attribute id
   * specified (make sure the attributeId >= 0). Note that the
   * attributeId is ignored if the storeAttributes variable was set to
   * 0 in the InitEdgeInsertion() method. It is the user's
   * responsibility to check if the edge has already been inserted
   * (use IsEdge()). Do not mix this method with the other two
   * InsertEdge() methods.
   */
  void InsertEdge(svtkIdType p1, svtkIdType p2, svtkIdType attributeId);

  /**
   * Insert the edge (p1,p2) into the table with the attribute id
   * specified (make sure the attributeId >= 0). Note that the
   * attributeId is ignored if the storeAttributes variable was set to
   * 0 in the InitEdgeInsertion() method. It is the user's
   * responsibility to check if the edge has already been inserted
   * (use IsEdge()). Do not mix this method with the other two
   * InsertEdge() methods.
   */
  void InsertEdge(svtkIdType p1, svtkIdType p2, void* ptr);

  /**
   * Return an integer id for the edge, or an attribute id of the edge
   * (p1,p2) if the edge has been previously defined (it depends upon
   * which version of InsertEdge() is being used); otherwise -1. The
   * unique integer id can be used to set and retrieve attributes to
   * the edge.
   */
  svtkIdType IsEdge(svtkIdType p1, svtkIdType p2);

  /**
   * Similar to above, but returns a void* pointer is InitEdgeInsertion()
   * has been called with storeAttributes==2. A nullptr pointer value
   * is returned if the edge does not exist.
   */
  void IsEdge(svtkIdType p1, svtkIdType p2, void*& ptr);

  /**
   * Initialize the point insertion process. The newPts is an object
   * representing point coordinates into which incremental insertion methods
   * place their data. The points are associated with the edge.
   */
  int InitPointInsertion(svtkPoints* newPts, svtkIdType estSize);

  /**
   * Insert a unique point on the specified edge. Invoke this method only
   * after InitPointInsertion() has been called. Return 0 if point was
   * already in the list, otherwise return 1.
   */
  int InsertUniquePoint(svtkIdType p1, svtkIdType p2, double x[3], svtkIdType& ptId);

  //@{
  /**
   * Return the number of edges that have been inserted thus far.
   */
  svtkGetMacro(NumberOfEdges, svtkIdType);
  //@}

  /**
   * Initialize traversal of edges in table.
   */
  void InitTraversal();

  /**
   * Traverse list of edges in table. Return the edge as (p1,p2), where p1
   * and p2 are point id's. Method return value is <0 if list is exhausted;
   * non-zero otherwise. The value of p1 is guaranteed to be <= p2.
   */
  svtkIdType GetNextEdge(svtkIdType& p1, svtkIdType& p2);

  /**
   * Similar to above, but fills a void* pointer if InitEdgeInsertion()
   * has been called with storeAttributes==2. A nullptr pointer value
   * is filled otherwise.  Returns 0 if list is exhausted.
   */
  int GetNextEdge(svtkIdType& p1, svtkIdType& p2, void*& ptr);

  /**
   * Reset the object and prepare for reinsertion of edges. Does not delete
   * memory like the Initialize() method.
   */
  void Reset();

protected:
  svtkEdgeTable();
  ~svtkEdgeTable() override;

  svtkIdList** Table;
  svtkIdType TableMaxId; // maximum point id inserted
  svtkIdType TableSize;  // allocated size of table
  int Position[2];
  int Extend;
  svtkIdType NumberOfEdges;
  svtkPoints* Points; // support point insertion

  int StoreAttributes;              //==0:no attributes stored;==1:svtkIdType;==2:void*
  svtkIdList** Attributes;           // used to store IdTypes attributes
  svtkVoidArray** PointerAttributes; // used to store void* pointers

  svtkIdList** Resize(svtkIdType size);

private:
  svtkEdgeTable(const svtkEdgeTable&) = delete;
  void operator=(const svtkEdgeTable&) = delete;
};

#endif
