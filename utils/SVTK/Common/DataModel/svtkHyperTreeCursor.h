/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHyperTreeCursor.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkHyperTreeCursor
 * @brief   Objects for depth-first traversal HyperTrees.
 *
 *
 * Objects that can perform depth-first traversal of HyperTrees.
 * This is an abstract class.
 * Cursors are created by the HyperTree implementation.
 *
 * @sa
 * svtkObject svtkHyperTree svtkHyperTreeGrid
 *
 * @par Thanks:
 * This class was written by Philippe Pebay, Joachim Pouderoux, and Charles Law, Kitware 2013
 * This class was modified by Guenole Harel and Jacques-Bernard Lekien 2014
 * This class was revised by Philippe Pebay, 2016
 * This work was supported by Commissariat a l'Energie Atomique (CEA/DIF)
 */

#ifndef svtkHyperTreeCursor_h
#define svtkHyperTreeCursor_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkHyperTree;

class SVTKCOMMONDATAMODEL_EXPORT svtkHyperTreeCursor : public svtkObject
{
public:
  svtkTypeMacro(svtkHyperTreeCursor, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Set the HyperTree to which the cursor is pointing.
   */
  virtual void SetTree(svtkHyperTree*) = 0;

  /**
   * Return the HyperTree to which the cursor is pointing.
   */
  virtual svtkHyperTree* GetTree() = 0;

  /**
   * Return the index of the current vertex in the tree.
   */
  virtual svtkIdType GetVertexId() = 0;

  /**
   * Is the cursor pointing to a leaf?
   */
  virtual bool IsLeaf() = 0;

  /**
   * Is the cursor at tree root?
   */
  virtual bool IsRoot() = 0;

  /**
   * Return the level of the vertex pointed by the cursor.
   * \post positive_result: result>=0
   */
  virtual unsigned int GetLevel() = 0;

  /**
   * Return the child number of the current vertex relative to its parent.
   * \pre not_root: !IsRoot().
   * \post valid_range: result>=0 && result<GetNumberOfChildren()
   */
  virtual int GetChildIndex() = 0;

  /**
   * Move the cursor to the root vertex.
   * \pre can be root
   * \post is_root: IsRoot()
   */
  virtual void ToRoot() = 0;

  /**
   * Move the cursor to the parent of the current vertex.
   * \pre not_root: !IsRoot()
   */
  virtual void ToParent() = 0;

  /**
   * Move the cursor to child `child' of the current vertex.
   * \pre not_leaf: !IsLeaf()
   * \pre valid_child: child>=0 && child<this->GetNumberOfChildren()
   */
  virtual void ToChild(int child) = 0;

  /**
   * Move the cursor to the same vertex pointed by `other'.
   * \pre other_exists: other!=0
   * \pre same_hypertree: this->SameTree(other);
   * \post equal: this->IsEqual(other)
   */
  virtual void ToSameVertex(svtkHyperTreeCursor* other) = 0;

  /**
   * Is `this' equal to `other'?
   * \pre other_exists: other!=0
   * \pre same_hypertree: this->SameTree(other);
   */
  virtual bool IsEqual(svtkHyperTreeCursor* other) = 0;

  /**
   * Create a copy of `this'.
   * \post results_exists:result!=0
   * \post same_tree: result->SameTree(this)
   */
  virtual svtkHyperTreeCursor* Clone() = 0;

  /**
   * Are `this' and `other' pointing on the same hypertree?
   * \pre other_exists: other!=0
   */
  virtual int SameTree(svtkHyperTreeCursor* other) = 0;

  /**
   * Return the number of children for each node (non-vertex leaf) of the tree.
   * \post positive_number: result>0
   */
  virtual int GetNumberOfChildren() = 0;

  /**
   * Return the dimension of the tree.
   * \post positive_result: result>0
   */
  virtual int GetDimension() = 0;

protected:
  // Constructor
  svtkHyperTreeCursor();
  ~svtkHyperTreeCursor() override;

private:
  svtkHyperTreeCursor(const svtkHyperTreeCursor&) = delete;
  void operator=(const svtkHyperTreeCursor&) = delete;
};
#endif
