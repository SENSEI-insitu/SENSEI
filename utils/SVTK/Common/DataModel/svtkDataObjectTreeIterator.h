/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataObjectTreeIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDataObjectTreeIterator
 * @brief   superclass for composite data iterators
 *
 * svtkDataObjectTreeIterator provides an interface for accessing datasets
 * in a collection (svtkDataObjectTreeIterator).
 */

#ifndef svtkDataObjectTreeIterator_h
#define svtkDataObjectTreeIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkCompositeDataIterator.h"
#include "svtkSmartPointer.h" //to store data sets

class svtkDataObjectTree;
class svtkDataObjectTreeInternals;
class svtkDataObjectTreeIndex;
class svtkDataObject;
class svtkInformation;

class SVTKCOMMONDATAMODEL_EXPORT svtkDataObjectTreeIterator : public svtkCompositeDataIterator
{
public:
  static svtkDataObjectTreeIterator* New();
  svtkTypeMacro(svtkDataObjectTreeIterator, svtkCompositeDataIterator);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Move the iterator to the beginning of the collection.
   */
  void GoToFirstItem() override;

  /**
   * Move the iterator to the next item in the collection.
   */
  void GoToNextItem() override;

  /**
   * Test whether the iterator is finished with the traversal.
   * Returns 1 for yes, and 0 for no.
   * It is safe to call any of the GetCurrent...() methods only when
   * IsDoneWithTraversal() returns 0.
   */
  int IsDoneWithTraversal() override;

  /**
   * Returns the current item. Valid only when IsDoneWithTraversal() returns 0.
   */
  svtkDataObject* GetCurrentDataObject() override;

  /**
   * Returns the meta-data associated with the current item.
   * Note that, depending on iterator implementation, the returned information
   * is not necessarily stored on the current object. So modifying the information
   * is forbidden.
   */
  svtkInformation* GetCurrentMetaData() override;

  /**
   * Returns if the a meta-data information object is present for the current
   * item. Return 1 on success, 0 otherwise.
   */
  int HasCurrentMetaData() override;

  /**
   * Flat index is an index obtained by traversing the tree in preorder.
   * This can be used to uniquely identify nodes in the tree.
   * Not valid if IsDoneWithTraversal() returns true.
   */
  unsigned int GetCurrentFlatIndex() override;

  //@{
  /**
   * If VisitOnlyLeaves is true, the iterator will only visit nodes
   * (sub-datasets) that are not composite. If it encounters a composite
   * data set, it will automatically traverse that composite dataset until
   * it finds non-composite datasets. With this options, it is possible to
   * visit all non-composite datasets in tree of composite datasets
   * (composite of composite of composite for example :-) ) If
   * VisitOnlyLeaves is false, GetCurrentDataObject() may return
   * svtkCompositeDataSet. By default, VisitOnlyLeaves is 1.
   */
  svtkSetMacro(VisitOnlyLeaves, svtkTypeBool);
  svtkGetMacro(VisitOnlyLeaves, svtkTypeBool);
  svtkBooleanMacro(VisitOnlyLeaves, svtkTypeBool);
  //@}

  //@{
  /**
   * If TraverseSubTree is set to true, the iterator will visit the entire tree
   * structure, otherwise it only visits the first level children. Set to 1 by
   * default.
   */
  svtkSetMacro(TraverseSubTree, svtkTypeBool);
  svtkGetMacro(TraverseSubTree, svtkTypeBool);
  svtkBooleanMacro(TraverseSubTree, svtkTypeBool);
  //@}

protected:
  svtkDataObjectTreeIterator();
  ~svtkDataObjectTreeIterator() override;

  // Use the macro to ensure MTime is updated:
  svtkSetMacro(CurrentFlatIndex, unsigned int);

  // Takes the current location to the next dataset. This traverses the tree in
  // preorder fashion.
  // If the current location is a composite dataset, next is its 1st child dataset.
  // If the current is not a composite dataset, then next is the next dataset.
  // This method gives no guarantees whether the current dataset will be
  // non-null or leaf.
  void NextInternal();

  /**
   * Returns the index for the current data object.
   */
  svtkDataObjectTreeIndex GetCurrentIndex();

  // Needs access to GetCurrentIndex().
  friend class svtkDataObjectTree;
  friend class svtkMultiDataSetInternal;

  unsigned int CurrentFlatIndex;

private:
  svtkDataObjectTreeIterator(const svtkDataObjectTreeIterator&) = delete;
  void operator=(const svtkDataObjectTreeIterator&) = delete;

  class svtkInternals;
  svtkInternals* Internals;
  friend class svtkInternals;

  svtkTypeBool TraverseSubTree;
  svtkTypeBool VisitOnlyLeaves;

  /**
   * Helper method used by svtkInternals to get access to the internals of
   * svtkDataObjectTree.
   */
  svtkDataObjectTreeInternals* GetInternals(svtkDataObjectTree*);

  // Cannot be called when this->IsDoneWithTraversal() return 1.
  void UpdateLocation();
};

#endif
