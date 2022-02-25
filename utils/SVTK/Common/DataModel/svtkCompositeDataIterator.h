/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCompositeDataIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCompositeDataIterator
 * @brief   superclass for composite data iterators
 *
 * svtkCompositeDataIterator provides an interface for accessing datasets
 * in a collection (svtkCompositeDataIterator).
 */

#ifndef svtkCompositeDataIterator_h
#define svtkCompositeDataIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkCompositeDataSet;
class svtkCompositeDataSetInternals;
class svtkCompositeDataSetIndex;
class svtkDataObject;
class svtkInformation;

class SVTKCOMMONDATAMODEL_EXPORT svtkCompositeDataIterator : public svtkObject
{
public:
  svtkTypeMacro(svtkCompositeDataIterator, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Set the composite dataset this iterator is iterating over.
   * Must be set before traversal begins.
   */
  virtual void SetDataSet(svtkCompositeDataSet* ds);
  svtkGetObjectMacro(DataSet, svtkCompositeDataSet);
  //@}

  /**
   * Begin iterating over the composite dataset structure.
   */
  virtual void InitTraversal();

  /**
   * Begin iterating over the composite dataset structure in reverse order.
   */
  virtual void InitReverseTraversal();

  /**
   * Move the iterator to the beginning of the collection.
   */
  virtual void GoToFirstItem() = 0;

  /**
   * Move the iterator to the next item in the collection.
   */
  virtual void GoToNextItem() = 0;

  /**
   * Test whether the iterator is finished with the traversal.
   * Returns 1 for yes, and 0 for no.
   * It is safe to call any of the GetCurrent...() methods only when
   * IsDoneWithTraversal() returns 0.
   */
  virtual int IsDoneWithTraversal() = 0;

  /**
   * Returns the current item. Valid only when IsDoneWithTraversal() returns 0.
   */
  virtual svtkDataObject* GetCurrentDataObject() = 0;

  /**
   * Returns the meta-data associated with the current item. This will allocate
   * a new svtkInformation object is none is already present. Use
   * HasCurrentMetaData to avoid unnecessary creation of svtkInformation objects.
   */
  virtual svtkInformation* GetCurrentMetaData() = 0;

  /**
   * Returns if the a meta-data information object is present for the current
   * item. Return 1 on success, 0 otherwise.
   */
  virtual int HasCurrentMetaData() = 0;

  //@{
  /**
   * If SkipEmptyNodes is true, then nullptr datasets will be skipped. Default is
   * true.
   */
  svtkSetMacro(SkipEmptyNodes, svtkTypeBool);
  svtkGetMacro(SkipEmptyNodes, svtkTypeBool);
  svtkBooleanMacro(SkipEmptyNodes, svtkTypeBool);
  //@}

  /**
   * Flat index is an index to identify the data in a composite data structure
   */
  virtual unsigned int GetCurrentFlatIndex() = 0;

  //@{
  /**
   * Returns if the iteration is in reverse order.
   */
  svtkGetMacro(Reverse, int);
  //@}

protected:
  svtkCompositeDataIterator();
  ~svtkCompositeDataIterator() override;

  // Use macro to ensure MTime is updated:
  svtkSetMacro(Reverse, int);

  svtkTypeBool SkipEmptyNodes;
  int Reverse;
  svtkCompositeDataSet* DataSet;

private:
  svtkCompositeDataIterator(const svtkCompositeDataIterator&) = delete;
  void operator=(const svtkCompositeDataIterator&) = delete;
};

#endif
