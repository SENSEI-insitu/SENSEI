/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUniformGridAMRDataIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkUniformGridAMRDataIterator
 * @brief   subclass of svtkCompositeDataIterator
 * with API to get current level and dataset index.
 *
 */

#ifndef svtkUniformGridAMRDataIterator_h
#define svtkUniformGridAMRDataIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkCompositeDataIterator.h"
#include "svtkSmartPointer.h" //for member variable Information

class svtkInformation;
class svtkAMRInformation;
class svtkAMRDataInternals;
class svtkUniformGridAMR;
class AMRIndexIterator;

class SVTKCOMMONDATAMODEL_EXPORT svtkUniformGridAMRDataIterator : public svtkCompositeDataIterator
{
public:
  static svtkUniformGridAMRDataIterator* New();
  svtkTypeMacro(svtkUniformGridAMRDataIterator, svtkCompositeDataIterator);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Returns the meta-data associated with the current item.
   * Note that this points to a single instance of svtkInformation object
   * allocated by the iterator and will be changed as soon as GoToNextItem is
   * called.
   */
  svtkInformation* GetCurrentMetaData() override;

  int HasCurrentMetaData() override { return 1; }

  /**
   * Returns the current item. Valid only when IsDoneWithTraversal() returns 0.
   */
  svtkDataObject* GetCurrentDataObject() override;

  /**
   * Flat index is an index obtained by traversing the tree in preorder.
   * This can be used to uniquely identify nodes in the tree.
   * Not valid if IsDoneWithTraversal() returns true.
   */
  unsigned int GetCurrentFlatIndex() override;

  /**
   * Returns the level for the current dataset.
   */
  virtual unsigned int GetCurrentLevel();

  /**
   * Returns the dataset index for the current data object. Valid only if the
   * current data is a leaf node i.e. no a composite dataset.
   */
  virtual unsigned int GetCurrentIndex();

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

protected:
  svtkUniformGridAMRDataIterator();
  ~svtkUniformGridAMRDataIterator() override;
  svtkSmartPointer<AMRIndexIterator> Iter;

private:
  svtkUniformGridAMRDataIterator(const svtkUniformGridAMRDataIterator&) = delete;
  void operator=(const svtkUniformGridAMRDataIterator&) = delete;

  svtkSmartPointer<svtkInformation> Information;
  svtkSmartPointer<svtkUniformGridAMR> AMR;
  svtkAMRInformation* AMRInfo;
  svtkAMRDataInternals* AMRData;

  void GetCurrentIndexPair(unsigned int& level, unsigned int& id);
};

#endif
