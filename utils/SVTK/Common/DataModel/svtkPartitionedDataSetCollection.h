/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPartitionedDataSetCollection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPartitionedDataSetCollection
 * @brief   Composite dataset that groups datasets as a collection.
 *
 * svtkPartitionedDataSetCollection is a svtkCompositeDataSet that stores
 * a collection of svtkPartitionedDataSets. These items can represent
 * different concepts depending on the context. For example, they can
 * represent region of different materials in a simulation or parts in
 * an assembly. It is not requires that items have anything in common.
 * For example, they can have completely different point or cell arrays.
 */

#ifndef svtkPartitionedDataSetCollection_h
#define svtkPartitionedDataSetCollection_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataObjectTree.h"

class svtkPartitionedDataSet;

class SVTKCOMMONDATAMODEL_EXPORT svtkPartitionedDataSetCollection : public svtkDataObjectTree
{
public:
  static svtkPartitionedDataSetCollection* New();
  svtkTypeMacro(svtkPartitionedDataSetCollection, svtkDataObjectTree);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return class name of data type (see svtkType.h for
   * definitions).
   */
  int GetDataObjectType() override { return SVTK_PARTITIONED_DATA_SET_COLLECTION; }

  /**
   * Set the number of blocks. This will cause allocation if the new number of
   * blocks is greater than the current size. All new blocks are initialized to
   * null.
   */
  void SetNumberOfPartitionedDataSets(unsigned int numDataSets);

  /**
   * Returns the number of blocks.
   */
  unsigned int GetNumberOfPartitionedDataSets();

  /**
   * Returns the block at the given index. It is recommended that one uses the
   * iterators to iterate over composite datasets rather than using this API.
   */
  svtkPartitionedDataSet* GetPartitionedDataSet(unsigned int idx);

  /**
   * Sets the data object as the given block. The total number of blocks will
   * be resized to fit the requested block no.
   */
  void SetPartitionedDataSet(unsigned int idx, svtkPartitionedDataSet* dataset);

  /**
   * Remove the given block from the dataset.
   */
  void RemovePartitionedDataSet(unsigned int idx);

  /**
   * Returns true if meta-data is available for a given block.
   */
  int HasMetaData(unsigned int idx) { return this->Superclass::HasChildMetaData(idx); }

  /**
   * Returns the meta-data for the block. If none is already present, a new
   * svtkInformation object will be allocated. Use HasMetaData to avoid
   * allocating svtkInformation objects.
   */
  svtkInformation* GetMetaData(unsigned int idx) { return this->Superclass::GetChildMetaData(idx); }

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkPartitionedDataSetCollection* GetData(svtkInformation* info);
  static svtkPartitionedDataSetCollection* GetData(svtkInformationVector* v, int i = 0);
  //@}

  /**
   * Unhiding superclass method.
   */
  svtkInformation* GetMetaData(svtkCompositeDataIterator* iter) override
  {
    return this->Superclass::GetMetaData(iter);
  }

  /**
   * Unhiding superclass method.
   */
  int HasMetaData(svtkCompositeDataIterator* iter) override
  {
    return this->Superclass::HasMetaData(iter);
  }

protected:
  svtkPartitionedDataSetCollection();
  ~svtkPartitionedDataSetCollection() override;

private:
  svtkPartitionedDataSetCollection(const svtkPartitionedDataSetCollection&) = delete;
  void operator=(const svtkPartitionedDataSetCollection&) = delete;
};

#endif
