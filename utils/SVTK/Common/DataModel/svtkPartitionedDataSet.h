/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPartitionedDataSet.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPartitionedDataSet
 * @brief   composite dataset to encapsulates a dataset consisting of
 * partitions.
 *
 * A svtkPartitionedDataSet dataset groups multiple datasets together.
 * For example, say a simulation running in parallel on 16 processes
 * generated 16 datasets that when considering together form a whole
 * dataset. These are referred to as the partitions of the whole dataset.
 * Now imagine that we want to load a volume of 16 partitions in a
 * visualization cluster of 4 nodes. Each node could get 4 partitions,
 * not necessarily forming a whole rectangular region. In this case,
 * it is not possible to append the 4 partitions together into a svtkImageData.
 * We can then collect these 4 partitions together using a
 * svtkPartitionedDataSet.
 *
 * It is required that all non-empty partitions have the same arrays
 * and that they can be processed together as a whole by the same kind of
 * filter. However, it is not required that they are of the same type.
 * For example, it is possible to have structured datasets together with
 * unstructured datasets as long as they are compatible meshes (i.e. can
 * be processed together for the same kind of filter).
 */

#ifndef svtkPartitionedDataSet_h
#define svtkPartitionedDataSet_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataObjectTree.h"

class svtkDataSet;
class SVTKCOMMONDATAMODEL_EXPORT svtkPartitionedDataSet : public svtkDataObjectTree
{
public:
  static svtkPartitionedDataSet* New();
  svtkTypeMacro(svtkPartitionedDataSet, svtkDataObjectTree);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return class name of data type (see svtkType.h for
   * definitions).
   */
  int GetDataObjectType() override { return SVTK_PARTITIONED_DATA_SET; }

  /**
   * Set the number of partitionss. This will cause allocation if the new number of
   * partitions is greater than the current size. All new partitions are initialized to
   * null.
   */
  void SetNumberOfPartitions(unsigned int numPartitions);

  /**
   * Returns the number of partitions.
   */
  unsigned int GetNumberOfPartitions();

  //@{
  /**
   * Returns the partition at the given index.
   */
  svtkDataSet* GetPartition(unsigned int idx);
  svtkDataObject* GetPartitionAsDataObject(unsigned int idx);
  //@}

  /**
   * Sets the data object as the given partition. The total number of partitions will
   * be resized to fit the requested partition no.
   */
  void SetPartition(unsigned int idx, svtkDataObject* partition);

  /**
   * Returns true if meta-data is available for a given partition.
   */
  int HasMetaData(unsigned int idx) { return this->Superclass::HasChildMetaData(idx); }

  /**
   * Returns the meta-data for the partition. If none is already present, a new
   * svtkInformation object will be allocated. Use HasMetaData to avoid
   * allocating svtkInformation objects.
   */
  svtkInformation* GetMetaData(unsigned int idx) { return this->Superclass::GetChildMetaData(idx); }

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkPartitionedDataSet* GetData(svtkInformation* info);
  static svtkPartitionedDataSet* GetData(svtkInformationVector* v, int i = 0);
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

  /**
   * Removes all partitions that have null datasets and resizes the dataset.
   * Note any meta data associated with the null datasets will get lost.
   */
  void RemoveNullPartitions();

protected:
  svtkPartitionedDataSet();
  ~svtkPartitionedDataSet() override;

private:
  svtkPartitionedDataSet(const svtkPartitionedDataSet&) = delete;
  void operator=(const svtkPartitionedDataSet&) = delete;
};

#endif
