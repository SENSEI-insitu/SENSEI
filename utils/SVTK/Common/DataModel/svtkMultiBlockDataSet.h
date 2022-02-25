/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMultiBlockDataSet.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkMultiBlockDataSet
 * @brief   Composite dataset that organizes datasets into
 * blocks.
 *
 * svtkMultiBlockDataSet is a svtkCompositeDataSet that stores
 * a hierarchy of datasets. The dataset collection consists of
 * multiple blocks. Each block can itself be a svtkMultiBlockDataSet, thus
 * providing for a full tree structure.
 * Sub-blocks are usually used to distribute blocks across processors.
 * For example, a 1 block dataset can be distributed as following:
 * @verbatim
 * proc 0:
 * Block 0:
 *   * ds 0
 *   * (null)
 *
 * proc 1:
 * Block 0:
 *   * (null)
 *   * ds 1
 * @endverbatim
 */

#ifndef svtkMultiBlockDataSet_h
#define svtkMultiBlockDataSet_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataObjectTree.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkMultiBlockDataSet : public svtkDataObjectTree
{
public:
  static svtkMultiBlockDataSet* New();
  svtkTypeMacro(svtkMultiBlockDataSet, svtkDataObjectTree);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return class name of data type (see svtkType.h for
   * definitions).
   */
  int GetDataObjectType() override { return SVTK_MULTIBLOCK_DATA_SET; }

  /**
   * Set the number of blocks. This will cause allocation if the new number of
   * blocks is greater than the current size. All new blocks are initialized to
   * null.
   */
  void SetNumberOfBlocks(unsigned int numBlocks);

  /**
   * Returns the number of blocks.
   */
  unsigned int GetNumberOfBlocks();

  /**
   * Returns the block at the given index. It is recommended that one uses the
   * iterators to iterate over composite datasets rather than using this API.
   */
  svtkDataObject* GetBlock(unsigned int blockno);

  /**
   * Sets the data object as the given block. The total number of blocks will
   * be resized to fit the requested block no.
   */
  void SetBlock(unsigned int blockno, svtkDataObject* block);

  /**
   * Remove the given block from the dataset.
   */
  void RemoveBlock(unsigned int blockno);

  /**
   * Returns true if meta-data is available for a given block.
   */
  int HasMetaData(unsigned int blockno) { return this->Superclass::HasChildMetaData(blockno); }

  /**
   * Returns the meta-data for the block. If none is already present, a new
   * svtkInformation object will be allocated. Use HasMetaData to avoid
   * allocating svtkInformation objects.
   */
  svtkInformation* GetMetaData(unsigned int blockno)
  {
    return this->Superclass::GetChildMetaData(blockno);
  }

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkMultiBlockDataSet* GetData(svtkInformation* info);
  static svtkMultiBlockDataSet* GetData(svtkInformationVector* v, int i = 0);
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
  svtkMultiBlockDataSet();
  ~svtkMultiBlockDataSet() override;

private:
  svtkMultiBlockDataSet(const svtkMultiBlockDataSet&) = delete;
  void operator=(const svtkMultiBlockDataSet&) = delete;
};

#endif
