/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMultiPieceDataSet.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkMultiPieceDataSet
 * @brief   composite dataset to encapsulates pieces of
 * dataset.
 *
 * A svtkMultiPieceDataSet dataset groups multiple data pieces together.
 * For example, say that a simulation broke a volume into 16 piece so that
 * each piece can be processed with 1 process in parallel. We want to load
 * this volume in a visualization cluster of 4 nodes. Each node will get 4
 * pieces, not necessarily forming a whole rectangular piece. In this case,
 * it is not possible to append the 4 pieces together into a svtkImageData.
 * In this case, these 4 pieces can be collected together using a
 * svtkMultiPieceDataSet.
 * Note that svtkMultiPieceDataSet is intended to be included in other composite
 * datasets eg. svtkMultiBlockDataSet, svtkHierarchicalBoxDataSet. Hence the lack
 * of algorithms producting svtkMultiPieceDataSet.
 */

#ifndef svtkMultiPieceDataSet_h
#define svtkMultiPieceDataSet_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkPartitionedDataSet.h"

class svtkDataSet;
class SVTKCOMMONDATAMODEL_EXPORT svtkMultiPieceDataSet : public svtkPartitionedDataSet
{
public:
  static svtkMultiPieceDataSet* New();
  svtkTypeMacro(svtkMultiPieceDataSet, svtkPartitionedDataSet);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return class name of data type (see svtkType.h for
   * definitions).
   */
  int GetDataObjectType() override { return SVTK_MULTIPIECE_DATA_SET; }

  /**
   * Set the number of pieces. This will cause allocation if the new number of
   * pieces is greater than the current size. All new pieces are initialized to
   * null.
   */
  void SetNumberOfPieces(unsigned int numpieces) { this->SetNumberOfPartitions(numpieces); }

  /**
   * Returns the number of pieces.
   */
  unsigned int GetNumberOfPieces() { return this->GetNumberOfPartitions(); }

  //@{
  /**
   * Returns the piece at the given index.
   */
  svtkDataSet* GetPiece(unsigned int pieceno) { return this->GetPartition(pieceno); }
  svtkDataObject* GetPieceAsDataObject(unsigned int pieceno)
  {
    return this->GetPartitionAsDataObject(pieceno);
  }
  //@}

  /**
   * Sets the data object as the given piece. The total number of pieces will
   * be resized to fit the requested piece no.
   */
  void SetPiece(unsigned int pieceno, svtkDataObject* piece) { this->SetPartition(pieceno, piece); }

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkMultiPieceDataSet* GetData(svtkInformation* info);
  static svtkMultiPieceDataSet* GetData(svtkInformationVector* v, int i = 0);
  //@}

protected:
  svtkMultiPieceDataSet();
  ~svtkMultiPieceDataSet() override;

private:
  svtkMultiPieceDataSet(const svtkMultiPieceDataSet&) = delete;
  void operator=(const svtkMultiPieceDataSet&) = delete;
};

#endif
