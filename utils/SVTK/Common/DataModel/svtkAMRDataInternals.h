/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAMRDataInternals.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkAMRDataInternals
 * @brief   container of svtkUniformGrid for an AMR data set
 *
 *
 * svtkAMRDataInternals stores a list of non-empty blocks of an AMR data set
 *
 * @sa
 * svtkOverlappingAMR, svtkAMRBox
 */

#ifndef svtkAMRDataInternals_h
#define svtkAMRDataInternals_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"
#include "svtkSmartPointer.h" //for storing smart pointers to blocks
#include <vector>            //for storing blocks

class svtkUniformGrid;
class SVTKCOMMONDATAMODEL_EXPORT svtkAMRDataInternals : public svtkObject
{
public:
  struct Block
  {
    svtkSmartPointer<svtkUniformGrid> Grid;
    unsigned int Index;
    Block(unsigned int i, svtkUniformGrid* g);
  };
  typedef std::vector<svtkAMRDataInternals::Block> BlockList;

  static svtkAMRDataInternals* New();
  svtkTypeMacro(svtkAMRDataInternals, svtkObject);

  void Initialize();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  void Insert(unsigned int index, svtkUniformGrid* grid);
  svtkUniformGrid* GetDataSet(unsigned int compositeIndex);

  virtual void ShallowCopy(svtkObject* src);

  bool Empty() const { return this->GetNumberOfBlocks() == 0; }

public:
  unsigned int GetNumberOfBlocks() const { return static_cast<unsigned int>(this->Blocks.size()); }
  const Block& GetBlock(unsigned int i) { return this->Blocks[i]; }
  const BlockList& GetAllBlocks() const { return this->Blocks; }

protected:
  svtkAMRDataInternals();
  ~svtkAMRDataInternals() override;

  void GenerateIndex(bool force = false);

  std::vector<Block> Blocks;
  std::vector<int>* InternalIndex; // map from the composite index to internal index
  bool GetInternalIndex(unsigned int compositeIndex, unsigned int& internalIndex);

private:
  svtkAMRDataInternals(const svtkAMRDataInternals&) = delete;
  void operator=(const svtkAMRDataInternals&) = delete;
};

#endif
