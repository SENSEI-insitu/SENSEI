/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHierarchicalBoxDataSet.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
//
// .SECTION Description
// svtkUniformGridAMR is a concrete implementation of
// svtkCompositeDataSet. The dataset type is restricted to
// svtkUniformGrid.

#ifndef svtkUniformGridAMR_h
#define svtkUniformGridAMR_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkCompositeDataSet.h"

class svtkCompositeDataIterator;
class svtkUniformGrid;
class svtkAMRInformation;
class svtkAMRDataInternals;

class SVTKCOMMONDATAMODEL_EXPORT svtkUniformGridAMR : public svtkCompositeDataSet
{
public:
  static svtkUniformGridAMR* New();
  svtkTypeMacro(svtkUniformGridAMR, svtkCompositeDataSet);

  // Description:
  // Return a new iterator (the iterator has to be deleted by the user).
  SVTK_NEWINSTANCE svtkCompositeDataIterator* NewIterator() override;

  // Description:
  // Return class name of data type (see svtkType.h for definitions).
  int GetDataObjectType() override { return SVTK_UNIFORM_GRID_AMR; }

  // Description:  // Print internal states
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // Description:
  // Restore data object to initial
  void Initialize() override;

  // Description:
  // Initialize the AMR.
  virtual void Initialize(int numLevels, const int* blocksPerLevel);

  // Description:
  // Set/Get the data description of this uniform grid instance,
  // e.g. SVTK_XYZ_GRID
  void SetGridDescription(int gridDescription);
  int GetGridDescription();

  // Description:
  // Return the number of levels
  unsigned int GetNumberOfLevels();

  // Description:
  // Return the total number of blocks, including nullptr blocks
  virtual unsigned int GetTotalNumberOfBlocks();

  // Description:
  // Returns the number of datasets at the given level, including null blocks
  unsigned int GetNumberOfDataSets(const unsigned int level);

  // Description:
  // Retrieve the bounds of the AMR domain
  void GetBounds(double bounds[6]);
  const double* GetBounds();
  void GetMin(double min[3]);
  void GetMax(double max[3]);

  // Description:
  // Unhiding superclass method.
  void SetDataSet(svtkCompositeDataIterator* iter, svtkDataObject* dataObj) override;

  // Description:
  // At the passed in level, set grid as the idx'th block at that level.
  // idx must be less than the number of data sets at that level.
  virtual void SetDataSet(unsigned int level, unsigned int idx, svtkUniformGrid* grid);

  // Description:
  // Return the data set pointed to by iter
  svtkDataObject* GetDataSet(svtkCompositeDataIterator* iter) override;

  // Description:
  // Get the data set using the index pair
  svtkUniformGrid* GetDataSet(unsigned int level, unsigned int idx);

  // Description:
  // Retrieves the composite index associated with the data at the given
  // (level,index) pair.
  int GetCompositeIndex(const unsigned int level, const unsigned int index);

  // Description:
  // Givenes the composite Idx (as set by SetCompositeIdx) this method returns the
  // corresponding level and dataset index within the level.
  void GetLevelAndIndex(const unsigned int compositeIdx, unsigned int& level, unsigned int& idx);

  // Description:
  // Override ShallowCopy/DeepCopy and CopyStructure
  void ShallowCopy(svtkDataObject* src) override;
  void DeepCopy(svtkDataObject* src) override;
  void CopyStructure(svtkCompositeDataSet* src) override;

  // Retrieve an instance of this class from an information object.
  static svtkUniformGridAMR* GetData(svtkInformation* info);
  static svtkUniformGridAMR* GetData(svtkInformationVector* v, int i = 0);

protected:
  svtkUniformGridAMR();
  ~svtkUniformGridAMR() override;

  // Description:
  // Get/Set the meta AMR meta data
  svtkGetObjectMacro(AMRData, svtkAMRDataInternals);

  svtkAMRInformation* AMRInfo;
  svtkAMRDataInternals* AMRData;
  double Bounds[6];

  // Description:
  // Get/Set the meta AMR meta data
  svtkGetObjectMacro(AMRInfo, svtkAMRInformation);
  virtual void SetAMRInfo(svtkAMRInformation*);

private:
  svtkUniformGridAMR(const svtkUniformGridAMR&) = delete;
  void operator=(const svtkUniformGridAMR&) = delete;

  friend class svtkUniformGridAMRDataIterator;
};

#endif
