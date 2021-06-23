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
/**
 * @class   svtkHierarchicalBoxDataSet
 * @brief   Backwards compatibility class
 *
 *
 * An empty class for backwards compatibility
 *
 * @sa
 * svtkUniformGridAM svtkOverlappingAMR svtkNonOverlappingAMR
 */

#ifndef svtkHierarchicalBoxDataSet_h
#define svtkHierarchicalBoxDataSet_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkOverlappingAMR.h"

class svtkInformation;
class svtkInformationVector;

class SVTKCOMMONDATAMODEL_EXPORT svtkHierarchicalBoxDataSet : public svtkOverlappingAMR
{
public:
  static svtkHierarchicalBoxDataSet* New();
  svtkTypeMacro(svtkHierarchicalBoxDataSet, svtkOverlappingAMR);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return a new iterator (the iterator has to be deleted by user).
   */
  SVTK_NEWINSTANCE svtkCompositeDataIterator* NewIterator() override;

  /**
   * Return class name of data type (see svtkType.h for definitions).
   */
  int GetDataObjectType() override { return SVTK_HIERARCHICAL_BOX_DATA_SET; }

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkHierarchicalBoxDataSet* GetData(svtkInformation* info);
  static svtkHierarchicalBoxDataSet* GetData(svtkInformationVector* v, int i = 0);
  //@}

protected:
  svtkHierarchicalBoxDataSet();
  ~svtkHierarchicalBoxDataSet() override;

private:
  svtkHierarchicalBoxDataSet(const svtkHierarchicalBoxDataSet&) = delete;
  void operator=(const svtkHierarchicalBoxDataSet&) = delete;
};

#endif
