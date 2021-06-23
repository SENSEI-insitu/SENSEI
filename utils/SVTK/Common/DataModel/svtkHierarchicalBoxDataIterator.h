/*=========================================================================

 Program:   Visualization Toolkit
 Module:    svtkHierarchicalBoxDataIterator.h

 Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
 All rights reserved.
 See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notice for more information.

 =========================================================================*/
/**
 * @class   svtkHierarchicalBoxDataIterator
 *
 *
 *  Empty class for backwards compatibility.
 */

#ifndef svtkHierarchicalBoxDataIterator_h
#define svtkHierarchicalBoxDataIterator_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkUniformGridAMRDataIterator.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkHierarchicalBoxDataIterator
  : public svtkUniformGridAMRDataIterator
{
public:
  static svtkHierarchicalBoxDataIterator* New();
  svtkTypeMacro(svtkHierarchicalBoxDataIterator, svtkUniformGridAMRDataIterator);
  void PrintSelf(ostream& os, svtkIndent indent) override;

protected:
  svtkHierarchicalBoxDataIterator();
  ~svtkHierarchicalBoxDataIterator() override;

private:
  svtkHierarchicalBoxDataIterator(const svtkHierarchicalBoxDataIterator&) = delete;
  void operator=(const svtkHierarchicalBoxDataIterator&) = delete;
};

#endif /* SVTKHIERARCHICALBOXDATAITERATOR_H_ */
