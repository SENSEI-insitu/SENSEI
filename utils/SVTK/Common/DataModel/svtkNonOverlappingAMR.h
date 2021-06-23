/*=========================================================================

 Program:   Visualization Toolkit
 Module:    svtkNonOverlappingAMR.h

 Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
 All rights reserved.
 See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notice for more information.

 =========================================================================*/
/**
 * @class   svtkNonOverlappingAMR
 *
 *
 *  A concrete instance of svtkUniformGridAMR to store uniform grids at different
 *  levels of resolution that do not overlap with each other.
 *
 * @sa
 * svtkUniformGridAMR
 */

#ifndef svtkNonOverlappingAMR_h
#define svtkNonOverlappingAMR_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkUniformGridAMR.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkNonOverlappingAMR : public svtkUniformGridAMR
{
public:
  static svtkNonOverlappingAMR* New();
  svtkTypeMacro(svtkNonOverlappingAMR, svtkUniformGridAMR);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Returns object type (see svtkType.h for definitions).
   */
  int GetDataObjectType() override { return SVTK_NON_OVERLAPPING_AMR; }

  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkNonOverlappingAMR* GetData(svtkInformation* info)
  {
    return svtkNonOverlappingAMR::SafeDownCast(Superclass::GetData(info));
  }
  static svtkNonOverlappingAMR* GetData(svtkInformationVector* v, int i = 0)
  {
    return svtkNonOverlappingAMR::SafeDownCast(Superclass::GetData(v, i));
  }

protected:
  svtkNonOverlappingAMR();
  ~svtkNonOverlappingAMR() override;

private:
  svtkNonOverlappingAMR(const svtkNonOverlappingAMR&) = delete;
  void operator=(const svtkNonOverlappingAMR&) = delete;
};

#endif /* svtkNonOverlappingAMR_h */
