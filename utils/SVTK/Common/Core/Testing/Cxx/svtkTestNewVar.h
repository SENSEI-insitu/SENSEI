/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTestNewVar.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   Test
 *
 * Tests instantiations of the svtkNew class template.
 */

#ifndef svtkTestNewVar_h
#define svtkTestNewVar_h

#include "svtkNew.h"
#include "svtkObject.h"

class svtkPoints2D;

class svtkTestNewVar : public svtkObject
{
public:
  static svtkTestNewVar* New();

  svtkTypeMacro(svtkTestNewVar, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Get the reference count for the points object.
   */
  svtkIdType GetPointsRefCount();

  /**
   * This is just for testing - return the points as a svtkObject so that it can
   * be assigned to a svtkSmartPointer without including the svtkPoints2D header
   * and defeating part of the point of the test.
   */
  svtkObject* GetPoints();

  /**
   * This is just for testing - return the points as a svtkObject so that it can
   * be assigned to a svtkSmartPointer without including the svtkPoints2D header
   * and defeating part of the point of the test.
   * Using implicit conversion to raw pointer.
   */
  svtkObject* GetPoints2();

protected:
  svtkTestNewVar();
  ~svtkTestNewVar() override;

  svtkNew<svtkPoints2D> Points;

private:
  svtkTestNewVar(const svtkTestNewVar&) = delete;
  void operator=(const svtkTestNewVar&) = delete;
};

#endif
