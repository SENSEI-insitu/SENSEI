/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImplicitBoolean.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkImplicitBoolean
 * @brief   implicit function consisting of boolean combinations of implicit functions
 *
 * svtkImplicitBoolean is an implicit function consisting of boolean
 * combinations of implicit functions. The class has a list of functions
 * (FunctionList) that are combined according to a specified operator
 * (SVTK_UNION or SVTK_INTERSECTION or SVTK_DIFFERENCE). You can use nested
 * combinations of svtkImplicitFunction's (and/or svtkImplicitBoolean) to create
 * elaborate implicit functions.  svtkImplicitBoolean is a concrete
 * implementation of svtkImplicitFunction.
 *
 * The operators work as follows. The SVTK_UNION operator takes the minimum
 * value of all implicit functions. The SVTK_INTERSECTION operator takes the
 * maximum value of all implicit functions. The SVTK_DIFFERENCE operator
 * subtracts the 2nd through last implicit functions from the first. The
 * SVTK_UNION_OF_MAGNITUDES takes the minimum absolute value of the
 * implicit functions.
 */

#ifndef svtkImplicitBoolean_h
#define svtkImplicitBoolean_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImplicitFunction.h"

class svtkImplicitFunctionCollection;

class SVTKCOMMONDATAMODEL_EXPORT svtkImplicitBoolean : public svtkImplicitFunction
{
public:
  svtkTypeMacro(svtkImplicitBoolean, svtkImplicitFunction);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  enum OperationType
  {
    SVTK_UNION = 0,
    SVTK_INTERSECTION,
    SVTK_DIFFERENCE,
    SVTK_UNION_OF_MAGNITUDES
  };

  /**
   * Default boolean method is union.
   */
  static svtkImplicitBoolean* New();

  //@{
  /**
   * Evaluate boolean combinations of implicit function using current operator.
   */
  using svtkImplicitFunction::EvaluateFunction;
  double EvaluateFunction(double x[3]) override;
  //@}

  /**
   * Evaluate gradient of boolean combination.
   */
  void EvaluateGradient(double x[3], double g[3]) override;

  /**
   * Override modified time retrieval because of object dependencies.
   */
  svtkMTimeType GetMTime() override;

  /**
   * Add another implicit function to the list of functions.
   */
  void AddFunction(svtkImplicitFunction* in);

  /**
   * Remove a function from the list of implicit functions to boolean.
   */
  void RemoveFunction(svtkImplicitFunction* in);

  /**
   * Return the collection of implicit functions.
   */
  svtkImplicitFunctionCollection* GetFunction() { return this->FunctionList; }

  //@{
  /**
   * Specify the type of boolean operation.
   */
  svtkSetClampMacro(OperationType, int, SVTK_UNION, SVTK_UNION_OF_MAGNITUDES);
  svtkGetMacro(OperationType, int);
  void SetOperationTypeToUnion() { this->SetOperationType(SVTK_UNION); }
  void SetOperationTypeToIntersection() { this->SetOperationType(SVTK_INTERSECTION); }
  void SetOperationTypeToDifference() { this->SetOperationType(SVTK_DIFFERENCE); }
  void SetOperationTypeToUnionOfMagnitudes() { this->SetOperationType(SVTK_UNION_OF_MAGNITUDES); }
  const char* GetOperationTypeAsString();
  //@}

protected:
  svtkImplicitBoolean();
  ~svtkImplicitBoolean() override;

  svtkImplicitFunctionCollection* FunctionList;

  int OperationType;

private:
  svtkImplicitBoolean(const svtkImplicitBoolean&) = delete;
  void operator=(const svtkImplicitBoolean&) = delete;
};

//@{
/**
 * Return the boolean operation type as a descriptive character string.
 */
inline const char* svtkImplicitBoolean::GetOperationTypeAsString(void)
{
  if (this->OperationType == SVTK_UNION)
  {
    return "Union";
  }
  else if (this->OperationType == SVTK_INTERSECTION)
  {
    return "Intersection";
  }
  else if (this->OperationType == SVTK_DIFFERENCE)
  {
    return "Difference";
  }
  else
  {
    return "UnionOfMagnitudes";
  }
}
//@}

#endif
