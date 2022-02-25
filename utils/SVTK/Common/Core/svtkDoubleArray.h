/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDoubleArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDoubleArray
 * @brief   dynamic, self-adjusting array of double
 *
 * svtkDoubleArray is an array of values of type double.  It provides
 * methods for insertion and retrieval of values and will
 * automatically resize itself to hold new data.
 */

#ifndef svtkDoubleArray_h
#define svtkDoubleArray_h

#include "svtkAOSDataArrayTemplate.h" // Real Superclass
#include "svtkCommonCoreModule.h"     // For export macro
#include "svtkDataArray.h"

// Fake the superclass for the wrappers.
#ifndef __SVTK_WRAP__
#define svtkDataArray svtkAOSDataArrayTemplate<double>
#endif
class SVTKCOMMONCORE_EXPORT svtkDoubleArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkDoubleArray, svtkDataArray);
#ifndef __SVTK_WRAP__
#undef svtkDataArray
#endif
  static svtkDoubleArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // This macro expands to the set of method declarations that
  // make up the interface of svtkAOSDataArrayTemplate, which is ignored
  // by the wrappers.
#if defined(__SVTK_WRAP__) || defined(__WRAP_GCCXML__)
  svtkCreateWrappedArrayInterface(double);
#endif

  /**
   * A faster alternative to SafeDownCast for downcasting svtkAbstractArrays.
   */
  static svtkDoubleArray* FastDownCast(svtkAbstractArray* source)
  {
    return static_cast<svtkDoubleArray*>(Superclass::FastDownCast(source));
  }

  /**
   * Get the minimum data value in its native type.
   */
  static double GetDataTypeValueMin() { return SVTK_DOUBLE_MIN; }

  /**
   * Get the maximum data value in its native type.
   */
  static double GetDataTypeValueMax() { return SVTK_DOUBLE_MAX; }

protected:
  svtkDoubleArray();
  ~svtkDoubleArray() override;

private:
  typedef svtkAOSDataArrayTemplate<double> RealSuperclass;

  svtkDoubleArray(const svtkDoubleArray&) = delete;
  void operator=(const svtkDoubleArray&) = delete;
};

// Define svtkArrayDownCast implementation:
svtkArrayDownCast_FastCastMacro(svtkDoubleArray);

#endif
