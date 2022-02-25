/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkFloatArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkFloatArray
 * @brief   dynamic, self-adjusting array of float
 *
 * svtkFloatArray is an array of values of type float.  It provides
 * methods for insertion and retrieval of values and will
 * automatically resize itself to hold new data.
 */

#ifndef svtkFloatArray_h
#define svtkFloatArray_h

#include "svtkAOSDataArrayTemplate.h" // Real Superclass
#include "svtkCommonCoreModule.h"     // For export macro
#include "svtkDataArray.h"

// Fake the superclass for the wrappers.
#ifndef __SVTK_WRAP__
#define svtkDataArray svtkAOSDataArrayTemplate<float>
#endif
class SVTKCOMMONCORE_EXPORT svtkFloatArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkFloatArray, svtkDataArray);
#ifndef __SVTK_WRAP__
#undef svtkDataArray
#endif

  static svtkFloatArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // This macro expands to the set of method declarations that
  // make up the interface of svtkAOSDataArrayTemplate, which is ignored
  // by the wrappers.
#if defined(__SVTK_WRAP__) || defined(__WRAP_GCCXML__)
  svtkCreateWrappedArrayInterface(float);
#endif

  /**
   * A faster alternative to SafeDownCast for downcasting svtkAbstractArrays.
   */
  static svtkFloatArray* FastDownCast(svtkAbstractArray* source)
  {
    return static_cast<svtkFloatArray*>(Superclass::FastDownCast(source));
  }

  /**
   * Get the minimum data value in its native type.
   */
  static float GetDataTypeValueMin() { return SVTK_FLOAT_MIN; }

  /**
   * Get the maximum data value in its native type.
   */
  static float GetDataTypeValueMax() { return SVTK_FLOAT_MAX; }

protected:
  svtkFloatArray();
  ~svtkFloatArray() override;

private:
  typedef svtkAOSDataArrayTemplate<float> RealSuperclass;

  svtkFloatArray(const svtkFloatArray&) = delete;
  void operator=(const svtkFloatArray&) = delete;
};

// Define svtkArrayDownCast implementation:
svtkArrayDownCast_FastCastMacro(svtkFloatArray);

#endif
