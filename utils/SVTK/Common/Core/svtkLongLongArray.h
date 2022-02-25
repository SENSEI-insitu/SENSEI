/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLongLongArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkLongLongArray
 * @brief   dynamic, self-adjusting array of long long
 *
 * svtkLongLongArray is an array of values of type long long.
 * It provides methods for insertion and retrieval of values and will
 * automatically resize itself to hold new data.
 *
 * This class should not be used directly, as it only exists on systems
 * where the long long type is defined.  If you need a 64 bit integer
 * data array, use svtkTypeInt64Array instead.
 */

#ifndef svtkLongLongArray_h
#define svtkLongLongArray_h

#include "svtkAOSDataArrayTemplate.h" // Real Superclass
#include "svtkCommonCoreModule.h"     // For export macro
#include "svtkDataArray.h"

// Fake the superclass for the wrappers.
#ifndef __SVTK_WRAP__
#define svtkDataArray svtkAOSDataArrayTemplate<long long>
#endif
class SVTKCOMMONCORE_EXPORT svtkLongLongArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkLongLongArray, svtkDataArray);
#ifndef __SVTK_WRAP__
#undef svtkDataArray
#endif
  static svtkLongLongArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // This macro expands to the set of method declarations that
  // make up the interface of svtkAOSDataArrayTemplate, which is ignored
  // by the wrappers.
#if defined(__SVTK_WRAP__) || defined(__WRAP_GCCXML__)
  svtkCreateWrappedArrayInterface(long long);
#endif

  /**
   * A faster alternative to SafeDownCast for downcasting svtkAbstractArrays.
   */
  static svtkLongLongArray* FastDownCast(svtkAbstractArray* source)
  {
    return static_cast<svtkLongLongArray*>(Superclass::FastDownCast(source));
  }

  /**
   * Get the minimum data value in its native type.
   */
  static long long GetDataTypeValueMin() { return SVTK_LONG_LONG_MIN; }

  /**
   * Get the maximum data value in its native type.
   */
  static long long GetDataTypeValueMax() { return SVTK_LONG_LONG_MAX; }

protected:
  svtkLongLongArray();
  ~svtkLongLongArray() override;

private:
  typedef svtkAOSDataArrayTemplate<long long> RealSuperclass;

  svtkLongLongArray(const svtkLongLongArray&) = delete;
  void operator=(const svtkLongLongArray&) = delete;
};

// Define svtkArrayDownCast implementation:
svtkArrayDownCast_FastCastMacro(svtkLongLongArray);

#endif
