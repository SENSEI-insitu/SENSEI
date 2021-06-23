/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLongArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkLongArray
 * @brief   dynamic, self-adjusting array of long
 *
 * svtkLongArray is an array of values of type long.  It provides
 * methods for insertion and retrieval of values and will
 * automatically resize itself to hold new data.
 *
 * The C++ standard does not define the exact size of the long type, so use
 * of this type directly is discouraged.  If an array of 32 bit integers is
 * needed, prefer svtkTypeInt32Array to this class.  If an array of 64 bit
 * integers is needed, prefer svtkTypeInt64Array to this class.
 */

#ifndef svtkLongArray_h
#define svtkLongArray_h

#include "svtkAOSDataArrayTemplate.h" // Real Superclass
#include "svtkCommonCoreModule.h"     // For export macro
#include "svtkDataArray.h"

// Fake the superclass for the wrappers.
#ifndef __SVTK_WRAP__
#define svtkDataArray svtkAOSDataArrayTemplate<long>
#endif
class SVTKCOMMONCORE_EXPORT svtkLongArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkLongArray, svtkDataArray);
#ifndef __SVTK_WRAP__
#undef svtkDataArray
#endif
  static svtkLongArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // This macro expands to the set of method declarations that
  // make up the interface of svtkAOSDataArrayTemplate, which is ignored
  // by the wrappers.
#if defined(__SVTK_WRAP__) || defined(__WRAP_GCCXML__)
  svtkCreateWrappedArrayInterface(long);
#endif

  /**
   * A faster alternative to SafeDownCast for downcasting svtkAbstractArrays.
   */
  static svtkLongArray* FastDownCast(svtkAbstractArray* source)
  {
    return static_cast<svtkLongArray*>(Superclass::FastDownCast(source));
  }

  /**
   * Get the minimum data value in its native type.
   */
  static long GetDataTypeValueMin() { return SVTK_LONG_MIN; }

  /**
   * Get the maximum data value in its native type.
   */
  static long GetDataTypeValueMax() { return SVTK_LONG_MAX; }

protected:
  svtkLongArray();
  ~svtkLongArray() override;

private:
  typedef svtkAOSDataArrayTemplate<long> RealSuperclass;

  svtkLongArray(const svtkLongArray&) = delete;
  void operator=(const svtkLongArray&) = delete;
};

// Define svtkArrayDownCast implementation:
svtkArrayDownCast_FastCastMacro(svtkLongArray);

#endif
