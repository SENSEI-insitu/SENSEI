/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnsignedLongArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkUnsignedLongArray
 * @brief   dynamic, self-adjusting array of unsigned long
 *
 * svtkUnsignedLongArray is an array of values of type unsigned long.
 * It provides methods for insertion and retrieval of values and will
 * automatically resize itself to hold new data.
 *
 * The C++ standard does not define the exact size of the unsigned long type,
 * so use of this type directly is discouraged.  If an array of 32 bit
 * unsigned integers is needed, prefer svtkTypeUInt32Array to this class.
 * If an array of 64 bit unsigned integers is needed, prefer
 * svtkUTypeInt64Array to this class.
 */

#ifndef svtkUnsignedLongArray_h
#define svtkUnsignedLongArray_h

#include "svtkAOSDataArrayTemplate.h" // Real Superclass
#include "svtkCommonCoreModule.h"     // For export macro
#include "svtkDataArray.h"

// Fake the superclass for the wrappers.
#ifndef __SVTK_WRAP__
#define svtkDataArray svtkAOSDataArrayTemplate<unsigned long>
#endif
class SVTKCOMMONCORE_EXPORT svtkUnsignedLongArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkUnsignedLongArray, svtkDataArray);
#ifndef __SVTK_WRAP__
#undef svtkDataArray
#endif
  static svtkUnsignedLongArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // This macro expands to the set of method declarations that
  // make up the interface of svtkAOSDataArrayTemplate, which is ignored
  // by the wrappers.
#if defined(__SVTK_WRAP__) || defined(__WRAP_GCCXML__)
  svtkCreateWrappedArrayInterface(unsigned long);
#endif

  /**
   * A faster alternative to SafeDownCast for downcasting svtkAbstractArrays.
   */
  static svtkUnsignedLongArray* FastDownCast(svtkAbstractArray* source)
  {
    return static_cast<svtkUnsignedLongArray*>(Superclass::FastDownCast(source));
  }

  /**
   * Get the minimum data value in its native type.
   */
  static unsigned long GetDataTypeValueMin() { return SVTK_UNSIGNED_LONG_MIN; }

  /**
   * Get the maximum data value in its native type.
   */
  static unsigned long GetDataTypeValueMax() { return SVTK_UNSIGNED_LONG_MAX; }

protected:
  svtkUnsignedLongArray();
  ~svtkUnsignedLongArray() override;

private:
  typedef svtkAOSDataArrayTemplate<unsigned long> RealSuperclass;

  svtkUnsignedLongArray(const svtkUnsignedLongArray&) = delete;
  void operator=(const svtkUnsignedLongArray&) = delete;
};

// Define svtkArrayDownCast implementation:
svtkArrayDownCast_FastCastMacro(svtkUnsignedLongArray);

#endif
