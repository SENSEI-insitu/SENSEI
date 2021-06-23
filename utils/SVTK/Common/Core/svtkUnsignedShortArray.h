/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnsignedShortArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkUnsignedShortArray
 * @brief   dynamic, self-adjusting array of unsigned short
 *
 * svtkUnsignedShortArray is an array of values of type unsigned short.
 * It provides methods for insertion and retrieval of values and will
 * automatically resize itself to hold new data.
 *
 * The C++ standard does not define the exact size of the unsigned short type,
 * so use of this type directly is discouraged.  If an array of 16 bit
 * unsigned integers is needed, prefer svtkTypeUInt16Array to this class.
 */

#ifndef svtkUnsignedShortArray_h
#define svtkUnsignedShortArray_h

#include "svtkAOSDataArrayTemplate.h" // Real Superclass
#include "svtkCommonCoreModule.h"     // For export macro
#include "svtkDataArray.h"

// Fake the superclass for the wrappers.
#ifndef __SVTK_WRAP__
#define svtkDataArray svtkAOSDataArrayTemplate<unsigned short>
#endif
class SVTKCOMMONCORE_EXPORT svtkUnsignedShortArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkUnsignedShortArray, svtkDataArray);
#ifndef __SVTK_WRAP__
#undef svtkDataArray
#endif
  static svtkUnsignedShortArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // This macro expands to the set of method declarations that
  // make up the interface of svtkAOSDataArrayTemplate, which is ignored
  // by the wrappers.
#if defined(__SVTK_WRAP__) || defined(__WRAP_GCCXML__)
  svtkCreateWrappedArrayInterface(unsigned short);
#endif

  /**
   * A faster alternative to SafeDownCast for downcasting svtkAbstractArrays.
   */
  static svtkUnsignedShortArray* FastDownCast(svtkAbstractArray* source)
  {
    return static_cast<svtkUnsignedShortArray*>(Superclass::FastDownCast(source));
  }

  /**
   * Get the minimum data value in its native type.
   */
  static unsigned short GetDataTypeValueMin() { return SVTK_UNSIGNED_SHORT_MIN; }

  /**
   * Get the maximum data value in its native type.
   */
  static unsigned short GetDataTypeValueMax() { return SVTK_UNSIGNED_SHORT_MAX; }

protected:
  svtkUnsignedShortArray();
  ~svtkUnsignedShortArray() override;

private:
  typedef svtkAOSDataArrayTemplate<unsigned short> RealSuperclass;

  svtkUnsignedShortArray(const svtkUnsignedShortArray&) = delete;
  void operator=(const svtkUnsignedShortArray&) = delete;
};

// Define svtkArrayDownCast implementation:
svtkArrayDownCast_FastCastMacro(svtkUnsignedShortArray);

#endif
