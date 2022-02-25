/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkShortArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkShortArray
 * @brief   dynamic, self-adjusting array of short
 *
 * svtkShortArray is an array of values of type short.  It provides
 * methods for insertion and retrieval of values and will
 * automatically resize itself to hold new data.
 *
 * The C++ standard does not define the exact size of the short type,
 * so use of this type directly is discouraged.  If an array of 16 bit
 * integers is needed, prefer svtkTypeInt16Array to this class.
 */

#ifndef svtkShortArray_h
#define svtkShortArray_h

#include "svtkAOSDataArrayTemplate.h" // Real Superclass
#include "svtkCommonCoreModule.h"     // For export macro
#include "svtkDataArray.h"

// Fake the superclass for the wrappers.
#ifndef __SVTK_WRAP__
#define svtkDataArray svtkAOSDataArrayTemplate<short>
#endif
class SVTKCOMMONCORE_EXPORT svtkShortArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkShortArray, svtkDataArray);
#ifndef __SVTK_WRAP__
#undef svtkDataArray
#endif
  static svtkShortArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // This macro expands to the set of method declarations that
  // make up the interface of svtkAOSDataArrayTemplate, which is ignored
  // by the wrappers.
#if defined(__SVTK_WRAP__) || defined(__WRAP_GCCXML__)
  svtkCreateWrappedArrayInterface(short);
#endif

  /**
   * A faster alternative to SafeDownCast for downcasting svtkAbstractArrays.
   */
  static svtkShortArray* FastDownCast(svtkAbstractArray* source)
  {
    return static_cast<svtkShortArray*>(Superclass::FastDownCast(source));
  }

  /**
   * Get the minimum data value in its native type.
   */
  static short GetDataTypeValueMin() { return SVTK_SHORT_MIN; }

  /**
   * Get the maximum data value in its native type.
   */
  static short GetDataTypeValueMax() { return SVTK_SHORT_MAX; }

protected:
  svtkShortArray();
  ~svtkShortArray() override;

private:
  typedef svtkAOSDataArrayTemplate<short> RealSuperclass;

  svtkShortArray(const svtkShortArray&) = delete;
  void operator=(const svtkShortArray&) = delete;
};

// Define svtkArrayDownCast implementation:
svtkArrayDownCast_FastCastMacro(svtkShortArray);

#endif
