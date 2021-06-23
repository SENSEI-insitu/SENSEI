/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkIntArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkIntArray
 * @brief   dynamic, self-adjusting array of int
 *
 * svtkIntArray is an array of values of type int.  It provides
 * methods for insertion and retrieval of values and will
 * automatically resize itself to hold new data.
 *
 * The C++ standard does not define the exact size of the int type, so use
 * of this type directly is discouraged.  If an array of 32 bit integers is
 * needed, prefer svtkTypeInt32Array to this class.
 */

#ifndef svtkIntArray_h
#define svtkIntArray_h

#include "svtkAOSDataArrayTemplate.h" // Real Superclass
#include "svtkCommonCoreModule.h"     // For export macro
#include "svtkDataArray.h"

// Fake the superclass for the wrappers.
#ifndef __SVTK_WRAP__
#define svtkDataArray svtkAOSDataArrayTemplate<int>
#endif
class SVTKCOMMONCORE_EXPORT svtkIntArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkIntArray, svtkDataArray);
#ifndef __SVTK_WRAP__
#undef svtkDataArray
#endif
  static svtkIntArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // This macro expands to the set of method declarations that
  // make up the interface of svtkAOSDataArrayTemplate, which is ignored
  // by the wrappers.
#if defined(__SVTK_WRAP__) || defined(__WRAP_GCCXML__)
  svtkCreateWrappedArrayInterface(int);
#endif

  /**
   * A faster alternative to SafeDownCast for downcasting svtkAbstractArrays.
   */
  static svtkIntArray* FastDownCast(svtkAbstractArray* source)
  {
    return static_cast<svtkIntArray*>(Superclass::FastDownCast(source));
  }

  /**
   * Get the minimum data value in its native type.
   */
  static int GetDataTypeValueMin() { return SVTK_INT_MIN; }

  /**
   * Get the maximum data value in its native type.
   */
  static int GetDataTypeValueMax() { return SVTK_INT_MAX; }

protected:
  svtkIntArray();
  ~svtkIntArray() override;

private:
  typedef svtkAOSDataArrayTemplate<int> RealSuperclass;

  svtkIntArray(const svtkIntArray&) = delete;
  void operator=(const svtkIntArray&) = delete;
};

// Define svtkArrayDownCast implementation:
svtkArrayDownCast_FastCastMacro(svtkIntArray);

#endif
