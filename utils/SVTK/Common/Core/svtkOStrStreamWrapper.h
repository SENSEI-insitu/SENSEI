/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkOStrStreamWrapper.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkOStrStreamWrapper
 * @brief   Wrapper for ostrstream.  Internal SVTK use only.
 *
 * Provides a wrapper around the C++ ostrstream class so that SVTK
 * source files need not include the full C++ streams library.  This
 * is intended to prevent cluttering of the translation unit and speed
 * up compilation.  Experimentation has revealed between 10% and 60%
 * less time for compilation depending on the platform.  This wrapper
 * is used by the macros in svtkSetGet.h.
 */

#ifndef svtkOStrStreamWrapper_h
#define svtkOStrStreamWrapper_h

#ifndef SVTK_SYSTEM_INCLUDES_INSIDE
Do_not_include_svtkOStrStreamWrapper_directly_svtkSystemIncludes_includes_it;
#endif

class SVTKCOMMONCORE_EXPORT svtkOStrStreamWrapper : public svtkOStreamWrapper
{
public:
  /**
   * Constructor.
   */
  svtkOStrStreamWrapper();

  /**
   * Destructor frees all used memory.
   */
  ~svtkOStrStreamWrapper() override;

  /**
   * Get the string that has been written.  This call transfers
   * ownership of the returned memory to the caller.  Call
   * rdbuf()->freeze(0) to return ownership to the svtkOStrStreamWrapper.
   */
  char* str();

  /**
   * Returns a pointer to this class.  This is a hack so that the old
   * ostrstream's s.rdbuf()->freeze(0) can work.
   */
  svtkOStrStreamWrapper* rdbuf();

  //@{
  /**
   * Set whether the memory is frozen.  The svtkOStrStreamWrapper will free
   * the memory returned by str() only if it is not frozen.
   */
  void freeze();
  void freeze(int);
  //@}

protected:
  // The pointer returned by str().
  char* Result;

  // Whether the caller of str() owns the memory.
  int Frozen;

private:
  svtkOStrStreamWrapper(const svtkOStrStreamWrapper& r) = delete;
  svtkOStrStreamWrapper& operator=(const svtkOStrStreamWrapper&) = delete;
};

#endif
// SVTK-HeaderTest-Exclude: svtkOStrStreamWrapper.h
