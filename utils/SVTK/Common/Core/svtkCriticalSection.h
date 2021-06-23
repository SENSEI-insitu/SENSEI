/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCriticalSection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCriticalSection
 * @brief   Critical section locking class
 *
 * svtkCriticalSection allows the locking of variables which are accessed
 * through different threads.  This header file also defines
 * svtkSimpleCriticalSection which is not a subclass of svtkObject.
 * The API is identical to that of svtkMutexLock, and the behavior is
 * identical as well, except on Windows 9x/NT platforms. The only difference
 * on these platforms is that svtkMutexLock is more flexible, in that
 * it works across processes as well as across threads, but also costs
 * more, in that it evokes a 600-cycle x86 ring transition. The
 * svtkCriticalSection provides a higher-performance equivalent (on
 * Windows) but won't work across processes. Since it is unclear how,
 * in svtk, an object at the svtk level can be shared across processes
 * in the first place, one should use svtkCriticalSection unless one has
 * a very good reason to use svtkMutexLock. If higher-performance equivalents
 * for non-Windows platforms (Irix, SunOS, etc) are discovered, they
 * should replace the implementations in this class
 */

#ifndef svtkCriticalSection_h
#define svtkCriticalSection_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"
#include "svtkSimpleCriticalSection.h" // For simple critical section

class SVTKCOMMONCORE_EXPORT svtkCriticalSection : public svtkObject
{
public:
  static svtkCriticalSection* New();

  svtkTypeMacro(svtkCriticalSection, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Lock the svtkCriticalSection
   */
  void Lock();

  /**
   * Unlock the svtkCriticalSection
   */
  void Unlock();

protected:
  svtkSimpleCriticalSection SimpleCriticalSection;
  svtkCriticalSection() {}
  ~svtkCriticalSection() override {}

private:
  svtkCriticalSection(const svtkCriticalSection&) = delete;
  void operator=(const svtkCriticalSection&) = delete;
};

inline void svtkCriticalSection::Lock()
{
  this->SimpleCriticalSection.Lock();
}

inline void svtkCriticalSection::Unlock()
{
  this->SimpleCriticalSection.Unlock();
}

#endif
