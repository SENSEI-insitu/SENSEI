/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMutexLock.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkMutexLock.h"
#include "svtkObjectFactory.h"

#ifdef SVTK_USE_WIN32_THREADS
#include "svtkWindows.h"
#endif

svtkStandardNewMacro(svtkMutexLock);

// New for the SimpleMutex
svtkSimpleMutexLock* svtkSimpleMutexLock::New()
{
  return new svtkSimpleMutexLock;
}

// Construct a new svtkMutexLock
svtkSimpleMutexLock::svtkSimpleMutexLock()
{
#ifdef SVTK_USE_WIN32_THREADS
  this->MutexLock = CreateMutex(nullptr, FALSE, nullptr);
#endif

#ifdef SVTK_USE_PTHREADS
  pthread_mutex_init(&(this->MutexLock), nullptr);
#endif
}

// Destruct the svtkMutexVariable
svtkSimpleMutexLock::~svtkSimpleMutexLock()
{
#ifdef SVTK_USE_WIN32_THREADS
  CloseHandle(this->MutexLock);
#endif

#ifdef SVTK_USE_PTHREADS
  pthread_mutex_destroy(&this->MutexLock);
#endif
}

// Lock the svtkMutexLock
void svtkSimpleMutexLock::Lock()
{
#ifdef SVTK_USE_WIN32_THREADS
  WaitForSingleObject(this->MutexLock, INFINITE);
#endif

#ifdef SVTK_USE_PTHREADS
  pthread_mutex_lock(&this->MutexLock);
#endif
}

// Unlock the svtkMutexLock
void svtkSimpleMutexLock::Unlock()
{
#ifdef SVTK_USE_WIN32_THREADS
  ReleaseMutex(this->MutexLock);
#endif

#ifdef SVTK_USE_PTHREADS
  pthread_mutex_unlock(&this->MutexLock);
#endif
}

void svtkMutexLock::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
