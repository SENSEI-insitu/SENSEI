/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCriticalSection.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkSimpleCriticalSection.h"

void svtkSimpleCriticalSection::Init()
{
#ifdef SVTK_USE_WIN32_THREADS
  // this->MutexLock = CreateMutex( nullptr, FALSE, nullptr );
  InitializeCriticalSection(&this->CritSec);
#endif

#ifdef SVTK_USE_PTHREADS
  pthread_mutex_init(&(this->CritSec), nullptr);
#endif
}

// Destruct the svtkMutexVariable
svtkSimpleCriticalSection::~svtkSimpleCriticalSection()
{
#ifdef SVTK_USE_WIN32_THREADS
  // CloseHandle(this->MutexLock);
  DeleteCriticalSection(&this->CritSec);
#endif

#ifdef SVTK_USE_PTHREADS
  pthread_mutex_destroy(&this->CritSec);
#endif
}

// Lock the svtkCriticalSection
void svtkSimpleCriticalSection::Lock()
{
#ifdef SVTK_USE_WIN32_THREADS
  // WaitForSingleObject( this->MutexLock, INFINITE );
  EnterCriticalSection(&this->CritSec);
#endif

#ifdef SVTK_USE_PTHREADS
  pthread_mutex_lock(&this->CritSec);
#endif
}

// Unlock the svtkCriticalSection
void svtkSimpleCriticalSection::Unlock()
{
#ifdef SVTK_USE_WIN32_THREADS
  // ReleaseMutex( this->MutexLock );
  LeaveCriticalSection(&this->CritSec);
#endif

#ifdef SVTK_USE_PTHREADS
  pthread_mutex_unlock(&this->CritSec);
#endif
}
