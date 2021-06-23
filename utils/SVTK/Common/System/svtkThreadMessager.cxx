/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkThreadMessager.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkThreadMessager.h"

#include "svtkObjectFactory.h"

#ifdef SVTK_USE_WIN32_THREADS
#include "svtkWindows.h"
#endif

svtkStandardNewMacro(svtkThreadMessager);

svtkThreadMessager::svtkThreadMessager()
{
#ifdef SVTK_USE_WIN32_THREADS
  this->WSignal = CreateEvent(0, FALSE, FALSE, 0);
#elif defined(SVTK_USE_PTHREADS)
  pthread_cond_init(&this->PSignal, nullptr);
  pthread_mutex_init(&this->Mutex, nullptr);
  pthread_mutex_lock(&this->Mutex);
#endif
}

svtkThreadMessager::~svtkThreadMessager()
{
#ifdef SVTK_USE_WIN32_THREADS
  CloseHandle(this->WSignal);
#elif defined(SVTK_USE_PTHREADS)
  pthread_mutex_unlock(&this->Mutex);
  pthread_mutex_destroy(&this->Mutex);
  pthread_cond_destroy(&this->PSignal);
#endif
}

void svtkThreadMessager::WaitForMessage()
{
#ifdef SVTK_USE_WIN32_THREADS
  WaitForSingleObject(this->WSignal, INFINITE);
#elif defined(SVTK_USE_PTHREADS)
  pthread_cond_wait(&this->PSignal, &this->Mutex);
#endif
}

//----------------------------------------------------------------------------
void svtkThreadMessager::SendWakeMessage()
{
#ifdef SVTK_USE_WIN32_THREADS
  SetEvent(this->WSignal);
#elif defined(SVTK_USE_PTHREADS)
  pthread_cond_broadcast(&this->PSignal);
#endif
}

void svtkThreadMessager::EnableWaitForReceiver()
{
#if defined(SVTK_USE_PTHREADS)
  pthread_mutex_lock(&this->Mutex);
#endif
}

void svtkThreadMessager::WaitForReceiver()
{
#if defined(SVTK_USE_PTHREADS)
  pthread_mutex_lock(&this->Mutex);
#endif
}

void svtkThreadMessager::DisableWaitForReceiver()
{
#if defined(SVTK_USE_PTHREADS)
  pthread_mutex_unlock(&this->Mutex);
#endif
}

void svtkThreadMessager::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
