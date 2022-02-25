/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMultiThreader.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkMultiThreader.h"

#include "svtkObjectFactory.h"
#include "svtkWindows.h"

svtkStandardNewMacro(svtkMultiThreader);

// Need to define "svtkExternCThreadFunctionType" to avoid warning on some
// platforms about passing function pointer to an argument expecting an
// extern "C" function.  Placing the typedef of the function pointer type
// inside an extern "C" block solves this problem.
#if defined(SVTK_USE_PTHREADS)
#include <pthread.h>
extern "C"
{
  typedef void* (*svtkExternCThreadFunctionType)(void*);
}
#else
typedef svtkThreadFunctionType svtkExternCThreadFunctionType;
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

// Initialize static member that controls global maximum number of threads
static int svtkMultiThreaderGlobalMaximumNumberOfThreads = 0;

void svtkMultiThreader::SetGlobalMaximumNumberOfThreads(int val)
{
  if (val == svtkMultiThreaderGlobalMaximumNumberOfThreads)
  {
    return;
  }
  svtkMultiThreaderGlobalMaximumNumberOfThreads = val;
}

int svtkMultiThreader::GetGlobalMaximumNumberOfThreads()
{
  return svtkMultiThreaderGlobalMaximumNumberOfThreads;
}

// 0 => Not initialized.
static int svtkMultiThreaderGlobalDefaultNumberOfThreads = 0;

void svtkMultiThreader::SetGlobalDefaultNumberOfThreads(int val)
{
  if (val == svtkMultiThreaderGlobalDefaultNumberOfThreads)
  {
    return;
  }
  svtkMultiThreaderGlobalDefaultNumberOfThreads = val;
}

int svtkMultiThreader::GetGlobalDefaultNumberOfThreads()
{
  if (svtkMultiThreaderGlobalDefaultNumberOfThreads == 0)
  {
    int num = 1; // default is 1

#ifdef SVTK_USE_PTHREADS
    // Default the number of threads to be the number of available
    // processors if we are using pthreads()
#ifdef _SC_NPROCESSORS_ONLN
    num = sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(_SC_NPROC_ONLN)
    num = sysconf(_SC_NPROC_ONLN);
#endif
#endif

#ifdef __APPLE__
    // Determine the number of CPU cores.
    // hw.logicalcpu takes into account cores/CPUs that are
    // disabled because of power management.
    size_t dataLen = sizeof(int); // 'num' is an 'int'
    int result = sysctlbyname("hw.logicalcpu", &num, &dataLen, nullptr, 0);
    if (result == -1)
    {
      num = 1;
    }
#endif

#ifdef _WIN32
    {
      SYSTEM_INFO sysInfo;
      GetSystemInfo(&sysInfo);
      num = sysInfo.dwNumberOfProcessors;
    }
#endif

#ifndef SVTK_USE_WIN32_THREADS
#ifndef SVTK_USE_PTHREADS
    // If we are not multithreading, the number of threads should
    // always be 1
    num = 1;
#endif
#endif

    // Lets limit the number of threads to SVTK_MAX_THREADS
    if (num > SVTK_MAX_THREADS)
    {
      num = SVTK_MAX_THREADS;
    }

    svtkMultiThreaderGlobalDefaultNumberOfThreads = num;
  }

  return svtkMultiThreaderGlobalDefaultNumberOfThreads;
}

// Constructor. Default all the methods to nullptr. Since the
// ThreadInfoArray is static, the ThreadIDs can be initialized here
// and will not change.
svtkMultiThreader::svtkMultiThreader()
{
  for (int i = 0; i < SVTK_MAX_THREADS; i++)
  {
    this->ThreadInfoArray[i].ThreadID = i;
    this->ThreadInfoArray[i].ActiveFlag = nullptr;
    this->ThreadInfoArray[i].ActiveFlagLock = nullptr;
    this->MultipleMethod[i] = nullptr;
    this->SpawnedThreadActiveFlag[i] = 0;
    this->SpawnedThreadActiveFlagLock[i] = nullptr;
    this->SpawnedThreadInfoArray[i].ThreadID = i;
  }

  this->SingleMethod = nullptr;
  this->NumberOfThreads = svtkMultiThreader::GetGlobalDefaultNumberOfThreads();
}

svtkMultiThreader::~svtkMultiThreader()
{
  for (int i = 0; i < SVTK_MAX_THREADS; i++)
  {
    delete this->ThreadInfoArray[i].ActiveFlagLock;
    delete this->SpawnedThreadActiveFlagLock[i];
  }
}

//----------------------------------------------------------------------------
int svtkMultiThreader::GetNumberOfThreads()
{
  int num = this->NumberOfThreads;
  if (svtkMultiThreaderGlobalMaximumNumberOfThreads > 0 &&
    num > svtkMultiThreaderGlobalMaximumNumberOfThreads)
  {
    num = svtkMultiThreaderGlobalMaximumNumberOfThreads;
  }
  return num;
}

// Set the user defined method that will be run on NumberOfThreads threads
// when SingleMethodExecute is called.
void svtkMultiThreader::SetSingleMethod(svtkThreadFunctionType f, void* data)
{
  this->SingleMethod = f;
  this->SingleData = data;
}

// Set one of the user defined methods that will be run on NumberOfThreads
// threads when MultipleMethodExecute is called. This method should be
// called with index = 0, 1, ..,  NumberOfThreads-1 to set up all the
// required user defined methods
void svtkMultiThreader::SetMultipleMethod(int index, svtkThreadFunctionType f, void* data)
{
  // You can only set the method for 0 through NumberOfThreads-1
  if (index >= this->NumberOfThreads)
  {
    svtkErrorMacro(<< "Can't set method " << index << " with a thread count of "
                  << this->NumberOfThreads);
  }
  else
  {
    this->MultipleMethod[index] = f;
    this->MultipleData[index] = data;
  }
}

// Execute the method set as the SingleMethod on NumberOfThreads threads.
void svtkMultiThreader::SingleMethodExecute()
{
  int thread_loop = 0;

#ifdef SVTK_USE_WIN32_THREADS
  DWORD threadId;
  HANDLE process_id[SVTK_MAX_THREADS] = {};
#endif

#ifdef SVTK_USE_PTHREADS
  pthread_t process_id[SVTK_MAX_THREADS] = {};
#endif

  if (!this->SingleMethod)
  {
    svtkErrorMacro(<< "No single method set!");
    return;
  }

  // obey the global maximum number of threads limit
  if (svtkMultiThreaderGlobalMaximumNumberOfThreads &&
    this->NumberOfThreads > svtkMultiThreaderGlobalMaximumNumberOfThreads)
  {
    this->NumberOfThreads = svtkMultiThreaderGlobalMaximumNumberOfThreads;
  }

#ifdef SVTK_USE_WIN32_THREADS
  // Using CreateThread on Windows
  //
  // We want to use CreateThread to start this->NumberOfThreads - 1
  // additional threads which will be used to call this->SingleMethod().
  // The parent thread will also call this routine.  When it is done,
  // it will wait for all the children to finish.
  //
  // First, start up the this->NumberOfThreads-1 processes.  Keep track
  // of their process ids for use later in the waitid call
  for (thread_loop = 1; thread_loop < this->NumberOfThreads; thread_loop++)
  {
    this->ThreadInfoArray[thread_loop].UserData = this->SingleData;
    this->ThreadInfoArray[thread_loop].NumberOfThreads = this->NumberOfThreads;
    process_id[thread_loop] = CreateThread(
      nullptr, 0, this->SingleMethod, ((void*)(&this->ThreadInfoArray[thread_loop])), 0, &threadId);
    if (process_id[thread_loop] == nullptr)
    {
      svtkErrorMacro("Error in thread creation !!!");
    }
  }

  // Now, the parent thread calls this->SingleMethod() itself
  this->ThreadInfoArray[0].UserData = this->SingleData;
  this->ThreadInfoArray[0].NumberOfThreads = this->NumberOfThreads;
  this->SingleMethod((void*)(&this->ThreadInfoArray[0]));

  // The parent thread has finished this->SingleMethod() - so now it
  // waits for each of the other processes to exit
  for (thread_loop = 1; thread_loop < this->NumberOfThreads; thread_loop++)
  {
    WaitForSingleObject(process_id[thread_loop], INFINITE);
  }

  // close the threads
  for (thread_loop = 1; thread_loop < this->NumberOfThreads; thread_loop++)
  {
    CloseHandle(process_id[thread_loop]);
  }
#endif

#ifdef SVTK_USE_PTHREADS
  // Using POSIX threads
  //
  // We want to use pthread_create to start this->NumberOfThreads-1 additional
  // threads which will be used to call this->SingleMethod(). The
  // parent thread will also call this routine.  When it is done,
  // it will wait for all the children to finish.
  //
  // First, start up the this->NumberOfThreads-1 processes.  Keep track
  // of their process ids for use later in the pthread_join call

  pthread_attr_t attr;

  pthread_attr_init(&attr);
#if !defined(__CYGWIN__)
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_PROCESS);
#endif

  for (thread_loop = 1; thread_loop < this->NumberOfThreads; thread_loop++)
  {
    this->ThreadInfoArray[thread_loop].UserData = this->SingleData;
    this->ThreadInfoArray[thread_loop].NumberOfThreads = this->NumberOfThreads;

    int threadError = pthread_create(&(process_id[thread_loop]), &attr,
      reinterpret_cast<svtkExternCThreadFunctionType>(this->SingleMethod),
      ((void*)(&this->ThreadInfoArray[thread_loop])));
    if (threadError != 0)
    {
      svtkErrorMacro(<< "Unable to create a thread.  pthread_create() returned " << threadError);
    }
  }

  // Now, the parent thread calls this->SingleMethod() itself
  this->ThreadInfoArray[0].UserData = this->SingleData;
  this->ThreadInfoArray[0].NumberOfThreads = this->NumberOfThreads;
  this->SingleMethod((void*)(&this->ThreadInfoArray[0]));

  // The parent thread has finished this->SingleMethod() - so now it
  // waits for each of the other processes to exit
  for (thread_loop = 1; thread_loop < this->NumberOfThreads; thread_loop++)
  {
    pthread_join(process_id[thread_loop], nullptr);
  }
#endif

#ifndef SVTK_USE_WIN32_THREADS
#ifndef SVTK_USE_PTHREADS
  // There is no multi threading, so there is only one thread.
  this->ThreadInfoArray[0].UserData = this->SingleData;
  this->ThreadInfoArray[0].NumberOfThreads = this->NumberOfThreads;
  this->SingleMethod((void*)(&this->ThreadInfoArray[0]));
#endif
#endif
}

void svtkMultiThreader::MultipleMethodExecute()
{
  int thread_loop;

#ifdef SVTK_USE_WIN32_THREADS
  DWORD threadId;
  HANDLE process_id[SVTK_MAX_THREADS] = {};
#endif

#ifdef SVTK_USE_PTHREADS
  pthread_t process_id[SVTK_MAX_THREADS] = {};
#endif

  // obey the global maximum number of threads limit
  if (svtkMultiThreaderGlobalMaximumNumberOfThreads &&
    this->NumberOfThreads > svtkMultiThreaderGlobalMaximumNumberOfThreads)
  {
    this->NumberOfThreads = svtkMultiThreaderGlobalMaximumNumberOfThreads;
  }

  for (thread_loop = 0; thread_loop < this->NumberOfThreads; thread_loop++)
  {
    if (this->MultipleMethod[thread_loop] == (svtkThreadFunctionType)nullptr)
    {
      svtkErrorMacro(<< "No multiple method set for: " << thread_loop);
      return;
    }
  }

#ifdef SVTK_USE_WIN32_THREADS
  // Using CreateThread on Windows
  //
  // We want to use CreateThread to start this->NumberOfThreads - 1
  // additional threads which will be used to call the NumberOfThreads-1
  // methods defined in this->MultipleMethods[](). The parent thread
  // will call this->MultipleMethods[NumberOfThreads-1]().  When it is done,
  // it will wait for all the children to finish.
  //
  // First, start up the this->NumberOfThreads-1 processes.  Keep track
  // of their process ids for use later in the waitid call
  for (thread_loop = 1; thread_loop < this->NumberOfThreads; thread_loop++)
  {
    this->ThreadInfoArray[thread_loop].UserData = this->MultipleData[thread_loop];
    this->ThreadInfoArray[thread_loop].NumberOfThreads = this->NumberOfThreads;
    process_id[thread_loop] = CreateThread(nullptr, 0, this->MultipleMethod[thread_loop],
      ((void*)(&this->ThreadInfoArray[thread_loop])), 0, &threadId);
    if (process_id[thread_loop] == nullptr)
    {
      svtkErrorMacro("Error in thread creation !!!");
    }
  }

  // Now, the parent thread calls the last method itself
  this->ThreadInfoArray[0].UserData = this->MultipleData[0];
  this->ThreadInfoArray[0].NumberOfThreads = this->NumberOfThreads;
  (this->MultipleMethod[0])((void*)(&this->ThreadInfoArray[0]));

  // The parent thread has finished its method - so now it
  // waits for each of the other threads to exit
  for (thread_loop = 1; thread_loop < this->NumberOfThreads; thread_loop++)
  {
    WaitForSingleObject(process_id[thread_loop], INFINITE);
  }

  // close the threads
  for (thread_loop = 1; thread_loop < this->NumberOfThreads; thread_loop++)
  {
    CloseHandle(process_id[thread_loop]);
  }
#endif

#ifdef SVTK_USE_PTHREADS
  // Using POSIX threads
  //
  // We want to use pthread_create to start this->NumberOfThreads - 1
  // additional
  // threads which will be used to call the NumberOfThreads-1 methods
  // defined in this->MultipleMethods[](). The parent thread
  // will call this->MultipleMethods[NumberOfThreads-1]().  When it is done,
  // it will wait for all the children to finish.
  //
  // First, start up the this->NumberOfThreads-1 processes.  Keep track
  // of their process ids for use later in the pthread_join call

  pthread_attr_t attr;

  pthread_attr_init(&attr);
#ifndef __CYGWIN__
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_PROCESS);
#endif

  for (thread_loop = 1; thread_loop < this->NumberOfThreads; thread_loop++)
  {
    this->ThreadInfoArray[thread_loop].UserData = this->MultipleData[thread_loop];
    this->ThreadInfoArray[thread_loop].NumberOfThreads = this->NumberOfThreads;
    pthread_create(&(process_id[thread_loop]), &attr,
      reinterpret_cast<svtkExternCThreadFunctionType>(this->MultipleMethod[thread_loop]),
      ((void*)(&this->ThreadInfoArray[thread_loop])));
  }

  // Now, the parent thread calls the last method itself
  this->ThreadInfoArray[0].UserData = this->MultipleData[0];
  this->ThreadInfoArray[0].NumberOfThreads = this->NumberOfThreads;
  (this->MultipleMethod[0])((void*)(&this->ThreadInfoArray[0]));

  // The parent thread has finished its method - so now it
  // waits for each of the other processes to exit
  for (thread_loop = 1; thread_loop < this->NumberOfThreads; thread_loop++)
  {
    pthread_join(process_id[thread_loop], nullptr);
  }
#endif

#ifndef SVTK_USE_WIN32_THREADS
#ifndef SVTK_USE_PTHREADS
  // There is no multi threading, so there is only one thread.
  this->ThreadInfoArray[0].UserData = this->MultipleData[0];
  this->ThreadInfoArray[0].NumberOfThreads = this->NumberOfThreads;
  (this->MultipleMethod[0])((void*)(&this->ThreadInfoArray[0]));
#endif
#endif
}

int svtkMultiThreader::SpawnThread(svtkThreadFunctionType f, void* userdata)
{
  int id;

  for (id = 0; id < SVTK_MAX_THREADS; id++)
  {
    if (this->SpawnedThreadActiveFlagLock[id] == nullptr)
    {
      this->SpawnedThreadActiveFlagLock[id] = new std::mutex;
    }
    std::lock_guard<std::mutex>(*this->SpawnedThreadActiveFlagLock[id]);
    if (this->SpawnedThreadActiveFlag[id] == 0)
    {
      // We've got a usable thread id, so grab it
      this->SpawnedThreadActiveFlag[id] = 1;
      break;
    }
  }

  if (id >= SVTK_MAX_THREADS)
  {
    svtkErrorMacro(<< "You have too many active threads!");
    return -1;
  }

  this->SpawnedThreadInfoArray[id].UserData = userdata;
  this->SpawnedThreadInfoArray[id].NumberOfThreads = 1;
  this->SpawnedThreadInfoArray[id].ActiveFlag = &this->SpawnedThreadActiveFlag[id];
  this->SpawnedThreadInfoArray[id].ActiveFlagLock = this->SpawnedThreadActiveFlagLock[id];

#ifdef SVTK_USE_WIN32_THREADS
  // Using CreateThread on Windows
  //
  DWORD threadId;
  this->SpawnedThreadProcessID[id] =
    CreateThread(nullptr, 0, f, ((void*)(&this->SpawnedThreadInfoArray[id])), 0, &threadId);
  if (this->SpawnedThreadProcessID[id] == nullptr)
  {
    svtkErrorMacro("Error in thread creation !!!");
  }
#endif

#ifdef SVTK_USE_PTHREADS
  // Using POSIX threads
  //
  pthread_attr_t attr;
  pthread_attr_init(&attr);
#ifndef __CYGWIN__
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_PROCESS);
#endif

  pthread_create(&(this->SpawnedThreadProcessID[id]), &attr,
    reinterpret_cast<svtkExternCThreadFunctionType>(f),
    ((void*)(&this->SpawnedThreadInfoArray[id])));

#endif

#ifndef SVTK_USE_WIN32_THREADS
#ifndef SVTK_USE_PTHREADS
  // There is no multi threading, so there is only one thread.
  // This won't work - so give an error message.
  svtkErrorMacro(<< "Cannot spawn thread in a single threaded environment!");
  delete this->SpawnedThreadActiveFlagLock[id];
  id = -1;
#endif
#endif

  return id;
}

void svtkMultiThreader::TerminateThread(int threadID)
{
  // check if the threadID argument is in range
  if (threadID >= SVTK_MAX_THREADS)
  {
    svtkErrorMacro("ThreadID is out of range. Must be less that " << SVTK_MAX_THREADS);
    return;
  }

  // If we don't have a lock, then this thread is definitely not active
  if (!this->SpawnedThreadActiveFlag[threadID])
  {
    return;
  }

  // If we do have a lock, use it and find out the status of the active flag
  int val = 0;
  {
    std::lock_guard<std::mutex>(*this->SpawnedThreadActiveFlagLock[threadID]);
    val = this->SpawnedThreadActiveFlag[threadID];
  }

  // If the active flag is 0, return since this thread is not active
  if (val == 0)
  {
    return;
  }

  // OK - now we know we have an active thread - set the active flag to 0
  // to indicate to the thread that it should terminate itself
  {
    std::lock_guard<std::mutex>(*this->SpawnedThreadActiveFlagLock[threadID]);
    this->SpawnedThreadActiveFlag[threadID] = 0;
  }

#ifdef SVTK_USE_WIN32_THREADS
  WaitForSingleObject(this->SpawnedThreadProcessID[threadID], INFINITE);
  CloseHandle(this->SpawnedThreadProcessID[threadID]);
#endif

#ifdef SVTK_USE_PTHREADS
  pthread_join(this->SpawnedThreadProcessID[threadID], nullptr);
#endif

#ifndef SVTK_USE_WIN32_THREADS
#ifndef SVTK_USE_PTHREADS
  // There is no multi threading, so there is only one thread.
  // This won't work - so give an error message.
  svtkErrorMacro(<< "Cannot terminate thread in single threaded environment!");
#endif
#endif

  delete this->SpawnedThreadActiveFlagLock[threadID];
  this->SpawnedThreadActiveFlagLock[threadID] = nullptr;
}

//----------------------------------------------------------------------------
svtkMultiThreaderIDType svtkMultiThreader::GetCurrentThreadID()
{
#if defined(SVTK_USE_PTHREADS)
  return pthread_self();
#elif defined(SVTK_USE_WIN32_THREADS)
  return GetCurrentThreadId();
#else
  // No threading implementation.  Assume all callers are in the same
  // thread.
  return 0;
#endif
}

svtkTypeBool svtkMultiThreader::IsThreadActive(int threadID)
{
  // check if the threadID argument is in range
  if (threadID >= SVTK_MAX_THREADS)
  {
    svtkErrorMacro("ThreadID is out of range. Must be less that " << SVTK_MAX_THREADS);
    return 0;
  }

  // If we don't have a lock, then this thread is not active
  if (this->SpawnedThreadActiveFlagLock[threadID] == nullptr)
  {
    return 0;
  }

  // We have a lock - use it to get the active flag value
  int val = 0;
  {
    std::lock_guard<std::mutex>(*this->SpawnedThreadActiveFlagLock[threadID]);
    val = this->SpawnedThreadActiveFlag[threadID];
  }

  // now return that value
  return val;
}

//----------------------------------------------------------------------------
svtkTypeBool svtkMultiThreader::ThreadsEqual(svtkMultiThreaderIDType t1, svtkMultiThreaderIDType t2)
{
#if defined(SVTK_USE_PTHREADS)
  return pthread_equal(t1, t2) != 0;
#elif defined(SVTK_USE_WIN32_THREADS)
  return t1 == t2;
#else
  // No threading implementation.  Assume all callers are in the same
  // thread.
  return 1;
#endif
}

// Print method for the multithreader
void svtkMultiThreader::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Thread Count: " << this->NumberOfThreads << "\n";
  os << indent
     << "Global Maximum Number Of Threads: " << svtkMultiThreaderGlobalMaximumNumberOfThreads
     << endl;
  os << "Thread system used: "
     <<
#ifdef SVTK_USE_PTHREADS
    "PTHREADS"
#elif defined SVTK_USE_WIN32_THREADS
    "WIN32 Threads"
#else
    "NO THREADS SUPPORT"
#endif
     << endl;
}
