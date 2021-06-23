/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMultiThreader.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkMultiThreader
 * @brief   A class for performing multithreaded execution
 *
 * svtkMultithreader is a class that provides support for multithreaded
 * execution using pthreads on POSIX systems, or Win32 threads on
 * Windows.  This class can be used to execute a single
 * method on multiple threads, or to specify a method per thread.
 */

#ifndef svtkMultiThreader_h
#define svtkMultiThreader_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

#include <mutex> // For std::mutex

#if defined(SVTK_USE_PTHREADS)
#include <pthread.h>   // Needed for PTHREAD implementation of mutex
#include <sys/types.h> // Needed for unix implementation of pthreads
#include <unistd.h>    // Needed for unix implementation of pthreads
#endif

// If SVTK_USE_PTHREADS is defined, then pthread_create() will be
// used to create multiple threads

// Defined in svtkSystemIncludes.h:
//   SVTK_MAX_THREADS

// If SVTK_USE_PTHREADS is defined, then the multithreaded
// function is of type void *, and returns nullptr
// Otherwise the type is void which is correct for WIN32

// Defined in svtkSystemIncludes.h:
//   SVTK_THREAD_RETURN_VALUE
//   SVTK_THREAD_RETURN_TYPE

#ifdef SVTK_USE_PTHREADS
typedef void* (*svtkThreadFunctionType)(void*);
typedef pthread_t svtkThreadProcessIDType;
// #define SVTK_THREAD_RETURN_VALUE  nullptr
// #define SVTK_THREAD_RETURN_TYPE   void *
typedef pthread_t svtkMultiThreaderIDType;
#endif

#ifdef SVTK_USE_WIN32_THREADS
typedef svtkWindowsLPTHREAD_START_ROUTINE svtkThreadFunctionType;
typedef svtkWindowsHANDLE svtkThreadProcessIDType;
// #define SVTK_THREAD_RETURN_VALUE 0
// #define SVTK_THREAD_RETURN_TYPE DWORD __stdcall
typedef svtkWindowsDWORD svtkMultiThreaderIDType;
#endif

#if !defined(SVTK_USE_PTHREADS) && !defined(SVTK_USE_WIN32_THREADS)
typedef void (*svtkThreadFunctionType)(void*);
typedef int svtkThreadProcessIDType;
// #define SVTK_THREAD_RETURN_VALUE
// #define SVTK_THREAD_RETURN_TYPE void
typedef int svtkMultiThreaderIDType;
#endif

class SVTKCOMMONCORE_EXPORT svtkMultiThreader : public svtkObject
{
public:
  static svtkMultiThreader* New();

  svtkTypeMacro(svtkMultiThreader, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * This is the structure that is passed to the thread that is
   * created from the SingleMethodExecute, MultipleMethodExecute or
   * the SpawnThread method. It is passed in as a void *, and it is
   * up to the method to cast correctly and extract the information.
   * The ThreadID is a number between 0 and NumberOfThreads-1 that indicates
   * the id of this thread. The NumberOfThreads is this->NumberOfThreads for
   * threads created from SingleMethodExecute or MultipleMethodExecute,
   * and it is 1 for threads created from SpawnThread.
   * The UserData is the (void *)arg passed into the SetSingleMethod,
   * SetMultipleMethod, or SpawnThread method.
   */
  class ThreadInfo
  {
  public:
    int ThreadID;
    int NumberOfThreads;
    int* ActiveFlag;
    std::mutex* ActiveFlagLock;
    void* UserData;
  };

  //@{
  /**
   * Get/Set the number of threads to create. It will be clamped to the range
   * 1 - SVTK_MAX_THREADS, so the caller of this method should check that the
   * requested number of threads was accepted.
   */
  svtkSetClampMacro(NumberOfThreads, int, 1, SVTK_MAX_THREADS);
  virtual int GetNumberOfThreads();
  //@}

  //@{
  /**
   * Set/Get the maximum number of threads to use when multithreading.
   * This limits and overrides any other settings for multithreading.
   * A value of zero indicates no limit.
   */
  static void SetGlobalMaximumNumberOfThreads(int val);
  static int GetGlobalMaximumNumberOfThreads();
  //@}

  //@{
  /**
   * Set/Get the value which is used to initialize the NumberOfThreads
   * in the constructor.  Initially this default is set to the number of
   * processors or SVTK_MAX_THREADS (which ever is less).
   */
  static void SetGlobalDefaultNumberOfThreads(int val);
  static int GetGlobalDefaultNumberOfThreads();
  //@}

  // These methods are excluded from wrapping 1) because the
  // wrapper gives up on them and 2) because they really shouldn't be
  // called from a script anyway.

  /**
   * Execute the SingleMethod (as define by SetSingleMethod) using
   * this->NumberOfThreads threads.
   */
  void SingleMethodExecute();

  /**
   * Execute the MultipleMethods (as define by calling SetMultipleMethod
   * for each of the required this->NumberOfThreads methods) using
   * this->NumberOfThreads threads.
   */
  void MultipleMethodExecute();

  /**
   * Set the SingleMethod to f() and the UserData field of the
   * ThreadInfo that is passed to it will be data.
   * This method (and all the methods passed to SetMultipleMethod)
   * must be of type svtkThreadFunctionType and must take a single argument of
   * type void *.
   */
  void SetSingleMethod(svtkThreadFunctionType, void* data);

  /**
   * Set the MultipleMethod at the given index to f() and the UserData
   * field of the ThreadInfo that is passed to it will be data.
   */
  void SetMultipleMethod(int index, svtkThreadFunctionType, void* data);

  /**
   * Create a new thread for the given function. Return a thread id
   * which is a number between 0 and SVTK_MAX_THREADS - 1. This id should
   * be used to kill the thread at a later time.
   */
  int SpawnThread(svtkThreadFunctionType, void* data);

  /**
   * Terminate the thread that was created with a SpawnThreadExecute()
   */
  void TerminateThread(int thread_id);

  /**
   * Determine if a thread is still active
   */
  svtkTypeBool IsThreadActive(int threadID);

  /**
   * Get the thread identifier of the calling thread.
   */
  static svtkMultiThreaderIDType GetCurrentThreadID();

  /**
   * Check whether two thread identifiers refer to the same thread.
   */
  static svtkTypeBool ThreadsEqual(svtkMultiThreaderIDType t1, svtkMultiThreaderIDType t2);

protected:
  svtkMultiThreader();
  ~svtkMultiThreader() override;

  // The number of threads to use
  int NumberOfThreads;

  // An array of thread info containing a thread id
  // (0, 1, 2, .. SVTK_MAX_THREADS-1), the thread count, and a pointer
  // to void so that user data can be passed to each thread
  ThreadInfo ThreadInfoArray[SVTK_MAX_THREADS];

  // The methods
  svtkThreadFunctionType SingleMethod;
  svtkThreadFunctionType MultipleMethod[SVTK_MAX_THREADS];

  // Storage of MutexFunctions and ints used to control spawned
  // threads and the spawned thread ids
  int SpawnedThreadActiveFlag[SVTK_MAX_THREADS];
  std::mutex* SpawnedThreadActiveFlagLock[SVTK_MAX_THREADS];
  svtkThreadProcessIDType SpawnedThreadProcessID[SVTK_MAX_THREADS];
  ThreadInfo SpawnedThreadInfoArray[SVTK_MAX_THREADS];

  // Internal storage of the data
  void* SingleData;
  void* MultipleData[SVTK_MAX_THREADS];

private:
  svtkMultiThreader(const svtkMultiThreader&) = delete;
  void operator=(const svtkMultiThreader&) = delete;
};

using ThreadInfoStruct = svtkMultiThreader::ThreadInfo;

#endif
