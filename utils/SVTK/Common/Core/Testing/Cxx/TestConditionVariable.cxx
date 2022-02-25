#include "svtkConditionVariable.h"
#include "svtkMultiThreader.h"
#include "svtksys/SystemTools.hxx"

#include <atomic>
#include <cstdlib>

typedef struct
{
  svtkMutexLock* Lock;
  svtkConditionVariable* Condition;
  std::atomic<int32_t> Done;
  int NumberOfWorkers;
} svtkThreadUserData;

SVTK_THREAD_RETURN_TYPE svtkTestCondVarThread(void* arg)
{
  int threadId = static_cast<svtkMultiThreader::ThreadInfo*>(arg)->ThreadID;
  int threadCount = static_cast<svtkMultiThreader::ThreadInfo*>(arg)->NumberOfThreads;
  svtkThreadUserData* td =
    static_cast<svtkThreadUserData*>(static_cast<svtkMultiThreader::ThreadInfo*>(arg)->UserData);
  if (td)
  {
    if (threadId == 0)
    {
      td->Lock->Lock();
      td->Done = 0;
      cout << "Thread " << (threadId + 1) << " of " << threadCount << " initializing.\n";
      cout.flush();
      td->Lock->Unlock();

      int i;
      for (i = 0; i < 2 * threadCount; ++i)
      {
        td->Lock->Lock();
        cout << "Signaling (count " << i << ")...\n";
        cout.flush();
        td->Lock->Unlock();
        td->Condition->Signal();

        // sleep( 1 );
      }

      i = 0;
      int currNumWorkers = 0;
      do
      {
        td->Lock->Lock();
        td->Done = 1;
        cout << "Broadcasting...\n";
        cout.flush();
        currNumWorkers = td->NumberOfWorkers;
        td->Lock->Unlock();
        td->Condition->Broadcast();
        svtksys::SystemTools::Delay(200); // 0.2 s between broadcasts
      } while (currNumWorkers > 0 && (i++ < 1000));
      if (i >= 1000)
      {
        exit(2);
      }
    }
    else
    {
      // Wait for thread 0 to initialize... Ugly but effective
      bool done = false;
      do
      {
        td->Lock->Lock();
        if (td->Done)
        {
          done = true;
          td->Lock->Unlock();
        }
        else
        {
          td->Lock->Unlock();
          svtksys::SystemTools::Delay(200); // 0.2 s between checking
        }
      } while (!done);

      // Wait for the condition and then note we were signaled.
      // This part looks like a Hansen Monitor:
      // ref: http://www.cs.utexas.edu/users/lorenzo/corsi/cs372h/07S/notes/Lecture12.pdf (page
      // 2/5), code on Tradeoff slide.

      td->Lock->Lock();
      while (td->Done <= 0)
      {
        cout << " Thread " << (threadId + 1) << " waiting.\n";
        cout.flush();
        // Wait() performs an Unlock internally.
        td->Condition->Wait(td->Lock);
        // Once Wait() returns, the lock is locked again.
        cout << " Thread " << (threadId + 1) << " responded.\n";
        cout.flush();
      }
      --td->NumberOfWorkers;
      td->Lock->Unlock();
    }

    td->Lock->Lock();
    cout << "  Thread " << (threadId + 1) << " of " << threadCount << " exiting.\n";
    cout.flush();
    td->Lock->Unlock();
  }
  else
  {
    cout << "No thread data!\n";
    cout << "  Thread " << (threadId + 1) << " of " << threadCount << " exiting.\n";
    cout.flush();
  }

  return SVTK_THREAD_RETURN_VALUE;
}

int TestConditionVariable(int, char*[])
{
  svtkMultiThreader* threader = svtkMultiThreader::New();
  int numThreads = threader->GetNumberOfThreads();

  svtkThreadUserData data;
  data.Lock = svtkMutexLock::New();
  data.Condition = svtkConditionVariable::New();
  data.Done = -1;
  data.NumberOfWorkers = numThreads - 1;

  threader->SetNumberOfThreads(numThreads);
  threader->SetSingleMethod(svtkTestCondVarThread, &data);
  threader->SingleMethodExecute();

  cout << "Done with threader.\n";
  cout.flush();

  svtkIndent indent;
  indent = indent.GetNextIndent();
  data.Condition->PrintSelf(cout, indent);

  data.Lock->Delete();
  data.Condition->Delete();
  threader->Delete();
  return 0;
}
