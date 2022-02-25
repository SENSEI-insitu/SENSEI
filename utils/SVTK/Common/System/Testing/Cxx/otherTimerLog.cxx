/*=========================================================================

  Program:   Visualization Toolkit
  Module:    otherTimerLog.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// .NAME
// .SECTION Description
// this program tests the TimerLog

#include "svtkDebugLeaks.h"
#include "svtkTimerLog.h"

#include <sstream>

// this is needed for the unlink call
#if defined(__CYGWIN__)
#include <sys/unistd.h>
#elif defined(_WIN32)
#include <io.h>
#endif

#include "svtkWindows.h" // for Sleep

void otherTimerLogTest(ostream& strm)
{
  // actual test
  float a = 1.0;
  int i, j;
  strm << "Test svtkTimerLog Start" << endl;
  svtkTimerLog* timer1 = svtkTimerLog::New();

  timer1->SetMaxEntries(8);
  timer1->StartTimer();
  for (j = 0; j < 4; j++)
  {
    timer1->FormatAndMarkEvent("%s%d", "start", j);
    for (i = 0; i < 10000000; i++)
    {
      a *= a;
    }
#ifndef _WIN32
    sleep(1);
#else
    Sleep(1000);
#endif
    timer1->InsertTimedEvent("Timed Event", .00001, 0);
    timer1->FormatAndMarkEvent("%s%d", "end", j);
  }
  timer1->StopTimer();
  strm << *timer1;
  strm << "GetElapsedTime: " << timer1->GetElapsedTime() << endl;
  strm << "GetCPUTime: " << timer1->GetCPUTime() << endl;
  timer1->DumpLog("timing");
  timer1->DumpLogWithIndents(&cerr, 0);
  timer1->ResetLog();
  timer1->CleanupLog();
  unlink("timing");

  cerr << "============== timer separator ================\n";

  timer1->ResetLog();
  timer1->SetMaxEntries(5);

  for (j = 0; j < 4; j++)
  {
    timer1->MarkStartEvent("Other");
    for (i = 0; i < 10000000; i++)
    {
      a *= a;
    }
#ifndef _WIN32
    sleep(1);
#else
    Sleep(1000);
#endif
    timer1->InsertTimedEvent("Other Timed Event", .00001, 0);
    timer1->MarkEndEvent("Other");
  }
  timer1->StopTimer();
  strm << *timer1;
  strm << "GetElapsedTime: " << timer1->GetElapsedTime() << endl;
  strm << "GetCPUTime: " << timer1->GetCPUTime() << endl;
  timer1->DumpLog("timing2");
  timer1->DumpLogWithIndents(&cerr, 0);
  timer1->PrintSelf(cerr, svtkIndent());
  timer1->ResetLog();
  timer1->CleanupLog();
  unlink("timing2");

  timer1->SetMaxEntries(50);

  timer1->Delete();
  strm << "Test svtkTimerLog End" << endl;
}

void timerLogScopeTest()
{
  {
    svtkTimerLogScope timer("Test");
    {
      svtkTimerLogScope timer2("Test2");
#ifndef _WIN32
      sleep(1);
#else
      Sleep(1000);
#endif
    }
#ifndef _WIN32
    sleep(1);
#else
    Sleep(1000);
#endif
  }
  svtkTimerLog::DumpLogWithIndents(&cerr, 0);
}

int otherTimerLog(int, char*[])
{
  std::ostringstream svtkmsg_with_warning_C4701;
  otherTimerLogTest(svtkmsg_with_warning_C4701);
  timerLogScopeTest();
  return 0;
}
