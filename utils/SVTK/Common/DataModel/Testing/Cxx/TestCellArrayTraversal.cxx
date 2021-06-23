/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestCellArray.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkCellArray.h"
#include "svtkCellArrayIterator.h"
#include "svtkLogger.h"
#include "svtkSmartPointer.h"
#include "svtkTimerLog.h"

namespace
{

void RunTest(bool use64BitStorage)
{
  const svtkIdType numTris = 25000;
  svtkIdType num;

  auto ca = svtkSmartPointer<svtkCellArray>::New();
  if (use64BitStorage)
  {
    cout << "\n=== Test performance of new svtkCellArray: 64-bit storage ===\n";
    ca->Use64BitStorage();
  }
  else
  {
    cout << "\n=== Test performance of new svtkCellArray: 32-bit storage ===\n";
    ca->Use32BitStorage();
  }

  svtkIdType tri[3] = { 0, 1, 2 };
  auto timer = svtkSmartPointer<svtkTimerLog>::New();

  svtkIdType npts;
  const svtkIdType* pts;

  // Insert
  num = 0;
  timer->StartTimer();
  for (auto i = 0; i < numTris; ++i)
  {
    ca->InsertNextCell(3, tri);
    ++num;
  }
  timer->StopTimer();
  cout << "Insert triangles: " << timer->GetElapsedTime() << "\n";
  cout << "   " << num << " triangles inserted\n";
  cout << "   Memory used: " << ca->GetActualMemorySize() << " kb\n";

  // Iterate directly over cell array
  num = 0;
  timer->StartTimer();
  for (ca->InitTraversal(); ca->GetNextCell(npts, pts);)
  {
    assert(npts == 3);
    ++num;
  }
  timer->StopTimer();
  cout << "Traverse cell array (legacy GetNextCell()): " << timer->GetElapsedTime() << "\n";
  cout << "   " << num << " triangles visited\n";

  // Iterate directly over cell array
  num = 0;
  timer->StartTimer();
  svtkIdType numCells = ca->GetNumberOfCells();
  for (auto cellId = 0; cellId < numCells; ++cellId)
  {
    ca->GetCellAtId(cellId, npts, pts);
    assert(npts == 3);
    ++num;
  }
  timer->StopTimer();
  cout << "Traverse cell array (new GetCellAtId()): " << timer->GetElapsedTime() << "\n";
  cout << "   " << num << " triangles visited\n";

  // Iterate using iterator
  num = 0;
  timer->StartTimer();
  auto iter = svtk::TakeSmartPointer(ca->NewIterator());
  for (iter->GoToFirstCell(); !iter->IsDoneWithTraversal(); iter->GoToNextCell())
  {
    iter->GetCurrentCell(npts, pts);
    assert(npts == 3);
    ++num;
  }
  timer->StopTimer();
  cout << "Iterator traversal: " << timer->GetElapsedTime() << "\n";
  cout << "   " << num << " triangles visited\n";
} // RunTest

void RunTests()
{
  // What is the size of svtkIdType?
  cout << "=== svtkIdType is: " << (sizeof(svtkIdType) * 8) << " bits ===\n";

  RunTest(false); // 32-bit
  RunTest(true);  // 64-bit
}

} // end anon namespace

int TestCellArrayTraversal(int, char*[])
{
  try
  {
    RunTests();
  }
  catch (std::exception& err)
  {
    svtkLog(ERROR, << err.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
