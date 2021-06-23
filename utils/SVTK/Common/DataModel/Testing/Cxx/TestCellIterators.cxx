/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestCellIterators.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkCellIterator.h"

#include "svtkCellArray.h"
#include "svtkFloatArray.h"
#include "svtkGenericCell.h"
#include "svtkNew.h"
#include "svtkPoints.h"
#include "svtkSmartPointer.h"
#include "svtkTestUtilities.h"
#include "svtkTimerLog.h"
#include "svtkUnsignedCharArray.h"
#include "svtkUnstructuredGrid.h"
#include "svtkUnstructuredGridReader.h"

#include <sstream>
#include <string>

// Enable/disable code that helps/hinders profiling.
#undef PROFILE
//#define PROFILE

// Enable benchmarks.
#undef BENCHMARK
//#define BENCHMARK

#ifdef BENCHMARK
#ifdef PROFILE
#define NUM_BENCHMARKS 10
#else // PROFILE
#define NUM_BENCHMARKS 100
#endif // PROFILE
#endif // BENCHMARK

//------------------------------------------------------------------------------
// Compare the cell type, point ids, and points in 'grid' with those returned
// in 'iter'.
bool testCellIterator(svtkCellIterator* iter, svtkUnstructuredGrid* grid)
{
  svtkIdType cellId = 0;
  svtkNew<svtkGenericCell> cell;
  iter->InitTraversal();
  while (!iter->IsDoneWithTraversal())
  {
    grid->GetCell(cellId, cell);

    if (iter->GetCellType() != cell->GetCellType())
    {
      cerr << "Type mismatch for cell " << cellId << endl;
      return false;
    }

    svtkIdType numPoints = iter->GetNumberOfPoints();
    if (numPoints != cell->GetNumberOfPoints())
    {
      cerr << "Number of points mismatch for cell " << cellId << endl;
      return false;
    }

    for (svtkIdType pointInd = 0; pointInd < numPoints; ++pointInd)
    {
      if (iter->GetPointIds()->GetId(pointInd) != cell->PointIds->GetId(pointInd))
      {
        cerr << "Point id mismatch in cell " << cellId << endl;
        return false;
      }

      double iterPoint[3];
      double cellPoint[3];
      iter->GetPoints()->GetPoint(pointInd, iterPoint);
      cell->Points->GetPoint(pointInd, cellPoint);
      if (iterPoint[0] != cellPoint[0] || iterPoint[1] != cellPoint[1] ||
        iterPoint[2] != cellPoint[2])
      {
        cerr << "Point mismatch in cell " << cellId << endl;
        return false;
      }
    }

    iter->GoToNextCell();
    ++cellId;
  }

  // ensure that we checked all of the cells
  if (cellId != grid->GetNumberOfCells())
  {
    cerr << "Iterator did not cover all cells in the dataset!" << endl;
    return false;
  }

  //  cout << "Verified " << cellId << " cells with a " << iter->GetClassName()
  //       << "." << endl;
  return true;
}

#define TEST_ITERATOR(iter_, className_)                                                           \
  if (std::string(#className_) != std::string(iter->GetClassName()))                               \
  {                                                                                                \
    cerr << "Unexpected iterator type (expected " #className_ ", got " << (iter_)->GetClassName()  \
         << ")" << endl;                                                                           \
    return false;                                                                                  \
  }                                                                                                \
                                                                                                   \
  if (!testCellIterator(iter_, grid))                                                              \
  {                                                                                                \
    cerr << #className_ << " test failed." << endl;                                                \
    return false;                                                                                  \
  }                                                                                                \
                                                                                                   \
  if (!testCellIterator(iter_, grid))                                                              \
  {                                                                                                \
    cerr << #className_ << " test failed after rewind." << endl;                                   \
    return false;                                                                                  \
  }

bool runValidation(svtkUnstructuredGrid* grid)
{
  // svtkDataSetCellIterator:
  svtkCellIterator* iter = grid->svtkDataSet::NewCellIterator();
  TEST_ITERATOR(iter, svtkDataSetCellIterator);
  iter->Delete();

  // svtkPointSetCellIterator:
  iter = grid->svtkPointSet::NewCellIterator();
  TEST_ITERATOR(iter, svtkPointSetCellIterator);
  iter->Delete();

  // svtkUnstructuredGridCellIterator:
  iter = grid->svtkUnstructuredGrid::NewCellIterator();
  TEST_ITERATOR(iter, svtkUnstructuredGridCellIterator);
  iter->Delete();

  return true;
}

// Benchmarking code follows:
#ifdef BENCHMARK

// Do-nothing function that ensures arguments passed in will not be compiled
// out. Aggressive optimization will otherwise remove portions of the following
// loops, throwing off the benchmark results:
namespace
{
std::stringstream _sink;
template <class Type>
void useData(const Type& data)
{
  _sink << data;
}
} // end anon namespace

// There are three signatures for each benchmark function:
// - double ()(svtkUnstructuredGrid *)
//   Iterate through cells in an unstructured grid, using raw memory when
//   possible.
// - double ()(svtkUnstructuredGrid *, int)
//   Iterator through cells in an unstructured grid, using API only
// - double ()(svtkCellIterator *)
//   Iterator through all cells available through the iterator.
double benchmarkTypeIteration(svtkUnstructuredGrid* grid)
{
  svtkIdType numCells = grid->GetNumberOfCells();
  svtkUnsignedCharArray* types = grid->GetCellTypesArray();
  unsigned char* ptr = types->GetPointer(0);
  unsigned char range[2] = { SVTK_UNSIGNED_CHAR_MAX, SVTK_UNSIGNED_CHAR_MIN };

  svtkNew<svtkTimerLog> timer;
  timer->StartTimer();
  for (int i = 0; i < numCells; ++i)
  {
    range[0] = std::min(range[0], ptr[i]);
    range[1] = std::max(range[1], ptr[i]);
  }
  timer->StopTimer();

  useData(range[0]);
  useData(range[1]);

  return timer->GetElapsedTime();
}

double benchmarkTypeIteration(svtkUnstructuredGrid* grid, int)
{
  svtkIdType numCells = grid->GetNumberOfCells();
  unsigned char tmp;
  unsigned char range[2] = { SVTK_UNSIGNED_CHAR_MAX, SVTK_UNSIGNED_CHAR_MIN };

  svtkNew<svtkTimerLog> timer;
  timer->StartTimer();
  for (int i = 0; i < numCells; ++i)
  {
    tmp = static_cast<unsigned char>(grid->GetCellType(i));
    range[0] = std::min(range[0], tmp);
    range[1] = std::max(range[1], tmp);
  }
  timer->StopTimer();

  useData(range[0]);
  useData(range[1]);

  return timer->GetElapsedTime();
}

double benchmarkTypeIteration(svtkCellIterator* iter)
{
  int range[2] = { SVTK_UNSIGNED_CHAR_MAX, SVTK_UNSIGNED_CHAR_MIN };
  int tmp;

  svtkNew<svtkTimerLog> timer;
  timer->StartTimer();
  for (iter->InitTraversal(); iter->IsDoneWithTraversal(); iter->GoToNextCell())
  {
    tmp = iter->GetCellType();
    range[0] = std::min(range[0], tmp);
    range[1] = std::max(range[1], tmp);
  }
  timer->StopTimer();

  useData(range[0]);
  useData(range[1]);

  return timer->GetElapsedTime();
}

double benchmarkPointIdIteration(svtkUnstructuredGrid* grid)
{
  svtkCellArray* cellArray = grid->GetCells();
  svtkIdType numCells = cellArray->GetNumberOfCells();
  svtkIdType* cellPtr = cellArray - ;
  svtkIdType range[2] = { SVTK_ID_MAX, SVTK_ID_MIN };
  svtkIdType cellSize;

  svtkNew<svtkTimerLog> timer;
  timer->StartTimer();
  for (svtkIdType cellId = 0; cellId < numCells; ++cellId)
  {
    cellSize = *(cellPtr++);
    for (svtkIdType pointIdx = 0; pointIdx < cellSize; ++pointIdx)
    {
      range[0] = std::min(range[0], cellPtr[pointIdx]);
      range[1] = std::max(range[1], cellPtr[pointIdx]);
    }
    cellPtr += cellSize;
  }
  timer->StopTimer();

  useData(range[0]);
  useData(range[1]);

  return timer->GetElapsedTime();
}

double benchmarkPointIdIteration(svtkUnstructuredGrid* grid, int)
{
  svtkIdType numCells = grid->GetNumberOfCells();
  svtkIdType range[2] = { SVTK_ID_MAX, SVTK_ID_MIN };
  svtkIdType cellSize;
  svtkIdList* cellPointIds = svtkIdList::New();
  svtkIdType* cellPtr;

  svtkNew<svtkTimerLog> timer;
  timer->StartTimer();
  for (svtkIdType cellId = 0; cellId < numCells; ++cellId)
  {
    grid->GetCellPoints(cellId, cellPointIds);
    cellSize = cellPointIds->GetNumberOfIds();
    cellPtr = cellPointIds->GetPointer(0);
    for (svtkIdType pointIdx = 0; pointIdx < cellSize; ++pointIdx)
    {
      range[0] = std::min(range[0], cellPtr[pointIdx]);
      range[1] = std::max(range[1], cellPtr[pointIdx]);
    }
  }
  timer->StopTimer();

  useData(range[0]);
  useData(range[1]);

  cellPointIds->Delete();

  return timer->GetElapsedTime();
}

double benchmarkPointIdIteration(svtkCellIterator* iter)
{
  svtkIdType range[2] = { SVTK_ID_MAX, SVTK_ID_MIN };
  svtkIdType* cellPtr;
  svtkIdType* cellEnd;

  svtkNew<svtkTimerLog> timer;
  timer->StartTimer();
  for (iter->InitTraversal(); iter->IsDoneWithTraversal(); iter->GoToNextCell())
  {
    cellPtr = iter->GetPointIds()->GetPointer(0);
    cellEnd = cellPtr + iter->GetNumberOfPoints();
    while (cellPtr != cellEnd)
    {
      range[0] = std::min(range[0], *cellPtr);
      range[1] = std::max(range[1], *cellPtr);
      ++cellPtr;
    }
  }
  timer->StopTimer();

  useData(range[0]);
  useData(range[1]);

  return timer->GetElapsedTime();
}

double benchmarkPointsIteration(svtkUnstructuredGrid* grid)
{
  svtkCellArray* cellArray = grid->GetCells();
  const svtkIdType numCells = cellArray->GetNumberOfCells();
  svtkIdType* cellPtr = cellArray - ;
  svtkIdType cellSize;

  svtkPoints* points = grid->GetPoints();
  svtkFloatArray* pointDataArray = svtkArrayDownCast<svtkFloatArray>(points->GetData());
  if (!pointDataArray)
  {
    return -1.0;
  }
  float* pointData = pointDataArray->GetPointer(0);
  float* point;
  float dummy[3] = { 0.f, 0.f, 0.f };

  svtkNew<svtkTimerLog> timer;
  timer->StartTimer();
  for (svtkIdType cellId = 0; cellId < numCells; ++cellId)
  {
    cellSize = *(cellPtr++);
    for (svtkIdType pointIdx = 0; pointIdx < cellSize; ++pointIdx)
    {
      point = pointData + 3 * cellPtr[pointIdx];
      dummy[0] += point[0];
      dummy[1] += point[1];
      dummy[2] += point[2];
    }
    cellPtr += cellSize;
  }
  timer->StopTimer();

  useData(dummy[0]);
  useData(dummy[1]);
  useData(dummy[2]);

  return timer->GetElapsedTime();
}

double benchmarkPointsIteration(svtkUnstructuredGrid* grid, int)
{
  svtkIdList* pointIds = svtkIdList::New();
  svtkIdType cellSize;
  svtkIdType* cellPtr;

  svtkPoints* points = grid->GetPoints();
  double point[3];
  double dummy[3] = { 0.f, 0.f, 0.f };

  svtkNew<svtkTimerLog> timer;
  timer->StartTimer();
  const svtkIdType numCells = grid->GetNumberOfCells();
  for (svtkIdType cellId = 0; cellId < numCells; ++cellId)
  {
    grid->GetCellPoints(cellId, pointIds);
    cellSize = pointIds->GetNumberOfIds();
    cellPtr = pointIds->GetPointer(0);
    for (svtkIdType pointIdx = 0; pointIdx < cellSize; ++pointIdx)
    {
      points->GetPoint(cellPtr[pointIdx], point);
      dummy[0] += point[0];
      dummy[1] += point[1];
      dummy[2] += point[2];
    }
  }
  timer->StopTimer();

  useData(dummy[0]);
  useData(dummy[1]);
  useData(dummy[2]);

  pointIds->Delete();

  return timer->GetElapsedTime();
}

double benchmarkPointsIteration(svtkCellIterator* iter)
{
  float dummy[3] = { 0.f, 0.f, 0.f };

  // Ensure that the call to GetPoints() is at a valid cell:
  iter->InitTraversal();
  if (!iter->IsDoneWithTraversal())
  {
    return -1.0;
  }
  svtkFloatArray* pointArray = svtkArrayDownCast<svtkFloatArray>(iter->GetPoints()->GetData());
  float* pointsData;
  float* pointsDataEnd;

  svtkNew<svtkTimerLog> timer;
  timer->StartTimer();
  for (iter->InitTraversal(); iter->IsDoneWithTraversal(); iter->GoToNextCell())
  {
    pointsData = pointArray->GetPointer(0);
    pointsDataEnd = pointsData + iter->GetNumberOfPoints();
    while (pointsData < pointsDataEnd)
    {
      dummy[0] += *pointsData++;
      dummy[1] += *pointsData++;
      dummy[2] += *pointsData++;
    }
  }
  timer->StopTimer();

  useData(dummy[0]);
  useData(dummy[1]);
  useData(dummy[2]);

  return timer->GetElapsedTime();
}

double benchmarkCellIteration(svtkUnstructuredGrid* grid)
{
  svtkGenericCell* cell = svtkGenericCell::New();
  svtkIdType numCells = grid->GetNumberOfCells();

  svtkNew<svtkTimerLog> timer;
  timer->StartTimer();
  for (svtkIdType cellId = 0; cellId < numCells; ++cellId)
  {
    grid->GetCell(cellId, cell);
  }
  timer->StopTimer();
  cell->Delete();
  return timer->GetElapsedTime();
}

double benchmarkCellIteration(svtkUnstructuredGrid* grid, int)
{
  // No real difference here....
  return benchmarkCellIteration(grid);
}

double benchmarkCellIteration(svtkCellIterator* it)
{
  svtkGenericCell* cell = svtkGenericCell::New();

  svtkNew<svtkTimerLog> timer;
  timer->StartTimer();
  for (it->InitTraversal(); it->IsDoneWithTraversal(); it->GoToNextCell())
  {
    it->GetCell(cell);
  }
  timer->StopTimer();
  cell->Delete();
  return timer->GetElapsedTime();
}

double benchmarkPiecewiseIteration(svtkUnstructuredGrid* grid)
{
  // Setup for types:
  svtkUnsignedCharArray* typeArray = grid->GetCellTypesArray();
  unsigned char* typePtr = typeArray->GetPointer(0);
  unsigned char typeRange[2] = { SVTK_UNSIGNED_CHAR_MAX, SVTK_UNSIGNED_CHAR_MIN };

  // Setup for point ids:
  svtkCellArray* cellArray = grid->GetCells();
  svtkIdType* cellArrayPtr = cellArray - ;
  svtkIdType ptIdRange[2] = { SVTK_ID_MAX, SVTK_ID_MIN };
  svtkIdType cellSize;

  // Setup for points:
  svtkPoints* points = grid->GetPoints();
  svtkFloatArray* pointDataArray = svtkArrayDownCast<svtkFloatArray>(points->GetData());
  if (!pointDataArray)
  {
    return -1.0;
  }
  float* pointData = pointDataArray->GetPointer(0);
  float* point;
  float dummy[3] = { 0.f, 0.f, 0.f };

  // Setup for cells
  svtkGenericCell* cell = svtkGenericCell::New();

  svtkIdType numCells = grid->GetNumberOfCells();
  svtkNew<svtkTimerLog> timer;
  timer->StartTimer();
  for (int i = 0; i < numCells; ++i)
  {
    // Types:
    typeRange[0] = std::min(typeRange[0], typePtr[i]);
    typeRange[1] = std::max(typeRange[1], typePtr[i]);

    cellSize = *(cellArrayPtr++);
    for (svtkIdType pointIdx = 0; pointIdx < cellSize; ++pointIdx)
    {
      // Point ids:
      ptIdRange[0] = std::min(ptIdRange[0], cellArrayPtr[pointIdx]);
      ptIdRange[1] = std::max(ptIdRange[1], cellArrayPtr[pointIdx]);

      // Points:
      point = pointData + 3 * cellArrayPtr[pointIdx];
      dummy[0] += point[0];
      dummy[1] += point[1];
      dummy[2] += point[2];
    }
    cellArrayPtr += cellSize;

    // Cell:
    grid->GetCell(i, cell);
  }
  timer->StopTimer();

  useData(typeRange[0]);
  useData(typeRange[1]);

  useData(ptIdRange[0]);
  useData(ptIdRange[1]);

  useData(dummy[0]);
  useData(dummy[1]);
  useData(dummy[2]);

  cell->Delete();

  return timer->GetElapsedTime();
}

double benchmarkPiecewiseIteration(svtkUnstructuredGrid* grid, int)
{
  // Setup for type
  unsigned char cellType;
  unsigned char typeRange[2] = { SVTK_UNSIGNED_CHAR_MAX, SVTK_UNSIGNED_CHAR_MIN };

  // Setup for point ids
  svtkIdType ptIdRange[2] = { SVTK_ID_MAX, SVTK_ID_MIN };
  svtkIdType cellSize;
  svtkIdList* cellPointIds = svtkIdList::New();
  svtkIdType* cellPtIdPtr;

  // Setup for points
  svtkPoints* points = grid->GetPoints();
  double point[3];
  double dummy[3] = { 0.f, 0.f, 0.f };

  // Setup for cells
  svtkGenericCell* cell = svtkGenericCell::New();

  svtkIdType numCells = grid->GetNumberOfCells();
  svtkNew<svtkTimerLog> timer;
  timer->StartTimer();
  for (svtkIdType cellId = 0; cellId < numCells; ++cellId)
  {
    // Cell type
    cellType = static_cast<unsigned char>(grid->GetCellType(cellId));
    typeRange[0] = std::min(typeRange[0], cellType);
    typeRange[1] = std::max(typeRange[1], cellType);

    grid->GetCellPoints(cellId, cellPointIds);
    cellSize = cellPointIds->GetNumberOfIds();
    cellPtIdPtr = cellPointIds->GetPointer(0);
    for (svtkIdType pointIdx = 0; pointIdx < cellSize; ++pointIdx)
    {
      // Point ids:
      ptIdRange[0] = std::min(ptIdRange[0], cellPtIdPtr[pointIdx]);
      ptIdRange[1] = std::max(ptIdRange[1], cellPtIdPtr[pointIdx]);

      // Points:
      points->GetPoint(cellPtIdPtr[pointIdx], point);
      dummy[0] += point[0];
      dummy[1] += point[1];
      dummy[2] += point[2];
    }

    // Cell:
    grid->GetCell(cellId, cell);
  }
  timer->StopTimer();

  useData(typeRange[0]);
  useData(typeRange[1]);

  useData(ptIdRange[0]);
  useData(ptIdRange[1]);

  useData(dummy[0]);
  useData(dummy[1]);
  useData(dummy[2]);

  cellPointIds->Delete();

  return timer->GetElapsedTime();
}

double benchmarkPiecewiseIteration(svtkCellIterator* iter)
{
  // Type setup:
  int typeRange[2] = { SVTK_UNSIGNED_CHAR_MAX, SVTK_UNSIGNED_CHAR_MIN };

  // Point ids setups:
  svtkIdType ptIdRange[2] = { SVTK_ID_MAX, SVTK_ID_MIN };
  svtkIdType* cellPtr;
  svtkIdType cellSize;

  // Points setup:
  float dummy[3] = { 0.f, 0.f, 0.f };
  float* pointsPtr;

  // Cell setup
  svtkGenericCell* cell = svtkGenericCell::New();

  svtkNew<svtkTimerLog> timer;
  timer->StartTimer();
  for (iter->InitTraversal(); iter->IsDoneWithTraversal(); iter->GoToNextCell())
  {
    // Types:
    typeRange[0] = std::min(typeRange[0], iter->GetCellType());
    typeRange[1] = std::max(typeRange[1], iter->GetCellType());

    cellPtr = iter->GetPointIds()->GetPointer(0);
    pointsPtr = static_cast<float*>(iter->GetPoints()->GetVoidPointer(0));
    cellSize = iter->GetPointIds()->GetNumberOfIds();
    while (cellSize-- > 0)
    {
      // Point Ids:
      ptIdRange[0] = std::min(ptIdRange[0], *cellPtr);
      ptIdRange[1] = std::max(ptIdRange[1], *cellPtr);
      ++cellPtr;

      // Points:
      dummy[0] += *pointsPtr++;
      dummy[1] += *pointsPtr++;
      dummy[2] += *pointsPtr++;
    }

    // Cell:
    iter->GetCell(cell);
  }
  timer->StopTimer();

  useData(typeRange[0]);
  useData(typeRange[1]);

  useData(ptIdRange[0]);
  useData(ptIdRange[1]);

  useData(dummy[0]);
  useData(dummy[1]);
  useData(dummy[2]);

  cell->Delete();

  return timer->GetElapsedTime();
}

#define BENCHMARK_ITERATORS(grid_, test_, bench_)                                                  \
  if (!runBenchmark(grid_, test_, bench_, bench_, bench_))                                         \
  {                                                                                                \
    cerr << "Benchmark '" << test_ << "' encountered an error." << endl;                           \
    return false;                                                                                  \
  }

typedef double (*BenchmarkRefType)(svtkUnstructuredGrid*);
typedef double (*BenchmarkApiType)(svtkUnstructuredGrid*, int);
typedef double (*BenchmarkIterType)(svtkCellIterator*);
bool runBenchmark(svtkUnstructuredGrid* grid, const std::string& test, BenchmarkRefType refBench,
  BenchmarkApiType apiBench, BenchmarkIterType iterBench)
{
  const int numBenchmarks = NUM_BENCHMARKS;
  double refTime = 0.;
  double apiTime = 0.;
  double dsTime = 0.;
  double psTime = 0.;
  double ugTime = 0.;

  svtkCellIterator* dsIter = grid->svtkDataSet::NewCellIterator();
  svtkCellIterator* psIter = grid->svtkPointSet::NewCellIterator();
  svtkCellIterator* ugIter = grid->NewCellIterator();

  cout << "Testing " << test << " (" << numBenchmarks << " samples):" << endl;

#ifdef PROFILE
  std::string prog;
  prog.resize(12, ' ');
  prog[0] = prog[11] = '|';
#endif // PROFILE

  for (int i = 0; i < numBenchmarks; ++i)
  {
#ifdef PROFILE
    std::fill_n(prog.begin() + 1, i * 10 / numBenchmarks, '=');
    cout << "\rProgress: " << prog << " (" << i << "/" << numBenchmarks << ")" << endl;
#endif // PROFILE

    refTime += refBench(grid);
    apiTime += apiBench(grid, 0);
    dsTime += iterBench(dsIter);
    psTime += iterBench(psIter);
    ugTime += iterBench(ugIter);
  }

#ifdef PROFILE
  std::fill_n(prog.begin() + 1, 10, '=');
  cout << "\rProgress: " << prog << " (" << numBenchmarks << "/" << numBenchmarks << ")" << endl;
#endif // PROFILE

  refTime /= static_cast<double>(numBenchmarks);
  apiTime /= static_cast<double>(numBenchmarks);
  dsTime /= static_cast<double>(numBenchmarks);
  psTime /= static_cast<double>(numBenchmarks);
  ugTime /= static_cast<double>(numBenchmarks);

  const std::string sep("\t");
  cout << std::setw(8)

       << "\t"
       << "Ref (raw)" << sep << "Ref (api)" << sep << "DSIter" << sep << "PSIter" << sep << "UGIter"
       << endl
       << "\t" << refTime << sep << apiTime << sep << dsTime << sep << psTime << sep << ugTime
       << endl;

  dsIter->Delete();
  psIter->Delete();
  ugIter->Delete();

  return true;
}

bool runBenchmarks(svtkUnstructuredGrid* grid)
{
  BENCHMARK_ITERATORS(grid, "cell type", benchmarkTypeIteration);
  BENCHMARK_ITERATORS(grid, "cell pointId", benchmarkPointIdIteration);
  BENCHMARK_ITERATORS(grid, "cell point", benchmarkPointsIteration);
  BENCHMARK_ITERATORS(grid, "cells", benchmarkCellIteration);
  BENCHMARK_ITERATORS(grid, "piecewise", benchmarkPiecewiseIteration);
  return true;
}
#endif // Benchmark

int TestCellIterators(int argc, char* argv[])
{
  // Load an unstructured grid dataset
  char* fileNameC = svtkTestUtilities::ExpandDataFileName(argc, argv, "Data/blowGeom.svtk");
  std::string fileName(fileNameC);
  delete[] fileNameC;

  svtkNew<svtkUnstructuredGridReader> reader;
  reader->SetFileName(fileName.c_str());
  reader->Update();
  svtkUnstructuredGrid* grid(reader->GetOutput());
  if (!grid)
  {
    cerr << "Error reading file: " << fileName << endl;
    return EXIT_FAILURE;
  }

#ifndef PROFILE
  if (!runValidation(grid))
  {
    return EXIT_FAILURE;
  }
#endif // not PROFILE

#ifdef BENCHMARK
  if (!runBenchmarks(grid))
  {
    return EXIT_FAILURE;
  }

  // Reference _sink to prevent optimizations from interfering with the
  // benchmarks.
  if (_sink.str().size() == 0)
  {
    return EXIT_FAILURE;
  }
#endif // BENCHMARK

  return EXIT_SUCCESS;
}
