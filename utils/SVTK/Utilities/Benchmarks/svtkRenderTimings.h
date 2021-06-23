/*=========================================================================

  Program:   Visualization Toolkit
  Module:    SVTKRenderTimings.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef svtkRenderTimings_h
#define svtkRenderTimings_h

/**
 * Define the classes we use for running timing benchmarks
 */

#include "svtkTimerLog.h"
#include "svtkUtilitiesBenchmarksModule.h"
#include <map>
#include <svtksys/CommandLineArguments.hxx>

class svtkRTTestResult;
class svtkRTTestSequence;
class svtkRenderTimings;

class SVTKUTILITIESBENCHMARKS_EXPORT svtkRTTest
{
public:
  // what is the name of this test
  std::string GetName() { return this->Name; }

  // when reporting a summary result use this key to
  // determine the amount of triangles rendered
  virtual const char* GetSecondSummaryResultName() = 0;

  // when reporting a summary result this is the
  // field that should be reported.
  virtual const char* GetSummaryResultName() = 0;

  // when reporting a summary result should we use the
  // largest value or smallest?
  virtual bool UseLargestSummaryResult() { return true; }

  // Set/Get the time allowed for this test
  // Tests should check if they are going more than 50%
  // beyond this number they should short circuit if
  // they can gracefully.
  virtual void SetTargetTime(float tt) { this->TargetTime = tt; }
  virtual float GetTargetTime() { return this->TargetTime; }

  void SetRenderSize(int width, int height)
  {
    this->RenderWidth = width;
    this->RenderHeight = height;
  }
  int GetRenderWidth() { return this->RenderWidth; }
  int GetRenderHeight() { return this->RenderHeight; }

  // run the test, argc and argv are extra arguments that the test might
  // use.
  virtual svtkRTTestResult Run(svtkRTTestSequence* ats, int argc, char* argv[]) = 0;

  svtkRTTest(const char* name)
  {
    this->TargetTime = 1.0;
    this->Name = name;
    RenderWidth = RenderHeight = 600;
  }

  virtual ~svtkRTTest() {}

protected:
  float TargetTime;
  std::string Name;
  int RenderWidth, RenderHeight;
};

class SVTKUTILITIESBENCHMARKS_EXPORT svtkRTTestResult
{
public:
  std::map<std::string, double> Results;
  int SequenceNumber;
  void ReportResults(svtkRTTest* test, ostream& ost)
  {
    ost << test->GetName();
    std::map<std::string, double>::iterator rItr;
    for (rItr = this->Results.begin(); rItr != this->Results.end(); ++rItr)
    {
      ost << ", " << rItr->first << ", " << rItr->second;
    }
    ost << "\n";
  }
};

class SVTKUTILITIESBENCHMARKS_EXPORT svtkRTTestSequence
{
public:
  virtual void Run();
  virtual void ReportSummaryResults(ostream& ost);
  virtual void ReportDetailedResults(ostream& ost);

  // tests should use these functions to determine what resolution
  // to use in scaling their test. The functions will always return
  // numbers then when multiplied will result in 1, 2, 3, or 5
  // times 10 to some power. These functions use the SequenceCount
  // to determine what number to return. When the dimensions
  // are not equal, we guarantee that the larger dimensions
  // come first
  void GetSequenceNumbers(int& xdim);
  void GetSequenceNumbers(int& xdim, int& ydim);
  void GetSequenceNumbers(int& xdim, int& ydim, int& zdim);
  void GetSequenceNumbers(int& xdim, int& ydim, int& zdim, int& wdim);

  // display the results in realtime using SVTK charting
  void SetChartResults(bool v) { this->ChartResults = v; }

  svtkRTTest* Test;
  float TargetTime;

  svtkRTTestSequence(svtkRenderTimings* rt)
  {
    this->Test = NULL;
    this->TargetTime = 10.0;
    this->RenderTimings = rt;
    this->ChartResults = true;
  }

  virtual ~svtkRTTestSequence() {}

protected:
  std::vector<svtkRTTestResult> TestResults;
  int SequenceCount;
  svtkRenderTimings* RenderTimings;
  bool ChartResults;
};

// a class to run a bunch of timing tests and
// report the results
class SVTKUTILITIESBENCHMARKS_EXPORT svtkRenderTimings
{
public:
  svtkRenderTimings();

  // get the sequence start and end values
  int GetSequenceStart() { return this->SequenceStart; }
  int GetSequenceEnd() { return this->SequenceEnd; }

  // get the maxmimum time allowed per step
  double GetSequenceStepTimeLimit() { return this->SequenceStepTimeLimit; }

  // get the render size
  int GetRenderWidth() { return this->RenderWidth; }
  int GetRenderHeight() { return this->RenderHeight; }

  // parse and act on the command line arguments
  int ParseCommandLineArguments(int argc, char* argv[]);

  // get the arguments
  svtksys::CommandLineArguments& GetArguments() { return this->Arguments; }

  std::string GetSystemName() { return this->SystemName; }

  std::vector<svtkRTTest*> TestsToRun;
  std::vector<svtkRTTestSequence*> TestSequences;

protected:
  int RunTests();
  void ReportResults();

private:
  std::string Regex; // regular expression for tests
  double TargetTime;
  std::string SystemName;
  svtksys::CommandLineArguments Arguments;
  bool DisplayHelp;
  bool ListTests;
  bool NoChartResults;
  int SequenceStart;
  int SequenceEnd;
  double SequenceStepTimeLimit;
  std::string DetailedResultsFileName;
  int RenderWidth;
  int RenderHeight;
};

#endif
// SVTK-HeaderTest-Exclude: svtkRenderTimings.h
