/*=========================================================================

  Program:   Visualization Toolkit
  Module:    GLBenchmarking.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkActor.h"
#include "svtkAxis.h"
#include "svtkCamera.h"
#include "svtkCellArray.h"
#include "svtkChartLegend.h"
#include "svtkChartXY.h"
#include "svtkContextScene.h"
#include "svtkContextView.h"
#include "svtkDelimitedTextWriter.h"
#include "svtkDoubleArray.h"
#include "svtkIntArray.h"
#include "svtkNew.h"
#include "svtkPlot.h"
#include "svtkPolyData.h"
#include "svtkPolyDataMapper.h"
#include "svtkRenderWindow.h"
#include "svtkRenderer.h"
#include "svtkTable.h"
#include "svtkTimerLog.h"
#include "svtkVariant.h"
#include "svtkVector.h"

#include "svtkParametricBoy.h"
#include "svtkParametricFunctionSource.h"
#include "svtkParametricTorus.h"

#include <svtksys/CommandLineArguments.hxx>

namespace svtk
{
class BenchmarkTest
{
public:
  BenchmarkTest() { ; }
  virtual ~BenchmarkTest() { ; }

  virtual svtkIdType Build(svtkRenderer*, const svtkVector2i&) { return 0; }
};

class SurfaceTest : public BenchmarkTest
{
public:
  SurfaceTest() {}

  ~SurfaceTest() override {}

  svtkIdType Build(svtkRenderer* renderer, const svtkVector2i& res) override
  {
    // svtkVector2i res(20, 50);
    svtkNew<svtkParametricBoy> parametricShape;
    svtkNew<svtkParametricFunctionSource> parametricSource;
    parametricSource->SetParametricFunction(parametricShape);
    parametricSource->SetUResolution(res[0] * 50);
    parametricSource->SetVResolution(res[1] * 100);
    parametricSource->Update();

    svtkNew<svtkPolyDataMapper> mapper;
    mapper->SetInputConnection(parametricSource->GetOutputPort());
    mapper->SetScalarRange(0, 360);
    svtkNew<svtkActor> actor;
    actor->SetMapper(mapper);
    renderer->AddActor(actor);

    return parametricSource->GetOutput()->GetPolys()->GetNumberOfCells();
  }
};

svtkVector2i GenerateSequenceNumbers(int sequenceCount)
{
  const int seqX[] = { 1, 2, 3, 5, 5, 5, 6, 10 };
  const int seqY[] = { 1, 1, 1, 1, 2, 4, 5, 5 };
  svtkVector2i val(1, 1);
  while (sequenceCount >= 8)
  {
    val[0] *= 10;
    val[1] *= 10;
    sequenceCount -= 8;
  }
  val[0] *= seqX[sequenceCount];
  val[1] *= seqY[sequenceCount];
  return val;
}

} // End namespace

bool runTest(svtkRenderer* renderer, svtkTable* results, int seq, int row, double timeout = 0.5)
{
  svtk::SurfaceTest surfaceTest;
  svtkIdType triangles = surfaceTest.Build(renderer, svtk::GenerateSequenceNumbers(seq));

  double startTime = svtkTimerLog::GetUniversalTime();
  svtkRenderWindow* window = renderer->GetRenderWindow();
  renderer->ResetCamera();
  window->Render();

  double firstFrameTime = svtkTimerLog::GetUniversalTime() - startTime;

  renderer->GetActiveCamera()->Azimuth(90);
  renderer->ResetCameraClippingRange();

  int frameCount = 50;
  for (int i = 0; i < frameCount; ++i)
  {
    window->Render();
    renderer->GetActiveCamera()->Azimuth(3);
    renderer->GetActiveCamera()->Elevation(1);
  }
  double subsequentFrameTime =
    (svtkTimerLog::GetUniversalTime() - startTime - firstFrameTime) / frameCount;

  results->SetValue(row, 0, triangles);
  results->SetValue(row, 1, firstFrameTime);
  results->SetValue(row, 2, subsequentFrameTime);
  results->SetValue(row, 3, triangles / subsequentFrameTime * 1e-6);
  results->Modified();

  cout << "First frame:\t" << firstFrameTime << "\nAverage frame:\t" << subsequentFrameTime
       << "\nTriangles (M):\t" << triangles * 1e-6 << "\nMtris/sec:\t"
       << triangles / subsequentFrameTime * 1e-6 << "\nRow:\t" << row << endl;

  return subsequentFrameTime <= timeout;
}

class Arguments
{
public:
  Arguments(int argc, char* argv[])
    : Start(0)
    , End(16)
    , Timeout(1.0)
    , FileName("results.csv")
    , DisplayHelp(false)
  {
    typedef svtksys::CommandLineArguments arg;
    this->Args.Initialize(argc, argv);
    this->Args.AddArgument(
      "--start", arg::SPACE_ARGUMENT, &this->Start, "Start of the test sequence sizes");
    this->Args.AddArgument(
      "--end", arg::SPACE_ARGUMENT, &this->End, "End of the test sequence sizes");
    this->Args.AddArgument("--timeout", arg::SPACE_ARGUMENT, &this->Timeout,
      "Maximum average frame time before test termination");
    this->Args.AddArgument(
      "--file", arg::SPACE_ARGUMENT, &this->FileName, "File to save results to");
    this->Args.AddBooleanArgument(
      "--help", &this->DisplayHelp, "Provide a listing of command line options");

    if (!this->Args.Parse())
    {
      cerr << "Problem parsing arguments" << endl;
    }

    if (this->DisplayHelp)
    {
      cout << "Usage" << endl << endl << this->Args.GetHelp() << endl;
    }
  }

  svtksys::CommandLineArguments Args;
  int Start;
  int End;
  double Timeout;
  std::string FileName;
  bool DisplayHelp;
};

int main(int argc, char* argv[])
{
  Arguments args(argc, argv);
  if (args.DisplayHelp)
  {
    return 0;
  }

  svtkNew<svtkRenderer> renderer;
  svtkNew<svtkRenderWindow> window;
  window->AddRenderer(renderer);
  window->SetSize(800, 600);
  renderer->SetBackground(0.2, 0.3, 0.4);
  svtkNew<svtkCamera> refCamera;
  refCamera->DeepCopy(renderer->GetActiveCamera());

  // Set up our results table, this will be used for our timings etc.
  svtkNew<svtkTable> results;
  svtkNew<svtkIntArray> tris;
  tris->SetName("Triangles");
  svtkNew<svtkDoubleArray> firstFrame;
  firstFrame->SetName("First Frame");
  svtkNew<svtkDoubleArray> averageFrame;
  averageFrame->SetName("Average Frame");
  svtkNew<svtkDoubleArray> triRate;
  triRate->SetName("Mtris/sec");
  results->AddColumn(tris);
  results->AddColumn(firstFrame);
  results->AddColumn(averageFrame);
  results->AddColumn(triRate);

  // Set up a chart to show the data being generated in real time.
  svtkNew<svtkContextView> chartView;
  chartView->GetRenderWindow()->SetSize(800, 600);
  svtkNew<svtkChartXY> chart;
  chartView->GetScene()->AddItem(chart);
  svtkPlot* plot = chart->AddPlot(svtkChart::LINE);
  plot->SetInputData(results, 0, 3);
  plot = chart->AddPlot(svtkChart::LINE);
  plot->SetInputData(results, 0, 1);
  chart->SetPlotCorner(plot, 1);
  plot = chart->AddPlot(svtkChart::LINE);
  plot->SetInputData(results, 0, 2);
  chart->SetPlotCorner(plot, 1);
  chart->GetAxis(svtkAxis::LEFT)->SetTitle("Mtris/sec");
  chart->GetAxis(svtkAxis::BOTTOM)->SetTitle("triangles");
  chart->GetAxis(svtkAxis::RIGHT)->SetTitle("time (sec)");
  chart->SetShowLegend(true);
  chart->GetLegend()->SetHorizontalAlignment(svtkChartLegend::LEFT);

  int startSeq = args.Start;
  int endSeq = args.End;
  results->SetNumberOfRows(endSeq - startSeq + 1);
  int row = 0;
  for (int i = startSeq; i <= endSeq; ++i)
  {
    cout << "Running sequence point " << i << endl;
    results->SetNumberOfRows(i - startSeq + 1);
    window->Render();
    renderer->RemoveAllViewProps();
    renderer->GetActiveCamera()->DeepCopy(refCamera);
    if (!runTest(renderer, results, i, row++, args.Timeout))
    {
      break;
    }
    if (results->GetNumberOfRows() > 1)
    {
      chart->RecalculateBounds();
      chartView->Render();
    }
  }

  svtkNew<svtkDelimitedTextWriter> writer;
  writer->SetInputData(results);
  writer->SetFileName(args.FileName.c_str());
  writer->Update();
  writer->Write();

  return 0;
}
