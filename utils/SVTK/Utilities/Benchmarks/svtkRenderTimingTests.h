/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkRenderTimingTests.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef svtkRenderTimingTests_h
#define svtkRenderTimingTests_h

/*
To add a test you must define a subclass of svtkRTTest and implement the
pure virtual functions. Then in the main section at the bottom of this
file add your test to the tests to be run and rebuild. See some of the
existing tests to get an idea of what to do.
*/

#include "svtkRenderTimings.h"

#include "svtkActor.h"
#include "svtkAutoInit.h"
#include "svtkCamera.h"
#include "svtkCellArray.h"
#include "svtkCullerCollection.h"
#include "svtkNew.h"
#include "svtkPointData.h"
#include "svtkPolyData.h"
#include "svtkPolyDataMapper.h"
#include "svtkRenderWindow.h"
#include "svtkRenderer.h"
#include "svtkRenderingOpenGLConfigure.h"

/*=========================================================================
Define a test for simple triangle mesh surfaces
=========================================================================*/
#include "svtkParametricBoy.h"
#include "svtkParametricFunctionSource.h"
#include "svtkParametricTorus.h"

class surfaceTest : public svtkRTTest
{
public:
  surfaceTest(const char* name, bool withColors, bool withNormals)
    : svtkRTTest(name)
  {
    this->WithColors = withColors;
    this->WithNormals = withNormals;
  }

  const char* GetSummaryResultName() override { return "Mtris/sec"; }

  const char* GetSecondSummaryResultName() override { return "Mtris"; }

  svtkRTTestResult Run(svtkRTTestSequence* ats, int /*argc*/, char* /* argv */[]) override
  {
    int ures, vres;
    ats->GetSequenceNumbers(ures, vres);

    // ------------------------------------------------------------
    // Create surface
    // ------------------------------------------------------------
    //  svtkNew<svtkParametricBoy> PB;
    svtkNew<svtkParametricTorus> PB;
    svtkNew<svtkParametricFunctionSource> PFS;
    PFS->SetParametricFunction(PB.Get());
    if (this->WithColors)
    {
      PFS->SetScalarModeToPhase();
    }
    else
    {
      PFS->SetScalarModeToNone();
    }
    if (this->WithNormals == false)
    {
      PFS->GenerateNormalsOff();
    }
    PFS->SetUResolution(ures * 50);
    PFS->SetVResolution(vres * 100);
    PFS->Update();

    svtkNew<svtkPolyDataMapper> mapper;
    mapper->SetInputConnection(PFS->GetOutputPort());
    mapper->SetScalarRange(0.0, 360.0);

    svtkNew<svtkActor> actor;
    actor->SetMapper(mapper.Get());

    // create a rendering window and renderer
    svtkNew<svtkRenderer> ren1;
    svtkNew<svtkRenderWindow> renWindow;
    renWindow->AddRenderer(ren1.Get());
    ren1->AddActor(actor.Get());

    // set the size/color of our window
    renWindow->SetSize(this->GetRenderWidth(), this->GetRenderHeight());
    ren1->SetBackground(0.2, 0.3, 0.5);

    // draw the resulting scene
    double startTime = svtkTimerLog::GetUniversalTime();
    renWindow->Render();
    double firstFrameTime = svtkTimerLog::GetUniversalTime() - startTime;
    ren1->GetActiveCamera()->Azimuth(90);
    ren1->ResetCameraClippingRange();

    int frameCount = 80;
    for (int i = 0; i < frameCount; i++)
    {
      renWindow->Render();
      ren1->GetActiveCamera()->Azimuth(1);
      ren1->GetActiveCamera()->Elevation(1);
      if ((svtkTimerLog::GetUniversalTime() - startTime - firstFrameTime) > this->TargetTime * 1.5)
      {
        frameCount = i + 1;
        break;
      }
    }
    double subsequentFrameTime =
      (svtkTimerLog::GetUniversalTime() - startTime - firstFrameTime) / frameCount;
    double numTris = PFS->GetOutput()->GetPolys()->GetNumberOfCells();

    svtkRTTestResult result;
    result.Results["first frame time"] = firstFrameTime;
    result.Results["subsequent frame time"] = subsequentFrameTime;
    result.Results["Mtris"] = 1.0e-6 * numTris;
    result.Results["Mtris/sec"] = 1.0e-6 * numTris / subsequentFrameTime;
    result.Results["triangles"] = numTris;

    return result;
  }

protected:
  bool WithNormals;
  bool WithColors;
};

/*=========================================================================
Define a test for glyphing
=========================================================================*/
#include "svtkElevationFilter.h"
#include "svtkGlyph3DMapper.h"
#include "svtkPlaneSource.h"
#include "svtkSphereSource.h"

class glyphTest : public svtkRTTest
{
public:
  glyphTest(const char* name)
    : svtkRTTest(name)
  {
  }

  const char* GetSummaryResultName() override { return "Mtris/sec"; }

  const char* GetSecondSummaryResultName() override { return "triangles"; }

  svtkRTTestResult Run(svtkRTTestSequence* ats, int /*argc*/, char* /* argv */[]) override
  {
    int res1, res2, res3, res4;
    ats->GetSequenceNumbers(res1, res2, res3, res4);

    // create
    svtkNew<svtkPlaneSource> plane;
    plane->SetResolution(res1 * 10, res2 * 10);
    plane->SetOrigin(-res1 * 5.0, -res2 * 5.0, 0.0);
    plane->SetPoint1(res1 * 5.0, -res2 * 5.0, 0.0);
    plane->SetPoint2(-res1 * 5.0, res2 * 5.0, 0.0);
    svtkNew<svtkElevationFilter> colors;
    colors->SetInputConnection(plane->GetOutputPort());
    colors->SetLowPoint(plane->GetOrigin());
    colors->SetHighPoint(res1 * 5.0, res2 * 5.0, 0.0);

    // create simple poly data so we can apply glyph
    svtkNew<svtkSphereSource> sphere;
    sphere->SetPhiResolution(5 * res3 + 2);
    sphere->SetThetaResolution(10 * res4);
    sphere->SetRadius(0.7);

    svtkNew<svtkGlyph3DMapper> mapper;
    mapper->SetInputConnection(colors->GetOutputPort());
    mapper->SetSourceConnection(sphere->GetOutputPort());
    mapper->SetScalarRange(0.0, 2.0);

    // svtkNew<svtkPolyDataMapper> mapper;
    // mapper->SetInputConnection(colors->GetOutputPort());
    // mapper->SetScalarRange(0.0,2.0);

    svtkNew<svtkActor> actor;
    actor->SetMapper(mapper.Get());

    // create a rendering window and renderer
    svtkNew<svtkRenderer> ren1;
    svtkNew<svtkRenderWindow> renWindow;
    renWindow->AddRenderer(ren1.Get());
    ren1->AddActor(actor.Get());

    // set the size/color of our window
    renWindow->SetSize(this->GetRenderWidth(), this->GetRenderHeight());
    ren1->SetBackground(0.2, 0.3, 0.5);

    // draw the resulting scene
    double startTime = svtkTimerLog::GetUniversalTime();
    renWindow->Render();
    double firstFrameTime = svtkTimerLog::GetUniversalTime() - startTime;

    int frameCount = 80;
    for (int i = 0; i < frameCount; i++)
    {
      renWindow->Render();
      ren1->GetActiveCamera()->Azimuth(0.5);
      ren1->GetActiveCamera()->Elevation(0.5);
      ren1->GetActiveCamera()->Zoom(1.01);
      ren1->ResetCameraClippingRange();
      if ((svtkTimerLog::GetUniversalTime() - startTime - firstFrameTime) > this->TargetTime * 1.5)
      {
        frameCount = i + 1;
        break;
      }
    }
    double subsequentFrameTime =
      (svtkTimerLog::GetUniversalTime() - startTime - firstFrameTime) / frameCount;
    double numTris = 100.0 * res1 * res2 * sphere->GetOutput()->GetPolys()->GetNumberOfCells();

    svtkRTTestResult result;
    result.Results["first frame time"] = firstFrameTime;
    result.Results["subsequent frame time"] = subsequentFrameTime;
    result.Results["Mtris"] = 1.0e-6 * numTris;
    result.Results["Mtris/sec"] = 1.0e-6 * numTris / subsequentFrameTime;
    result.Results["triangles"] = numTris;

    return result;
  }

protected:
};

/*=========================================================================
Define a test for molecules
=========================================================================*/
#include "svtkBoxMuellerRandomSequence.h"
#include "svtkMath.h"
#include "svtkMolecule.h"
#include "svtkMoleculeMapper.h"
#include "svtkPointLocator.h"

class moleculeTest : public svtkRTTest
{
public:
  moleculeTest(const char* name, bool atomsOnly = false)
    : svtkRTTest(name)
  {
    this->AtomsOnly = atomsOnly;
  }

  const char* GetSummaryResultName() override
  {
    return this->AtomsOnly ? "Atoms/sec" : "Atoms+Bonds/sec";
  }

  const char* GetSecondSummaryResultName() override
  {
    return this->AtomsOnly ? "Atoms" : "Atoms+Bonds";
  }

  svtkRTTestResult Run(svtkRTTestSequence* ats, int /*argc*/, char* /* argv */[]) override
  {
    int res1;
    ats->GetSequenceNumbers(res1);

    svtkNew<svtkBoxMuellerRandomSequence> rs;
    svtkNew<svtkMolecule> mol;
    svtkNew<svtkPointLocator> pl;

    // build a molecule
    float scale = 3.0 * pow(static_cast<double>(res1), 0.33);
    double pos[3];
    svtkNew<svtkPolyData> pointSet;
    svtkNew<svtkPoints> pts;
    pointSet->SetPoints(pts.GetPointer());
    double bounds[6];
    bounds[0] = 0.0;
    bounds[2] = 0.0;
    bounds[4] = 0.0;
    bounds[1] = scale;
    bounds[3] = scale;
    bounds[5] = scale;
    pl->SetDataSet(pointSet.GetPointer());
    pl->InitPointInsertion(pointSet->GetPoints(), bounds, 10 * res1);
    for (int i = 0; i < res1 * 100; i++)
    {
      pos[0] = scale * rs->GetValue();
      rs->Next();
      pos[1] = scale * rs->GetValue();
      rs->Next();
      pos[2] = scale * rs->GetValue();
      rs->Next();
      pl->InsertPoint(i, pos);
      int molType = i % 9 > 5 ? i % 9 : 1; // a lot of H, some N O CA
      mol->AppendAtom(molType, pos[0], pos[1], pos[2]);
    }

    // now add some bonds
    if (!this->AtomsOnly)
    {
      svtkNew<svtkIdList> ids;
      int bondCount = 0;
      while (bondCount < res1 * 60)
      {
        pos[0] = scale * rs->GetValue();
        rs->Next();
        pos[1] = scale * rs->GetValue();
        rs->Next();
        pos[2] = scale * rs->GetValue();
        rs->Next();
        pl->FindClosestNPoints(2, pos, ids.GetPointer());
        // are the atoms close enough?
        if (svtkMath::Distance2BetweenPoints(mol->GetAtomPosition(ids->GetId(0)).GetData(),
              mol->GetAtomPosition(ids->GetId(1)).GetData()) < 4.0)
        {
          int bondType = bondCount % 10 == 9 ? 3 : (bondCount % 10) / 7 + 1;
          mol->AppendBond(ids->GetId(0), ids->GetId(1), bondType);
          bondCount++;
        }
      }
    }

    svtkNew<svtkMoleculeMapper> mapper;
    mapper->SetInputData(mol.GetPointer());
    mapper->UseBallAndStickSettings();

    svtkNew<svtkActor> actor;
    actor->SetMapper(mapper.GetPointer());

    // create a rendering window and renderer
    svtkNew<svtkRenderer> ren1;
    svtkNew<svtkRenderWindow> renWindow;
    renWindow->AddRenderer(ren1.GetPointer());
    ren1->AddActor(actor.GetPointer());

    // set the size/color of our window
    renWindow->SetSize(this->GetRenderWidth(), this->GetRenderHeight());
    ren1->SetBackground(0.2, 0.3, 0.5);

    // draw the resulting scene
    double startTime = svtkTimerLog::GetUniversalTime();
    renWindow->Render();
    double firstFrameTime = svtkTimerLog::GetUniversalTime() - startTime;
    ren1->GetActiveCamera()->Zoom(1.5);

    int frameCount = 80;
    for (int i = 0; i < frameCount; i++)
    {
      renWindow->Render();
      ren1->GetActiveCamera()->Azimuth(0.5);
      ren1->GetActiveCamera()->Elevation(0.5);
      ren1->GetActiveCamera()->Zoom(1.01);
      // ren1->ResetCameraClippingRange();
      if ((svtkTimerLog::GetUniversalTime() - startTime - firstFrameTime) > this->TargetTime * 1.5)
      {
        frameCount = i + 1;
        break;
      }
    }
    double subsequentFrameTime =
      (svtkTimerLog::GetUniversalTime() - startTime - firstFrameTime) / frameCount;
    double numAtoms = mol->GetNumberOfAtoms();

    svtkRTTestResult result;
    result.Results["first frame time"] = firstFrameTime;
    result.Results["subsequent frame time"] = subsequentFrameTime;
    result.Results["Atoms"] = numAtoms;
    result.Results["Bonds"] = mol->GetNumberOfBonds();
    result.Results["Atoms+Bonds"] = (numAtoms + mol->GetNumberOfBonds());
    result.Results["Atoms+Bonds/sec"] = (numAtoms + mol->GetNumberOfBonds()) / subsequentFrameTime;
    result.Results["Atoms/sec"] = numAtoms / subsequentFrameTime;

    return result;
  }

protected:
  bool AtomsOnly;
};

/*=========================================================================
Define a test for volume rendering
=========================================================================*/
#include "svtkColorTransferFunction.h"
#include "svtkGPUVolumeRayCastMapper.h"
#include "svtkPiecewiseFunction.h"
#include "svtkRTAnalyticSource.h"
#include "svtkVolume.h"
#include "svtkVolumeMapper.h"
#include "svtkVolumeProperty.h"

class volumeTest : public svtkRTTest
{
public:
  volumeTest(const char* name, bool withShading)
    : svtkRTTest(name)
  {
    this->WithShading = withShading;
  }

  const char* GetSummaryResultName() override { return "Mvoxels/sec"; }

  const char* GetSecondSummaryResultName() override { return "Mvoxels"; }

  svtkRTTestResult Run(svtkRTTestSequence* ats, int /*argc*/, char* /* argv */[]) override
  {
    int res1, res2, res3;
    ats->GetSequenceNumbers(res1, res2, res3);

    svtkNew<svtkRTAnalyticSource> wavelet;
    wavelet->SetWholeExtent(
      -50 * res1 - 1, 50 * res1, -50 * res2 - 1, 50 * res2, -50 * res3 - 1, 50 * res3);
    wavelet->Update();

    svtkNew<svtkGPUVolumeRayCastMapper> volumeMapper;
    volumeMapper->SetInputConnection(wavelet->GetOutputPort());
    volumeMapper->AutoAdjustSampleDistancesOff();
    volumeMapper->SetSampleDistance(0.9);

    svtkNew<svtkVolumeProperty> volumeProperty;
    svtkNew<svtkColorTransferFunction> ctf;
    ctf->AddRGBPoint(33.34, 0.23, 0.3, 0.75);
    ctf->AddRGBPoint(72.27, 0.79, 0.05, 0.22);
    ctf->AddRGBPoint(110.3, 0.8, 0.75, 0.82);
    ctf->AddRGBPoint(134.19, 0.78, 0.84, 0.04);
    ctf->AddRGBPoint(159.84, 0.07, 0.87, 0.43);
    ctf->AddRGBPoint(181.96, 0.84, 0.31, 0.48);
    ctf->AddRGBPoint(213.803, 0.73, 0.62, 0.8);
    ctf->AddRGBPoint(255.38, 0.75, 0.19, 0.05);
    ctf->AddRGBPoint(286.33, 0.7, 0.02, 0.15);
    ctf->SetColorSpaceToHSV();

    svtkNew<svtkPiecewiseFunction> pwf;
    pwf->AddPoint(33.35, 0.0);
    pwf->AddPoint(81.99, 0.01);
    pwf->AddPoint(128.88, 0.02);
    pwf->AddPoint(180.19, 0.03);
    pwf->AddPoint(209.38, 0.04);
    pwf->AddPoint(286.33, 0.05);

    volumeProperty->SetColor(ctf.GetPointer());
    volumeProperty->SetScalarOpacity(pwf.GetPointer());

    svtkNew<svtkVolume> volume;
    volume->SetMapper(volumeMapper.GetPointer());
    volume->SetProperty(volumeProperty.GetPointer());
    if (this->WithShading)
    {
      volumeProperty->ShadeOn();
    }

    // create a rendering window and renderer
    svtkNew<svtkRenderer> ren1;
    svtkNew<svtkRenderWindow> renWindow;
    renWindow->AddRenderer(ren1.GetPointer());
    ren1->AddActor(volume.GetPointer());

    // set the size/color of our window
    renWindow->SetSize(this->GetRenderWidth(), this->GetRenderHeight());
    ren1->SetBackground(0.2, 0.3, 0.4);

    // draw the resulting scene
    double startTime = svtkTimerLog::GetUniversalTime();
    renWindow->Render();
    double firstFrameTime = svtkTimerLog::GetUniversalTime() - startTime;
    ren1->GetActiveCamera()->Zoom(1.2);
    ren1->ResetCameraClippingRange();

    int frameCount = 80;
    for (int i = 0; i < frameCount; i++)
    {
      renWindow->Render();
      ren1->GetActiveCamera()->Azimuth(0.5);
      ren1->GetActiveCamera()->Elevation(0.5);
      ren1->ResetCameraClippingRange();
      if ((svtkTimerLog::GetUniversalTime() - startTime - firstFrameTime) > this->TargetTime * 1.5)
      {
        frameCount = i + 1;
        break;
      }
    }
    double subsequentFrameTime =
      (svtkTimerLog::GetUniversalTime() - startTime - firstFrameTime) / frameCount;

    svtkRTTestResult result;
    result.Results["first frame time"] = firstFrameTime;
    result.Results["subsequent frame time"] = subsequentFrameTime;
    result.Results["Mvoxels/sec"] = static_cast<double>(res1 * res2 * res3) / subsequentFrameTime;
    result.Results["Mvoxels"] = res1 * res2 * res3;

    return result;
  }

protected:
  bool WithShading;
};

/*=========================================================================
Define a test for depth peeling transluscent geometry.
=========================================================================*/
#include "svtkParametricFunctionSource.h"
#include "svtkParametricTorus.h"
#include "svtkProperty.h"
#include "svtkTransform.h"

class depthPeelingTest : public svtkRTTest
{
public:
  depthPeelingTest(const char* name, bool withNormals)
    : svtkRTTest(name)
    , WithNormals(withNormals)
  {
  }

  const char* GetSummaryResultName() override { return "subsequent frame time"; }

  const char* GetSecondSummaryResultName() override { return "first frame time"; }

  svtkRTTestResult Run(svtkRTTestSequence* ats, int /*argc*/, char* /* argv */[]) override
  {
    int ures, vres;
    ats->GetSequenceNumbers(ures, vres);

    // ------------------------------------------------------------
    // Create surface
    // ------------------------------------------------------------
    svtkNew<svtkParametricTorus> PB;
    svtkNew<svtkParametricFunctionSource> PFS;
    PFS->SetParametricFunction(PB.Get());
    if (this->WithNormals == false)
    {
      PFS->GenerateNormalsOff();
    }
    PFS->SetUResolution(ures * 50);
    PFS->SetVResolution(vres * 100);
    PFS->Update();

    svtkNew<svtkPolyDataMapper> mapper;
    mapper->SetInputConnection(PFS->GetOutputPort());
    mapper->SetScalarRange(0.0, 360.0);

    // create a rendering window and renderer
    svtkNew<svtkRenderer> ren1;
    svtkNew<svtkRenderWindow> renWindow;
    renWindow->SetMultiSamples(0);
    renWindow->SetAlphaBitPlanes(1);
    renWindow->AddRenderer(ren1.Get());

    // Setup depth peeling to render an exact scene:
    ren1->UseDepthPeelingOn();
    ren1->SetMaximumNumberOfPeels(100);
    ren1->SetOcclusionRatio(0.);

    // Create a set of 10 colored translucent actors at slight offsets:
    const int NUM_ACTORS = 10;
    const unsigned char colors[NUM_ACTORS][4] = {
      { 255, 0, 0, 32 },
      { 0, 255, 0, 32 },
      { 0, 0, 255, 32 },
      { 128, 128, 0, 32 },
      { 0, 128, 128, 32 },
      { 128, 0, 128, 32 },
      { 128, 64, 64, 32 },
      { 64, 128, 64, 32 },
      { 64, 64, 128, 32 },
      { 64, 64, 64, 32 },
    };

    for (int i = 0; i < NUM_ACTORS; ++i)
    {
      svtkNew<svtkActor> actor;
      actor->SetMapper(mapper.Get());
      actor->GetProperty()->SetColor(colors[i][0] / 255., colors[i][1] / 255., colors[i][2] / 255.);
      actor->GetProperty()->SetOpacity(colors[i][3] / 255.);

      svtkNew<svtkTransform> xform;
      xform->Identity();
      xform->RotateX(i * (180. / static_cast<double>(NUM_ACTORS)));
      actor->SetUserTransform(xform.Get());

      ren1->AddActor(actor.Get());
    }

    // set the size/color of our window
    renWindow->SetSize(this->GetRenderWidth(), this->GetRenderHeight());
    ren1->SetBackground(0.2, 0.3, 0.5);

    // draw the resulting scene
    double startTime = svtkTimerLog::GetUniversalTime();
    renWindow->Render();
    double firstFrameTime = svtkTimerLog::GetUniversalTime() - startTime;
    ren1->GetActiveCamera()->Azimuth(90);
    ren1->ResetCameraClippingRange();

    int frameCount = 80;
    for (int i = 0; i < frameCount; i++)
    {
      renWindow->Render();
      ren1->GetActiveCamera()->Azimuth(1);
      ren1->GetActiveCamera()->Elevation(1);
      if ((svtkTimerLog::GetUniversalTime() - startTime - firstFrameTime) > this->TargetTime * 1.5)
      {
        frameCount = i + 1;
        break;
      }
    }
    double subsequentFrameTime =
      (svtkTimerLog::GetUniversalTime() - startTime - firstFrameTime) / frameCount;
    double numTris = PFS->GetOutput()->GetPolys()->GetNumberOfCells();
    numTris *= NUM_ACTORS;

    svtkRTTestResult result;
    result.Results["first frame time"] = firstFrameTime;
    result.Results["subsequent frame time"] = subsequentFrameTime;
    result.Results["FPS"] = 1. / subsequentFrameTime;
    result.Results["triangles"] = numTris;

    return result;
  }

protected:
  bool WithNormals;
};

/*=========================================================================
Define a test for simple triangle mesh surfaces
=========================================================================*/
#include "svtkParametricBoy.h"
#include "svtkParametricFunctionSource.h"
#include "svtkParametricTorus.h"

class manyActorTest : public svtkRTTest
{
public:
  manyActorTest(const char* name)
    : svtkRTTest(name)
  {
  }

  const char* GetSummaryResultName() override { return "actors"; }

  const char* GetSecondSummaryResultName() override { return "frames/sec"; }

  svtkRTTestResult Run(svtkRTTestSequence* ats, int /*argc*/, char* /* argv */[]) override
  {
    int ures, vres;
    ats->GetSequenceNumbers(ures, vres);

    // ------------------------------------------------------------
    // Create surface
    // ------------------------------------------------------------
    //  svtkNew<svtkParametricBoy> PB;
    svtkNew<svtkParametricTorus> PB;
    svtkNew<svtkParametricFunctionSource> PFS;
    PFS->SetParametricFunction(PB.Get());
    //  PFS->SetScalarModeToPhase();
    PFS->SetScalarModeToNone();
    //  PFS->GenerateNormalsOff();
    PFS->SetUResolution(10);
    PFS->SetVResolution(20);
    PFS->Update();

    // create a rendering window and renderer
    svtkNew<svtkRenderer> ren1;
    // ren1->RemoveCuller(ren1->GetCullers()->GetLastItem());
    svtkNew<svtkRenderWindow> renWindow;
    renWindow->AddRenderer(ren1.Get());

    // create many actors
    for (int u = 0; u < ures * 10; ++u)
    {
      for (int v = 0; v < vres * 10; ++v)
      {
        svtkNew<svtkPolyDataMapper> mapper;
        mapper->SetInputConnection(PFS->GetOutputPort());
        mapper->SetScalarRange(0.0, 360.0);
        mapper->StaticOn();

        svtkNew<svtkActor> actor;
        actor->SetMapper(mapper.Get());
        actor->ForceOpaqueOn();
        ren1->AddActor(actor.Get());
      }
    }

    // set the size/color of our window
    renWindow->SetSize(this->GetRenderWidth(), this->GetRenderHeight());
    ren1->SetBackground(0.2, 0.3, 0.5);

    // draw the resulting scene
    double startTime = svtkTimerLog::GetUniversalTime();
    renWindow->Render();
    double firstFrameTime = svtkTimerLog::GetUniversalTime() - startTime;
    ren1->GetActiveCamera()->Azimuth(90);
    ren1->GetActiveCamera()->Zoom(0.3);
    ren1->ResetCameraClippingRange();

    int frameCount = 80;
    for (int i = 0; i < frameCount; i++)
    {
      renWindow->Render();
      ren1->GetActiveCamera()->Azimuth(1);
      ren1->GetActiveCamera()->Elevation(1);
      if ((svtkTimerLog::GetUniversalTime() - startTime - firstFrameTime) > this->TargetTime * 1.5)
      {
        frameCount = i + 1;
        break;
      }
    }
    double subsequentFrameTime =
      (svtkTimerLog::GetUniversalTime() - startTime - firstFrameTime) / frameCount;

    svtkRTTestResult result;
    result.Results["first frame time"] = firstFrameTime;
    result.Results["subsequent frame time"] = subsequentFrameTime;
    result.Results["frames/sec"] = 1.0 / subsequentFrameTime;
    result.Results["actors"] = 100 * ures * vres;

    return result;
  }

protected:
};

#endif
// SVTK-HeaderTest-Exclude: svtkRenderTimingTests.h
