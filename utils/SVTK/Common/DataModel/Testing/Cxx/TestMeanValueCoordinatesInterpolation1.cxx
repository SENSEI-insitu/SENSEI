/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestMeanValueCoordinatesInterpolation.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkActor.h"
#include "svtkCamera.h"
#include "svtkCellArray.h"
#include "svtkClipPolyData.h"
#include "svtkDebugLeaks.h"
#include "svtkElevationFilter.h"
#include "svtkIdTypeArray.h"
#include "svtkLight.h"
#include "svtkLightCollection.h"
#include "svtkNew.h"
#include "svtkPlane.h"
#include "svtkPlaneSource.h"
#include "svtkPolyData.h"
#include "svtkPolyDataMapper.h"
#include "svtkProbePolyhedron.h"
#include "svtkProperty.h"
#include "svtkRegressionTestImage.h"
#include "svtkRenderWindow.h"
#include "svtkRenderWindowInteractor.h"
#include "svtkRenderer.h"
#include "svtkSmartPointer.h"
#include "svtkSphereSource.h"

int TestMeanValueCoordinatesInterpolation1(int argc, char* argv[])
{
  svtkSmartPointer<svtkRenderer> renderer = svtkSmartPointer<svtkRenderer>::New();
  svtkSmartPointer<svtkRenderer> renderer1 = svtkSmartPointer<svtkRenderer>::New();
  renderer->SetViewport(0, 0, 0.5, 1);

  svtkSmartPointer<svtkRenderWindow> renWin = svtkSmartPointer<svtkRenderWindow>::New();
  renWin->SetMultiSamples(0);
  renWin->AddRenderer(renderer);
  renWin->AddRenderer(renderer1);
  renderer1->SetViewport(0.5, 0, 1, 1);

  svtkSmartPointer<svtkRenderWindowInteractor> iren =
    svtkSmartPointer<svtkRenderWindowInteractor>::New();
  iren->SetRenderWindow(renWin);

  //
  // Case 0: triangle meshes
  //
  // Create a sphere
  svtkSmartPointer<svtkSphereSource> sphere = svtkSmartPointer<svtkSphereSource>::New();
  sphere->SetThetaResolution(51);
  sphere->SetPhiResolution(17);

  // Generate some scalars on the sphere
  svtkSmartPointer<svtkElevationFilter> ele = svtkSmartPointer<svtkElevationFilter>::New();
  ele->SetInputConnection(sphere->GetOutputPort());
  ele->SetLowPoint(-0.5, 0, 0);
  ele->SetHighPoint(0.5, 0, 0);
  ele->Update();

  // Now clip the sphere in half and display it
  svtkSmartPointer<svtkPlane> plane = svtkSmartPointer<svtkPlane>::New();
  plane->SetOrigin(0, 0, 0);
  plane->SetNormal(0, 0, 1);

  svtkSmartPointer<svtkClipPolyData> clip = svtkSmartPointer<svtkClipPolyData>::New();
  clip->SetInputConnection(ele->GetOutputPort());
  clip->SetClipFunction(plane);

  svtkSmartPointer<svtkPolyDataMapper> sphereMapper = svtkSmartPointer<svtkPolyDataMapper>::New();
  sphereMapper->SetInputConnection(clip->GetOutputPort());

  svtkSmartPointer<svtkActor> sphereActor = svtkSmartPointer<svtkActor>::New();
  sphereActor->SetMapper(sphereMapper);

  // Okay now sample the sphere mesh with a plane and see how it interpolates
  svtkSmartPointer<svtkPlaneSource> pSource = svtkSmartPointer<svtkPlaneSource>::New();
  pSource->SetOrigin(-1.0, -1.0, 0);
  pSource->SetPoint1(1.0, -1.0, 0);
  pSource->SetPoint2(-1.0, 1.0, 0);
  pSource->SetXResolution(50);
  pSource->SetYResolution(50);

  // interpolation 0: use the faster MVC algorithm specialized for triangle meshes.
  svtkSmartPointer<svtkProbePolyhedron> interp = svtkSmartPointer<svtkProbePolyhedron>::New();
  interp->SetInputConnection(pSource->GetOutputPort());
  interp->SetSourceConnection(ele->GetOutputPort());

  svtkSmartPointer<svtkPolyDataMapper> interpMapper = svtkSmartPointer<svtkPolyDataMapper>::New();
  interpMapper->SetInputConnection(interp->GetOutputPort());

  svtkSmartPointer<svtkActor> interpActor = svtkSmartPointer<svtkActor>::New();
  interpActor->SetMapper(interpMapper);

  //
  // Case 1: general meshes
  //
  // Create a sphere
  svtkSmartPointer<svtkSphereSource> sphere1 = svtkSmartPointer<svtkSphereSource>::New();
  sphere1->SetThetaResolution(51);
  sphere1->SetPhiResolution(17);

  // Generate some scalars on the sphere
  svtkSmartPointer<svtkElevationFilter> ele1 = svtkSmartPointer<svtkElevationFilter>::New();
  ele1->SetInputConnection(sphere1->GetOutputPort());
  ele1->SetLowPoint(-0.5, 0, 0);
  ele1->SetHighPoint(0.5, 0, 0);
  ele1->Update();

  // create a cell with 4 points
  svtkPolyData* spherePoly = svtkPolyData::SafeDownCast(ele1->GetOutput());
  svtkCellArray* polys = spherePoly->GetPolys();

  // merge the first two cell, this will make svtkProbePolyhedron select the
  // more general MVC algorithm.
  svtkNew<svtkIdTypeArray> legacyArray;
  polys->ExportLegacyFormat(legacyArray);
  svtkIdType* p = legacyArray->GetPointer(0);
  svtkIdType pids[4] = { p[1], p[2], p[6], p[3] };

  svtkSmartPointer<svtkCellArray> newPolys = svtkSmartPointer<svtkCellArray>::New();
  newPolys->Initialize();
  for (int i = 2; i < polys->GetNumberOfCells(); i++)
  {
    pids[0] = p[4 * i + 1];
    pids[1] = p[4 * i + 2];
    pids[2] = p[4 * i + 3];
    newPolys->InsertNextCell(3, pids);
  }
  spherePoly->SetPolys(newPolys);

  // Now clip the sphere in half and display it
  svtkSmartPointer<svtkPlane> plane1 = svtkSmartPointer<svtkPlane>::New();
  plane1->SetOrigin(0, 0, 0);
  plane1->SetNormal(0, 0, 1);

  svtkSmartPointer<svtkClipPolyData> clip1 = svtkSmartPointer<svtkClipPolyData>::New();
  clip1->SetInputData(spherePoly);
  clip1->SetClipFunction(plane1);

  svtkSmartPointer<svtkPolyDataMapper> sphereMapper1 = svtkSmartPointer<svtkPolyDataMapper>::New();
  sphereMapper1->SetInputConnection(clip1->GetOutputPort());

  svtkSmartPointer<svtkActor> sphereActor1 = svtkSmartPointer<svtkActor>::New();
  sphereActor1->SetMapper(sphereMapper1);

  // Okay now sample the sphere mesh with a plane and see how it interpolates
  svtkSmartPointer<svtkPlaneSource> pSource1 = svtkSmartPointer<svtkPlaneSource>::New();
  pSource1->SetOrigin(-1.0, -1.0, 0);
  pSource1->SetPoint1(1.0, -1.0, 0);
  pSource1->SetPoint2(-1.0, 1.0, 0);
  pSource1->SetXResolution(50);
  pSource1->SetYResolution(50);

  // interpolation 1: use the more general but slower MVC algorithm.
  svtkSmartPointer<svtkProbePolyhedron> interp1 = svtkSmartPointer<svtkProbePolyhedron>::New();
  interp1->SetInputConnection(pSource1->GetOutputPort());
  interp1->SetSourceConnection(ele1->GetOutputPort());

  svtkSmartPointer<svtkPolyDataMapper> interpMapper1 = svtkSmartPointer<svtkPolyDataMapper>::New();
  interpMapper1->SetInputConnection(interp1->GetOutputPort());

  svtkSmartPointer<svtkActor> interpActor1 = svtkSmartPointer<svtkActor>::New();
  interpActor1->SetMapper(interpMapper1);

  //
  // add actors to renderer
  //
  svtkSmartPointer<svtkProperty> lightProperty = svtkSmartPointer<svtkProperty>::New();
  lightProperty->LightingOff();
  sphereActor->SetProperty(lightProperty);
  interpActor->SetProperty(lightProperty);
  interpActor1->SetProperty(lightProperty);

  renderer->AddActor(sphereActor);
  renderer->AddActor(interpActor);
  renderer->ResetCamera();
  renderer->SetBackground(1, 1, 1);

  renderer1->AddActor(sphereActor);
  renderer1->AddActor(interpActor1);
  renderer1->ResetCamera();
  renderer1->SetBackground(1, 1, 1);

  renWin->SetSize(600, 300);

  // interact with data
  renWin->Render();

  int retVal = svtkRegressionTestImage(renWin);

  if (retVal == svtkRegressionTester::DO_INTERACTOR)
  {
    iren->Start();
  }

  return !retVal;
}
