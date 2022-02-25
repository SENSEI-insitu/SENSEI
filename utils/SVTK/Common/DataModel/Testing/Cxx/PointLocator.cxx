/*=========================================================================

  Program:   Visualization Toolkit
  Module:    PointLocator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkActor.h"
#include "svtkPointLocator.h"
#include "svtkPolyData.h"
#include "svtkPolyDataMapper.h"
#include "svtkProperty.h"
#include "svtkRenderWindow.h"
#include "svtkRenderWindowInteractor.h"
#include "svtkRenderer.h"
#include "svtkSphereSource.h"

#include "svtkDebugLeaks.h"
#include "svtkRegressionTestImage.h"

int PointLocator(int argc, char* argv[])
{
  svtkRenderer* renderer = svtkRenderer::New();
  svtkRenderWindow* renWin = svtkRenderWindow::New();
  renWin->AddRenderer(renderer);
  svtkRenderWindowInteractor* iren = svtkRenderWindowInteractor::New();
  iren->SetRenderWindow(renWin);

  svtkSphereSource* sphere = svtkSphereSource::New();
  sphere->SetThetaResolution(8);
  sphere->SetPhiResolution(8);
  sphere->SetRadius(1.0);
  sphere->Update();
  svtkPolyDataMapper* sphereMapper = svtkPolyDataMapper::New();
  sphereMapper->SetInputConnection(sphere->GetOutputPort());
  svtkActor* sphereActor = svtkActor::New();
  sphereActor->SetMapper(sphereMapper);

  svtkSphereSource* spot = svtkSphereSource::New();
  spot->SetPhiResolution(6);
  spot->SetThetaResolution(6);
  spot->SetRadius(0.1);

  svtkPolyDataMapper* spotMapper = svtkPolyDataMapper::New();
  spotMapper->SetInputConnection(spot->GetOutputPort());

  // Build a locator
  svtkPointLocator* pointLocator = svtkPointLocator::New();
  pointLocator->SetDataSet(sphere->GetOutput());
  pointLocator->BuildLocator();

  //
  double p1[] = { 2.0, 1.0, 3.0 };

  // Find closest point
  svtkIdType ptId;
  double dist;
  p1[0] = 0.1;
  p1[1] = -0.2;
  p1[2] = 0.2;
  ptId = pointLocator->FindClosestPoint(p1);
  svtkActor* closestPointActor = svtkActor::New();
  closestPointActor->SetMapper(spotMapper);
  // TODO cleanupo
  closestPointActor->SetPosition(sphere->GetOutput()->GetPoints()->GetPoint(ptId)[0],
    sphere->GetOutput()->GetPoints()->GetPoint(ptId)[1],
    sphere->GetOutput()->GetPoints()->GetPoint(ptId)[2]);
  closestPointActor->GetProperty()->SetColor(0.0, 1.0, 0.0);

  // Find closest point within radius
  float radius = 5.0;
  p1[0] = .2;
  p1[1] = 1.0;
  p1[2] = 1.0;
  ptId = pointLocator->FindClosestPointWithinRadius(radius, p1, dist);
  svtkActor* closestPointActor2 = svtkActor::New();
  closestPointActor2->SetMapper(spotMapper);
  // TODO cleanup
  closestPointActor2->SetPosition(sphere->GetOutput()->GetPoints()->GetPoint(ptId)[0],
    sphere->GetOutput()->GetPoints()->GetPoint(ptId)[1],
    sphere->GetOutput()->GetPoints()->GetPoint(ptId)[2]);
  closestPointActor2->GetProperty()->SetColor(0.0, 1.0, 0.0);

  renderer->AddActor(sphereActor);
  renderer->AddActor(closestPointActor);
  renderer->AddActor(closestPointActor2);
  renderer->SetBackground(1, 1, 1);
  renWin->SetSize(300, 300);

  // interact with data
  renWin->Render();

  int retVal = svtkRegressionTestImage(renWin);
  if (retVal == svtkRegressionTester::DO_INTERACTOR)
  {
    iren->Start();
  }

  // Clean up
  renderer->Delete();
  renWin->Delete();
  iren->Delete();
  sphere->Delete();
  sphereMapper->Delete();
  sphereActor->Delete();
  spot->Delete();
  spotMapper->Delete();
  closestPointActor->Delete();
  closestPointActor2->Delete();
  pointLocator->FreeSearchStructure();
  pointLocator->Delete();

  return !retVal;
}
