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
#include "svtkDebugLeaks.h"
#include "svtkDoubleArray.h"
#include "svtkLight.h"
#include "svtkLightCollection.h"
#include "svtkMath.h"
#include "svtkPlane.h"
#include "svtkPlaneSource.h"
#include "svtkPointData.h"
#include "svtkPolyData.h"
#include "svtkPolyDataMapper.h"
#include "svtkPolygon.h"
#include "svtkProbeFilter.h"
#include "svtkProperty.h"
#include "svtkRegressionTestImage.h"
#include "svtkRenderWindow.h"
#include "svtkRenderWindowInteractor.h"
#include "svtkRenderer.h"
#include "svtkSmartPointer.h"
#include "svtkUnstructuredGrid.h"

// Test MVC interpolation of polygon cell
int TestMeanValueCoordinatesInterpolation2(int argc, char* argv[])
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
  // Case 0: convex pentagon
  //
  // create a regular pentagon
  double pentagon[5][3];
  for (int i = 0; i < 5; i++)
  {
    pentagon[i][0] = sin(svtkMath::RadiansFromDegrees(72.0 * i));
    pentagon[i][1] = cos(svtkMath::RadiansFromDegrees(72.0 * i));
    pentagon[i][2] = 0.0;
  }

  svtkSmartPointer<svtkCellArray> pentagonCell = svtkSmartPointer<svtkCellArray>::New();
  pentagonCell->InsertNextCell(5);
  for (svtkIdType i = 0; i < 5; i++)
  {
    pentagonCell->InsertCellPoint(i);
  }

  svtkSmartPointer<svtkPoints> pentagonPoints = svtkSmartPointer<svtkPoints>::New();
  pentagonPoints->Initialize();
  for (int i = 0; i < 5; i++)
  {
    pentagonPoints->InsertNextPoint(pentagon[i]);
  }

  svtkSmartPointer<svtkDoubleArray> pointDataArray = svtkSmartPointer<svtkDoubleArray>::New();
  pointDataArray->Initialize();
  for (int i = 0; i < 5; i++)
  {
    pointDataArray->InsertNextValue((pentagon[i][0] + 1.0) / 2.0);
  }

  svtkSmartPointer<svtkPolyData> polydata = svtkSmartPointer<svtkPolyData>::New();
  polydata->SetPoints(pentagonPoints);
  polydata->SetPolys(pentagonCell);
  polydata->GetPointData()->SetScalars(pointDataArray);

  svtkPolygon* polygon = static_cast<svtkPolygon*>(polydata->GetCell(0));
  polygon->SetUseMVCInterpolation(1);

  // Okay now sample on a plane and see how it interpolates
  svtkSmartPointer<svtkPlaneSource> pSource = svtkSmartPointer<svtkPlaneSource>::New();
  pSource->SetOrigin(-1.0, -1.0, 0);
  pSource->SetPoint1(1.0, -1.0, 0);
  pSource->SetPoint2(-1.0, 1.0, 0);
  pSource->SetXResolution(100);
  pSource->SetYResolution(100);

  // mvc interpolation
  svtkSmartPointer<svtkProbeFilter> interp = svtkSmartPointer<svtkProbeFilter>::New();
  interp->SetInputConnection(pSource->GetOutputPort());
  interp->SetSourceData(polydata);

  svtkSmartPointer<svtkPolyDataMapper> interpMapper = svtkSmartPointer<svtkPolyDataMapper>::New();
  interpMapper->SetInputConnection(interp->GetOutputPort());

  svtkSmartPointer<svtkActor> interpActor = svtkSmartPointer<svtkActor>::New();
  interpActor->SetMapper(interpMapper);

  //
  // Case 1: convex polygon meshes
  //
  pentagon[0][0] = 0.0;
  pentagon[0][1] = 0.0;
  pentagon[0][2] = 0.0;

  svtkSmartPointer<svtkPoints> pentagonPoints1 = svtkSmartPointer<svtkPoints>::New();
  pentagonPoints1->Initialize();
  for (int i = 0; i < 5; i++)
  {
    pentagonPoints1->InsertNextPoint(pentagon[i]);
  }

  svtkSmartPointer<svtkCellArray> pentagonCell1 = svtkSmartPointer<svtkCellArray>::New();
  pentagonCell1->InsertNextCell(5);
  for (svtkIdType i = 0; i < 5; i++)
  {
    pentagonCell1->InsertCellPoint(i);
  }

  svtkSmartPointer<svtkDoubleArray> pointDataArray1 = svtkSmartPointer<svtkDoubleArray>::New();
  pointDataArray1->Initialize();
  for (int i = 0; i < 5; i++)
  {
    pointDataArray1->InsertNextValue((pentagon[i][0] + 1.0) / 2.0);
  }

  svtkSmartPointer<svtkPolyData> polydata1 = svtkSmartPointer<svtkPolyData>::New();
  polydata1->SetPoints(pentagonPoints1);
  polydata1->SetPolys(pentagonCell1);
  polydata1->GetPointData()->SetScalars(pointDataArray1);

  svtkPolygon* polygon1 = static_cast<svtkPolygon*>(polydata1->GetCell(0));
  polygon1->SetUseMVCInterpolation(1);

  // Okay now sample on a plane and see how it interpolates
  svtkSmartPointer<svtkPlaneSource> pSource1 = svtkSmartPointer<svtkPlaneSource>::New();
  pSource1->SetOrigin(-1.0, -1.0, 0);
  pSource1->SetPoint1(1.0, -1.0, 0);
  pSource1->SetPoint2(-1.0, 1.0, 0);
  pSource1->SetXResolution(100);
  pSource1->SetYResolution(100);

  // interpolation 1: use the more general but slower MVC algorithm.
  svtkSmartPointer<svtkProbeFilter> interp1 = svtkSmartPointer<svtkProbeFilter>::New();
  interp1->SetInputConnection(pSource1->GetOutputPort());
  interp1->SetSourceData(polydata1);

  svtkSmartPointer<svtkPolyDataMapper> interpMapper1 = svtkSmartPointer<svtkPolyDataMapper>::New();
  interpMapper1->SetInputConnection(interp1->GetOutputPort());

  svtkSmartPointer<svtkActor> interpActor1 = svtkSmartPointer<svtkActor>::New();
  interpActor1->SetMapper(interpMapper1);

  //
  // add actors to renderer
  //
  svtkSmartPointer<svtkProperty> lightProperty = svtkSmartPointer<svtkProperty>::New();
  lightProperty->LightingOff();
  interpActor->SetProperty(lightProperty);
  interpActor1->SetProperty(lightProperty);

  renderer->AddActor(interpActor);
  renderer->ResetCamera();
  renderer->SetBackground(1, 1, 1);

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
