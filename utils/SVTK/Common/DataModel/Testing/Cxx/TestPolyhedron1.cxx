/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestPolyhedron1.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkActor.h"
#include "svtkCellArray.h"
#include "svtkCellData.h"
#include "svtkDataArray.h"
#include "svtkDataSetMapper.h"
#include "svtkDoubleArray.h"
#include "svtkExtractEdges.h"
#include "svtkMath.h"
#include "svtkPlane.h"
#include "svtkPlaneSource.h"
#include "svtkPointData.h"
#include "svtkPointLocator.h"
#include "svtkPoints.h"
#include "svtkPolyhedron.h"
#include "svtkProperty.h"
#include "svtkRenderWindow.h"
#include "svtkRenderWindowInteractor.h"
#include "svtkRenderer.h"
#include "svtkSmartPointer.h"
#include "svtkUnstructuredGrid.h"

#include "svtkRegressionTestImage.h"
#include "svtkTestUtilities.h"

// Test of svtkPolyhedron. A dodecahedron is created for testing clip and contour
int TestPolyhedron1(int argc, char* argv[])
{
  // create a dodecahedron
  double dodechedronPoint[20][3] = {
    { 1.21412, 0, 1.58931 },
    { 0.375185, 1.1547, 1.58931 },
    { -0.982247, 0.713644, 1.58931 },
    { -0.982247, -0.713644, 1.58931 },
    { 0.375185, -1.1547, 1.58931 },
    { 1.96449, 0, 0.375185 },
    { 0.607062, 1.86835, 0.375185 },
    { -1.58931, 1.1547, 0.375185 },
    { -1.58931, -1.1547, 0.375185 },
    { 0.607062, -1.86835, 0.375185 },
    { 1.58931, 1.1547, -0.375185 },
    { -0.607062, 1.86835, -0.375185 },
    { -1.96449, 0, -0.375185 },
    { -0.607062, -1.86835, -0.375185 },
    { 1.58931, -1.1547, -0.375185 },
    { 0.982247, 0.713644, -1.58931 },
    { -0.375185, 1.1547, -1.58931 },
    { -1.21412, 0, -1.58931 },
    { -0.375185, -1.1547, -1.58931 },
    { 0.982247, -0.713644, -1.58931 },
  };
  svtkSmartPointer<svtkPoints> dodechedronPoints = svtkSmartPointer<svtkPoints>::New();
  dodechedronPoints->Initialize();
  for (int i = 0; i < 20; i++)
  {
    dodechedronPoints->InsertNextPoint(dodechedronPoint[i]);
  }

  svtkIdType dodechedronPointsIds[20] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19 };

  svtkIdType dodechedronFace[12][5] = {
    { 0, 1, 2, 3, 4 },
    { 0, 5, 10, 6, 1 },
    { 1, 6, 11, 7, 2 },
    { 2, 7, 12, 8, 3 },
    { 3, 8, 13, 9, 4 },
    { 4, 9, 14, 5, 0 },
    { 15, 10, 5, 14, 19 },
    { 16, 11, 6, 10, 15 },
    { 17, 12, 7, 11, 16 },
    { 18, 13, 8, 12, 17 },
    { 19, 14, 9, 13, 18 },
    { 19, 18, 17, 16, 15 },
  };

  svtkSmartPointer<svtkCellArray> dodechedronFaces = svtkSmartPointer<svtkCellArray>::New();
  for (int i = 0; i < 12; i++)
  {
    dodechedronFaces->InsertNextCell(5, dodechedronFace[i]);
  }

  double offset = 0; // 0.375185;

  double normal[3] = { 0.0, 0.0, 1.0 };
  double origin[3] = { 0.0, 0.0, offset };
  double x[3] = { 1.0, 0.0, 0.0 };
  double y[3] = { 0.0, 1.0, 0.0 };

  svtkSmartPointer<svtkPlaneSource> planeSource = svtkSmartPointer<svtkPlaneSource>::New();
  planeSource->SetNormal(normal);
  planeSource->SetOrigin(origin);
  planeSource->SetPoint1(origin[0] + 5 * x[0], origin[1] + 5 * x[1], origin[2] + 5 * x[2]);
  planeSource->SetPoint2(origin[0] + 7 * y[0], origin[1] + 7 * y[1], origin[2] + 7 * y[2]);
  planeSource->SetCenter(origin);
  planeSource->SetResolution(1, 1);
  planeSource->Update();

  svtkSmartPointer<svtkPlane> plane = svtkSmartPointer<svtkPlane>::New();
  plane->SetNormal(normal);
  plane->SetOrigin(origin);
  svtkSmartPointer<svtkDoubleArray> pointDataArray = svtkSmartPointer<svtkDoubleArray>::New();
  pointDataArray->Initialize();
  for (int i = 0; i < 20; i++)
  {
    cout << plane->EvaluateFunction(dodechedronPoint[i]) << endl;
    pointDataArray->InsertNextValue(plane->EvaluateFunction(dodechedronPoint[i]) + 0.01);
  }

  svtkSmartPointer<svtkDoubleArray> cellDataArray = svtkSmartPointer<svtkDoubleArray>::New();
  cellDataArray->Initialize();
  for (int i = 0; i < 12; i++)
  {
    cellDataArray->InsertNextValue(static_cast<double>(1.0));
  }

  svtkNew<svtkIdTypeArray> legacyFaces;
  dodechedronFaces->ExportLegacyFormat(legacyFaces);

  svtkSmartPointer<svtkUnstructuredGrid> ugrid = svtkSmartPointer<svtkUnstructuredGrid>::New();
  ugrid->SetPoints(dodechedronPoints);
  ugrid->InsertNextCell(SVTK_POLYHEDRON, 20, dodechedronPointsIds, 12, legacyFaces->GetPointer(0));
  ugrid->GetPointData()->SetScalars(pointDataArray);
  // ugrid->GetCellData()->SetScalars(cellDataArray);

  svtkPolyhedron* polyhedron = static_cast<svtkPolyhedron*>(ugrid->GetCell(0));
  svtkPolyData* planePoly = planeSource->GetOutput();
  polyhedron->GetPolyData()->GetPointData()->SetScalars(pointDataArray);
  // polyhedron->GetPolyData()->GetCellData()->SetScalars(cellDataArray);

  // test contour
  svtkSmartPointer<svtkPointLocator> locator = svtkSmartPointer<svtkPointLocator>::New();
  svtkSmartPointer<svtkCellArray> resultPolys = svtkSmartPointer<svtkCellArray>::New();
  svtkSmartPointer<svtkPointData> resultPd = svtkSmartPointer<svtkPointData>::New();
  svtkSmartPointer<svtkCellData> resultCd = svtkSmartPointer<svtkCellData>::New();
  svtkSmartPointer<svtkPoints> resultPoints = svtkSmartPointer<svtkPoints>::New();
  resultPoints->DeepCopy(ugrid->GetPoints());
  locator->InitPointInsertion(resultPoints, ugrid->GetBounds());

  polyhedron->Contour(0, ugrid->GetPointData()->GetScalars(), locator, nullptr, nullptr,
    resultPolys, ugrid->GetPointData(), resultPd, ugrid->GetCellData(), 0, resultCd);

  // output the contour
  svtkSmartPointer<svtkUnstructuredGrid> contourResult = svtkSmartPointer<svtkUnstructuredGrid>::New();
  contourResult->SetPoints(locator->GetPoints());
  contourResult->SetCells(SVTK_POLYGON, resultPolys);
  contourResult->GetPointData()->DeepCopy(resultPd);

  // test clip
  svtkSmartPointer<svtkPointLocator> locator1 = svtkSmartPointer<svtkPointLocator>::New();
  svtkSmartPointer<svtkCellArray> resultPolys1 = svtkSmartPointer<svtkCellArray>::New();
  svtkSmartPointer<svtkPointData> resultPd1 = svtkSmartPointer<svtkPointData>::New();
  svtkSmartPointer<svtkCellData> resultCd1 = svtkSmartPointer<svtkCellData>::New();
  svtkSmartPointer<svtkPoints> resultPoints1 = svtkSmartPointer<svtkPoints>::New();
  resultPoints1->DeepCopy(ugrid->GetPoints());
  locator1->InitPointInsertion(resultPoints1, ugrid->GetBounds());

  polyhedron->Clip(0, ugrid->GetPointData()->GetScalars(), locator1, resultPolys1,
    ugrid->GetPointData(), resultPd1, ugrid->GetCellData(), 0, resultCd1, 1);

  // output the clipped polyhedron
  svtkSmartPointer<svtkUnstructuredGrid> clipResult = svtkSmartPointer<svtkUnstructuredGrid>::New();
  clipResult->SetPoints(locator1->GetPoints());
  clipResult->SetCells(SVTK_POLYHEDRON, resultPolys1);
  clipResult->GetPointData()->DeepCopy(resultPd1);

  // create actors
  svtkSmartPointer<svtkDataSetMapper> mapper = svtkSmartPointer<svtkDataSetMapper>::New();
  mapper->SetInputData(polyhedron->GetPolyData());

  svtkSmartPointer<svtkActor> actor = svtkSmartPointer<svtkActor>::New();
  actor->SetMapper(mapper);

  svtkSmartPointer<svtkDataSetMapper> planeMapper = svtkSmartPointer<svtkDataSetMapper>::New();
  planeMapper->SetInputData(planePoly);

  svtkSmartPointer<svtkActor> planeActor = svtkSmartPointer<svtkActor>::New();
  planeActor->SetMapper(planeMapper);

  svtkSmartPointer<svtkDataSetMapper> contourMapper = svtkSmartPointer<svtkDataSetMapper>::New();
  contourMapper->SetInputData(contourResult);

  svtkSmartPointer<svtkActor> contourActor = svtkSmartPointer<svtkActor>::New();
  contourActor->SetMapper(contourMapper);

  svtkSmartPointer<svtkDataSetMapper> clipPolyhedronMapper = svtkSmartPointer<svtkDataSetMapper>::New();
  clipPolyhedronMapper->SetInputData(clipResult);

  svtkSmartPointer<svtkActor> clipPolyhedronActor = svtkSmartPointer<svtkActor>::New();
  clipPolyhedronActor->SetMapper(clipPolyhedronMapper);

  // Create rendering infrastructure
  svtkSmartPointer<svtkProperty> prop = svtkSmartPointer<svtkProperty>::New();
  prop->LightingOff();
  prop->SetRepresentationToSurface();
  prop->EdgeVisibilityOn();
  prop->SetLineWidth(3.0);
  prop->SetOpacity(1.0);
  prop->SetInterpolationToFlat();

  svtkSmartPointer<svtkProperty> prop1 = svtkSmartPointer<svtkProperty>::New();
  prop1->LightingOff();
  prop1->SetRepresentationToSurface();
  prop1->EdgeVisibilityOn();
  prop1->SetLineWidth(3.0);
  prop1->SetOpacity(0.5);
  prop1->SetInterpolationToFlat();

  // set property
  actor->SetProperty(prop1);
  planeActor->SetProperty(prop1);
  contourActor->SetProperty(prop1);
  clipPolyhedronActor->SetProperty(prop);

  svtkSmartPointer<svtkRenderer> ren = svtkSmartPointer<svtkRenderer>::New();
  ren->AddActor(actor);
  ren->AddActor(planeActor);
  ren->AddActor(contourActor);
  ren->AddActor(clipPolyhedronActor);
  ren->SetBackground(.5, .5, .5);

  svtkSmartPointer<svtkRenderWindow> renWin = svtkSmartPointer<svtkRenderWindow>::New();
  renWin->SetMultiSamples(0);
  renWin->AddRenderer(ren);

  svtkSmartPointer<svtkRenderWindowInteractor> iren =
    svtkSmartPointer<svtkRenderWindowInteractor>::New();
  iren->SetRenderWindow(renWin);

  iren->Initialize();

  renWin->Render();

  int retVal = svtkRegressionTestImage(renWin);
  if (retVal == svtkRegressionTester::DO_INTERACTOR)
  {
    iren->Start();
  }

  return !retVal;
}
