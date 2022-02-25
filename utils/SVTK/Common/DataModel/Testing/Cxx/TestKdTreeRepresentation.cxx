/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestKdTreeBoxSelection.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkActor.h"
#include "svtkCamera.h"
#include "svtkGlyph3D.h"
#include "svtkIdTypeArray.h"
#include "svtkInteractorStyleRubberBandPick.h"
#include "svtkKdTree.h"
#include "svtkPoints.h"
#include "svtkPolyData.h"
#include "svtkPolyDataMapper.h"
#include "svtkProperty.h"
#include "svtkRegressionTestImage.h"
#include "svtkRenderWindow.h"
#include "svtkRenderWindowInteractor.h"
#include "svtkRenderer.h"
#include "svtkSmartPointer.h"
#include "svtkSphereSource.h"

#define SVTK_CREATE(type, name) svtkSmartPointer<type> name = svtkSmartPointer<type>::New()

int TestKdTreeFunctions()
{
  int retVal = 0;

  const int num_points = 10;
  double p[num_points][3] = {
    { 0.840188, 0.394383, 0.783099 },
    { 0.79844, 0.911647, 0.197551 },
    { 0.335223, 0.76823, 0.277775 },
    { 0.55397, 0.477397, 0.628871 },
    { 0.364784, 0.513401, 0.95223 },
    { 0.916195, 0.635712, 0.717297 },
    { 0.141603, 0.606969, 0.0163006 },
    { 0.242887, 0.137232, 0.804177 },
    { 0.156679, 0.400944, 0.12979 },
    { 0.108809, 0.998925, 0.218257 },
  };

  svtkSmartPointer<svtkPoints> points = svtkSmartPointer<svtkPoints>::New();
  for (int i = 0; i < num_points; i++)
  {
    points->InsertNextPoint(p[i]);
  }

  svtkSmartPointer<svtkKdTree> kd = svtkSmartPointer<svtkKdTree>::New();
  kd->BuildLocatorFromPoints(points);

  double distance;
  svtkIdType id = kd->FindClosestPoint(0.5, 0.5, 0.5, distance);
  if (id != 3)
  {
    cerr << "FindClosestPoint failed" << endl;
    retVal++;
  }

  double area[6] = {
    0.2, 0.8, //
    0.2, 0.8, //
    0.2, 0.8  //
  };
  svtkSmartPointer<svtkIdTypeArray> ids = svtkSmartPointer<svtkIdTypeArray>::New();
  kd->FindPointsInArea(area, ids);
  svtkIdType count = ids->GetNumberOfValues();
  if (count != 2)
  {
    cerr << "FindPointsInArea failed" << endl;
    retVal++;
  }

  double center[3] = { 0.0, 0.0, 0.0 };
  svtkSmartPointer<svtkIdList> idList = svtkSmartPointer<svtkIdList>::New();
  kd->FindPointsWithinRadius(10, center, idList);
  svtkIdType n = idList->GetNumberOfIds();
  if (n != 10)
  {
    cerr << "FindPointsWithinRadius failed" << endl;
    retVal++;
  }

  return retVal;
}

int TestKdTreeRepresentation(int argc, char* argv[])
{
  double glyphSize = 0.05;
  const svtkIdType num_points = 10;
  // random points generated on Linux (rand does not work the same on different
  // platforms)
  double p[num_points][3] = {
    { 0.840188, 0.394383, 0.783099 },
    { 0.79844, 0.911647, 0.197551 },
    { 0.335223, 0.76823, 0.277775 },
    { 0.55397, 0.477397, 0.628871 },
    { 0.364784, 0.513401, 0.95223 },
    { 0.916195, 0.635712, 0.717297 },
    { 0.141603, 0.606969, 0.0163006 },
    { 0.242887, 0.137232, 0.804177 },
    { 0.156679, 0.400944, 0.12979 },
    { 0.108809, 0.998925, 0.218257 },
  };

  // generate random points
  SVTK_CREATE(svtkPolyData, pointData);
  SVTK_CREATE(svtkPoints, points);
  points->SetDataTypeToDouble();
  points->SetNumberOfPoints(num_points);
  pointData->AllocateEstimate(num_points, 1);
  for (svtkIdType i = 0; i < num_points; ++i)
  {
    points->SetPoint(i, p[i]);
    pointData->InsertNextCell(SVTK_VERTEX, 1, &i);
  }
  pointData->SetPoints(points);

  // create a kdtree
  SVTK_CREATE(svtkKdTree, kdTree);
  kdTree->SetMinCells(1);
  kdTree->BuildLocatorFromPoints(points);

  // generate a kdtree representation
  SVTK_CREATE(svtkPolyData, kdTreeRepr);
  kdTree->GenerateRepresentation(/*kdTree->GetLevel()*/ 2, kdTreeRepr);
  SVTK_CREATE(svtkPolyDataMapper, kdTreeReprMapper);
  kdTreeReprMapper->SetInputData(kdTreeRepr);

  SVTK_CREATE(svtkActor, kdTreeReprActor);
  kdTreeReprActor->SetMapper(kdTreeReprMapper);
  kdTreeReprActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
  kdTreeReprActor->GetProperty()->SetRepresentationToWireframe();
  kdTreeReprActor->GetProperty()->SetLineWidth(4);
  kdTreeReprActor->GetProperty()->LightingOff();

  //
  // Create vertex glyphs
  //
  SVTK_CREATE(svtkSphereSource, sphere);
  sphere->SetRadius(glyphSize);

  SVTK_CREATE(svtkGlyph3D, glyph);
  glyph->SetInputData(0, pointData);
  glyph->SetInputConnection(1, sphere->GetOutputPort());

  SVTK_CREATE(svtkPolyDataMapper, glyphMapper);
  glyphMapper->SetInputConnection(glyph->GetOutputPort());

  SVTK_CREATE(svtkActor, glyphActor);
  glyphActor->SetMapper(glyphMapper);

  //
  // Set up render window
  //

  SVTK_CREATE(svtkCamera, camera);
  svtkSmartPointer<svtkCamera>::New();
  camera->SetPosition(-10, 10, 20);
  camera->SetFocalPoint(0, 0, 0);

  SVTK_CREATE(svtkRenderer, ren);
  ren->AddActor(glyphActor);
  ren->AddActor(kdTreeReprActor);
  ren->SetActiveCamera(camera);
  ren->ResetCamera();

  SVTK_CREATE(svtkRenderWindow, win);
  win->AddRenderer(ren);

  SVTK_CREATE(svtkRenderWindowInteractor, iren);
  iren->SetRenderWindow(win);
  iren->Initialize();

  SVTK_CREATE(svtkInteractorStyleRubberBandPick, interact);
  iren->SetInteractorStyle(interact);

  int retVal = svtkRegressionTestImage(win);
  if (retVal == svtkRegressionTester::DO_INTERACTOR)
  {
    iren->Start();
    retVal = svtkRegressionTester::PASSED;
  }
  retVal = !retVal;
  retVal += TestKdTreeFunctions();
  return retVal;
}
