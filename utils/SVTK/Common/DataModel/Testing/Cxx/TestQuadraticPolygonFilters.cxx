/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestQuadraticPolygonFilters.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkActor.h"
#include "svtkCamera.h"
#include "svtkCellData.h"
#include "svtkCellPicker.h"
#include "svtkClipDataSet.h"
#include "svtkContourFilter.h"
#include "svtkDataSetMapper.h"
#include "svtkDoubleArray.h"
#include "svtkGeometryFilter.h"
#include "svtkIdTypeArray.h"
#include "svtkMath.h"
#include "svtkNew.h"
#include "svtkOutlineFilter.h"
#include "svtkPlane.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkPolyDataMapper.h"
#include "svtkPolyDataNormals.h"
#include "svtkProperty.h"
#include "svtkRegressionTestImage.h"
#include "svtkRenderWindow.h"
#include "svtkRenderWindowInteractor.h"
#include "svtkRenderer.h"
#include "svtkTransform.h"
#include "svtkUnstructuredGrid.h"

int TestPicker(svtkRenderWindow* renWin, svtkRenderer* renderer);

svtkIdType GetCellIdFromPickerPosition(svtkRenderer* ren, int x, int y);

int TestQuadraticPolygonFilters(int argc, char* argv[])
{
  // create the object
  int npts = 12;

  svtkIdType* connectivityQuadPoly1 = new svtkIdType[npts / 2];
  svtkIdType* connectivityQuadPoly2 = new svtkIdType[npts / 2];
  svtkIdType* connectivityQuads = new svtkIdType[npts];

  svtkNew<svtkPoints> points;
  points->SetNumberOfPoints(npts);

  double ray = 1.0;
  double thetaStep = 4.0 * svtkMath::Pi() / npts;
  double theta;
  for (int i = 0; i < npts / 2; i++)
  {
    if (i < npts / 4)
    {
      theta = thetaStep * i * 2;
    }
    else
    {
      theta = thetaStep * (i - npts / 4) * 2 + thetaStep;
    }

    double x = ray * cos(theta);
    double y = ray * sin(theta);
    points->SetPoint(i, x, y, 0.0);
    points->SetPoint(npts / 2 + i, x, y, 1.0);

    connectivityQuadPoly1[i] = i;
    connectivityQuadPoly2[i] = npts / 2 + i;
    if (i < npts / 4)
    {
      connectivityQuads[4 * i + 0] = i;
      connectivityQuads[4 * i + 1] = (i + 1) % (npts / 4);
      connectivityQuads[4 * i + 2] = ((i + 1) % (npts / 4)) + npts / 2;
      connectivityQuads[4 * i + 3] = i + npts / 2;
    }
  }

  svtkNew<svtkUnstructuredGrid> ugrid;
  ugrid->SetPoints(points);
  ugrid->InsertNextCell(SVTK_QUADRATIC_POLYGON, npts / 2, connectivityQuadPoly1);
  ugrid->InsertNextCell(SVTK_QUADRATIC_POLYGON, npts / 2, connectivityQuadPoly2);
  for (int i = 0; i < npts / 4; i++)
  {
    ugrid->InsertNextCell(SVTK_QUAD, 4, connectivityQuads + i * 4);
  }

  delete[] connectivityQuadPoly1;
  delete[] connectivityQuadPoly2;
  delete[] connectivityQuads;

  // to get the cell id with the picker
  svtkNew<svtkIdTypeArray> id;
  id->SetName("CellID");
  id->SetNumberOfComponents(1);
  id->SetNumberOfTuples(ugrid->GetNumberOfCells());
  for (int i = 0; i < ugrid->GetNumberOfCells(); i++)
  {
    id->SetValue(i, i);
  }
  ugrid->GetCellData()->AddArray(id);

  // Setup the scalars
  svtkNew<svtkDoubleArray> scalars;
  scalars->SetNumberOfComponents(1);
  scalars->SetNumberOfTuples(ugrid->GetNumberOfPoints());
  scalars->SetName("Scalars");
  scalars->SetValue(0, 1);
  scalars->SetValue(1, 2);
  scalars->SetValue(2, 2);
  scalars->SetValue(3, 1);
  scalars->SetValue(4, 2);
  scalars->SetValue(5, 1);
  scalars->SetValue(6, 1);
  scalars->SetValue(7, 2);
  scalars->SetValue(8, 2);
  scalars->SetValue(9, 1);
  scalars->SetValue(10, 2);
  scalars->SetValue(11, 1);
  ugrid->GetPointData()->SetScalars(scalars);

  // clip filter
  // svtkNew<svtkPlane> plane;
  // plane->SetOrigin(0, 0, 0);
  // plane->SetNormal(1, 0, 0);
  svtkNew<svtkClipDataSet> clip;
  // clip->SetClipFunction(plane);
  // clip->GenerateClipScalarsOn();
  clip->SetValue(1.5);
  clip->SetInputData(ugrid);
  clip->Update();
  svtkNew<svtkDataSetMapper> clipMapper;
  clipMapper->SetInputConnection(clip->GetOutputPort());
  clipMapper->SetScalarRange(1.0, 2.0);
  clipMapper->InterpolateScalarsBeforeMappingOn();
  svtkNew<svtkActor> clipActor;
  clipActor->SetPosition(0.0, 2.0, 0.0);
  clipActor->SetMapper(clipMapper);

  // contour filter
  svtkNew<svtkContourFilter> contourFilter;
  contourFilter->SetInputData(ugrid);
  contourFilter->SetValue(0, 1.5);
  contourFilter->Update();
  svtkNew<svtkPolyDataNormals> contourNormals;
  contourNormals->SetInputConnection(contourFilter->GetOutputPort());
  svtkNew<svtkPolyDataMapper> contourMapper;
  contourMapper->SetInputConnection(contourNormals->GetOutputPort());
  contourMapper->ScalarVisibilityOff();
  svtkNew<svtkActor> contourActor;
  contourActor->SetMapper(contourMapper);
  contourActor->GetProperty()->SetColor(0, 0, 0);
  contourActor->SetPosition(0.0, 0.01, 0.01);

  // outline filter
  svtkNew<svtkOutlineFilter> outlineFilter;
  outlineFilter->SetInputData(ugrid);
  svtkNew<svtkPolyDataMapper> outlineMapper;
  outlineMapper->SetInputConnection(outlineFilter->GetOutputPort());
  svtkNew<svtkActor> outlineActor;
  outlineActor->SetMapper(outlineMapper);
  outlineActor->GetProperty()->SetColor(0, 0, 0);
  outlineActor->SetPosition(0.0, 0.01, 0.01);

  // geometry filter
  svtkNew<svtkGeometryFilter> geometryFilter;
  geometryFilter->SetInputData(ugrid);
  geometryFilter->Update();
  svtkNew<svtkPolyDataMapper> geometryMapper;
  geometryMapper->SetInputConnection(geometryFilter->GetOutputPort());
  geometryMapper->SetScalarRange(1.0, 2.0);
  geometryMapper->InterpolateScalarsBeforeMappingOn();
  svtkNew<svtkActor> geometryActor;
  geometryActor->SetMapper(geometryMapper);

  // drawing
  svtkNew<svtkRenderer> ren;
  ren->SetBackground(1, 1, 1);
  ren->AddActor(geometryActor);
  ren->AddActor(outlineActor);
  ren->AddActor(clipActor);
  ren->AddActor(contourActor);
  svtkNew<svtkRenderWindow> renWin;
  renWin->AddRenderer(ren);
  renWin->SetSize(600, 600);
  renWin->SetMultiSamples(0);
  svtkNew<svtkRenderWindowInteractor> iren;
  iren->SetRenderWindow(renWin);
  renWin->Render();

  // tests
  if (TestPicker(renWin, ren) == EXIT_FAILURE)
  {
    return EXIT_FAILURE;
  }

  int retVal = svtkRegressionTestImage(renWin);
  if (retVal == svtkRegressionTester::DO_INTERACTOR)
  {
    iren->Start();
    retVal = svtkRegressionTester::PASSED;
  }

  return (retVal == svtkRegressionTester::PASSED) ? EXIT_SUCCESS : EXIT_FAILURE;
}

int TestPicker(svtkRenderWindow* renWin, svtkRenderer* renderer)
{
  // Sets the camera
  double cPos[3] = { 5.65647, 0.857996, 6.71491 };
  double cUp[3] = { 0.0212226, 0.999769, 0.00352794 };
  svtkCamera* camera = renderer->GetActiveCamera();
  camera->SetPosition(cPos);
  camera->SetViewUp(cUp);
  renderer->ResetCameraClippingRange();
  renWin->Render();
  renWin->Render();
  renWin->Render();

  // Sets the reference values
  int nbTests = 17;
  int values[] = {
    218, 244, 1, //
    290, 244, 1, //
    201, 168, 1, //
    319, 166, 1, //
    223, 63, 1,  //
    303, 46, 1,  //
    330, 238, 2, //
    420, 173, 2, //
    376, 165, 2, //
    372, 128, 4, //
    411, 149, 4, //
    348, 266, 0, //
    416, 203, 0, //
    391, 269, 0, //
    412, 119, 0, //
    391, 61, 0,  //
    340, 72, 0   //
  };

  for (int i = 0; i < nbTests * 3; i += 3)
  {
    if (GetCellIdFromPickerPosition(renderer, values[i], values[i + 1]) != values[i + 2])
    {
      cerr << "ERROR:  selected cell type is "
           << GetCellIdFromPickerPosition(renderer, values[i], values[i + 1]) << ", should be "
           << values[i + 2] << endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

svtkIdType GetCellIdFromPickerPosition(svtkRenderer* ren, int x, int y)
{
  svtkNew<svtkCellPicker> picker;
  picker->SetTolerance(0.0005);

  // Pick from this location.
  picker->Pick(x, y, 0, ren);

  svtkIdType cellId = -1;
  if (picker->GetDataSet())
  {
    svtkIdTypeArray* ids =
      svtkArrayDownCast<svtkIdTypeArray>(picker->GetDataSet()->GetCellData()->GetArray("CellID"));
    cellId = ids->GetValue(picker->GetCellId());
  }

  return cellId;
}
