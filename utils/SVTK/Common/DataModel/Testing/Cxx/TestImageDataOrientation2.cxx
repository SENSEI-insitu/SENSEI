/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestImageDataOrientation2.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// .NAME Test orientation for image data
// .SECTION Description
// This program tests the location of an oriented Image Data by using a
// non-identity direction matrix and extracting points of the image data
// that fall within a sphere.

#include "svtkDebugLeaks.h"
#include "svtkGlyph3D.h"
#include "svtkImageData.h"
#include "svtkMath.h"
#include "svtkMatrix4x4.h"
#include "svtkNew.h"
#include "svtkPolyDataMapper.h"
#include "svtkProperty.h"
#include "svtkRegressionTestImage.h"
#include "svtkRenderWindow.h"
#include "svtkRenderWindowInteractor.h"
#include "svtkRenderer.h"
#include "svtkSelectEnclosedPoints.h"
#include "svtkSphereSource.h"
#include "svtkThresholdPoints.h"

int TestImageDataOrientation2(int argc, char* argv[])
{
  // Standard rendering classes
  svtkNew<svtkRenderer> renderer;
  svtkNew<svtkRenderWindow> renWin;
  renWin->AddRenderer(renderer);
  svtkNew<svtkRenderWindowInteractor> iren;
  iren->SetRenderWindow(renWin);

  // Create an oriented image data
  double angle = -svtkMath::Pi() / 4;
  double direction[9] = { cos(angle), sin(angle), 0, -sin(angle), cos(angle), 0, 0, 0, 1 };
  svtkNew<svtkImageData> image;
  image->SetExtent(0, 6, 0, 10, 0, 10);
  image->SetOrigin(-0.4, 0.2, -0.6);
  image->SetSpacing(0.4, -0.25, 0.25);
  image->SetDirectionMatrix(direction);
  image->AllocateScalars(SVTK_DOUBLE, 0);

  // Create a containing surface
  svtkNew<svtkSphereSource> ss;
  ss->SetPhiResolution(25);
  ss->SetThetaResolution(38);
  ss->SetCenter(0, 0, 0);
  ss->SetRadius(2.5);
  svtkNew<svtkPolyDataMapper> sphereMapper;
  sphereMapper->SetInputConnection(ss->GetOutputPort());
  svtkNew<svtkActor> sphereActor;
  sphereActor->SetMapper(sphereMapper);
  sphereActor->GetProperty()->SetRepresentationToWireframe();

  svtkNew<svtkSelectEnclosedPoints> select;
  select->SetInputData(image);
  select->SetSurfaceConnection(ss->GetOutputPort());

  // Now extract points
  svtkNew<svtkThresholdPoints> thresh;
  thresh->SetInputConnection(select->GetOutputPort());
  thresh->SetInputArrayToProcess(
    0, 0, 0, svtkDataObject::FIELD_ASSOCIATION_POINTS, "SelectedPoints");
  thresh->ThresholdByUpper(0.5);

  // Show points as glyphs
  svtkNew<svtkSphereSource> glyph;
  svtkNew<svtkGlyph3D> glypher;
  glypher->SetInputConnection(thresh->GetOutputPort());
  glypher->SetSourceConnection(glyph->GetOutputPort());
  glypher->SetScaleModeToDataScalingOff();
  glypher->SetScaleFactor(0.15);

  svtkNew<svtkPolyDataMapper> pointsMapper;
  pointsMapper->SetInputConnection(glypher->GetOutputPort());
  pointsMapper->ScalarVisibilityOff();

  svtkNew<svtkActor> pointsActor;
  pointsActor->SetMapper(pointsMapper);
  pointsActor->GetProperty()->SetColor(0, 0, 1);

  // Add actors
  //  renderer->AddActor(sphereActor);
  renderer->AddActor(pointsActor);

  // Standard testing code.
  renWin->SetSize(400, 400);
  renWin->Render();
  int retVal = svtkRegressionTestImage(renWin);
  if (retVal == svtkRegressionTester::DO_INTERACTOR)
  {
    iren->Start();
  }

  return !retVal;
}
