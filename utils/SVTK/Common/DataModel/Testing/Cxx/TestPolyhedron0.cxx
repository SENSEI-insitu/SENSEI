/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestPolyhedron.cxx

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
#include "svtkCubeSource.h"
#include "svtkDataArray.h"
#include "svtkDataSetMapper.h"
#include "svtkElevationFilter.h"
#include "svtkExtractEdges.h"
#include "svtkGenericCell.h"
#include "svtkIdList.h"
#include "svtkPointData.h"
#include "svtkPointLocator.h"
#include "svtkPoints.h"
#include "svtkPolyhedron.h"
#include "svtkProperty.h"
#include "svtkRenderWindow.h"
#include "svtkRenderWindowInteractor.h"
#include "svtkRenderer.h"
#include "svtkShrinkFilter.h"
#include "svtkSmartPointer.h"
#include "svtkStructuredGridReader.h"
#include "svtkUnstructuredGrid.h"
#include "svtkXMLUnstructuredGridReader.h"
#include "svtkXMLUnstructuredGridWriter.h"

#include "svtkRegressionTestImage.h"
#include "svtkTestUtilities.h"

#define compare_doublevec(x, y, e)                                                                 \
  (((x[0] - y[0]) < (e)) && ((x[0] - y[0]) > -(e)) && ((x[1] - y[1]) < (e)) &&                     \
    ((x[1] - y[1]) > -(e)) && ((x[2] - y[2]) < (e)) && ((x[2] - y[2]) > -(e)))

#define compare_double(x, y, e) ((x) - (y) < (e) && (x) - (y) > -(e))

// Test of svtkPolyhedron. A structured grid is converted to a polyhedral
// mesh.
int TestPolyhedron0(int argc, char* argv[])
{
  // create the a cube
  svtkSmartPointer<svtkCubeSource> cube = svtkSmartPointer<svtkCubeSource>::New();
  cube->SetXLength(10);
  cube->SetYLength(10);
  cube->SetZLength(20);
  cube->SetCenter(0, 0, 0);
  cube->Update();

  // add scaler
  svtkSmartPointer<svtkElevationFilter> ele = svtkSmartPointer<svtkElevationFilter>::New();
  ele->SetInputConnection(cube->GetOutputPort());
  ele->SetLowPoint(0, 0, -10);
  ele->SetHighPoint(0, 0, 10);
  ele->Update();
  svtkPolyData* poly = svtkPolyData::SafeDownCast(ele->GetOutput());

  // create a test polyhedron
  svtkIdType pointIds[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

  svtkSmartPointer<svtkCellArray> faces = svtkSmartPointer<svtkCellArray>::New();
  svtkIdType face0[4] = { 0, 2, 6, 4 };
  svtkIdType face1[4] = { 1, 3, 7, 5 };
  svtkIdType face2[4] = { 0, 1, 3, 2 };
  svtkIdType face3[4] = { 4, 5, 7, 6 };
  svtkIdType face4[4] = { 0, 1, 5, 4 };
  svtkIdType face5[4] = { 2, 3, 7, 6 };
  faces->InsertNextCell(4, face0);
  faces->InsertNextCell(4, face1);
  faces->InsertNextCell(4, face2);
  faces->InsertNextCell(4, face3);
  faces->InsertNextCell(4, face4);
  faces->InsertNextCell(4, face5);

  svtkSmartPointer<svtkUnstructuredGrid> ugrid0 = svtkSmartPointer<svtkUnstructuredGrid>::New();
  ugrid0->SetPoints(poly->GetPoints());
  ugrid0->GetPointData()->DeepCopy(poly->GetPointData());

  svtkNew<svtkIdTypeArray> legacyFaces;
  faces->ExportLegacyFormat(legacyFaces);

  ugrid0->InsertNextCell(SVTK_POLYHEDRON, 8, pointIds, 6, legacyFaces->GetPointer(0));

  svtkPolyhedron* polyhedron = static_cast<svtkPolyhedron*>(ugrid0->GetCell(0));

  svtkCellArray* cell = ugrid0->GetCells();
  svtkNew<svtkIdTypeArray> pids;
  cell->ExportLegacyFormat(pids);
  std::cout << "num of cells: " << cell->GetNumberOfCells() << std::endl;
  std::cout << "num of tuples: " << pids->GetNumberOfTuples() << std::endl;
  for (int i = 0; i < pids->GetNumberOfTuples(); i++)
  {
    std::cout << pids->GetValue(i) << " ";
  }
  std::cout << std::endl;
  cell->Print(std::cout);

  // Print out basic information
  std::cout << "Testing polyhedron is a cube of with bounds "
            << "[-5, 5, -5, 5, -10, 10]. It has " << polyhedron->GetNumberOfEdges() << " edges and "
            << polyhedron->GetNumberOfFaces() << " faces." << std::endl;

  double p1[3] = { -100, 0, 0 };
  double p2[3] = { 100, 0, 0 };
  double tol = 0.001;
  double t, x[3], pc[3];
  int subId = 0;

  //
  // test writer
  svtkSmartPointer<svtkXMLUnstructuredGridWriter> writer =
    svtkSmartPointer<svtkXMLUnstructuredGridWriter>::New();
  writer->SetInputData(ugrid0);
  writer->SetFileName("test.vtu");
  writer->SetDataModeToAscii();
  writer->Update();
  std::cout << "finished writing the polyhedron mesh to test.vth " << std::endl;

  //
  // test reader
  svtkSmartPointer<svtkXMLUnstructuredGridReader> reader =
    svtkSmartPointer<svtkXMLUnstructuredGridReader>::New();
  reader->SetFileName("test.vtu");
  reader->Update();
  std::cout << "finished reading the polyhedron mesh from test.vth " << std::endl;

  svtkUnstructuredGrid* ugrid = reader->GetOutput();
  polyhedron = svtkPolyhedron::SafeDownCast(ugrid->GetCell(0));

  // write again to help compare
  writer->SetInputData(ugrid);
  writer->SetFileName("test1.vtu");
  writer->SetDataModeToAscii();
  writer->Update();

  // test the polyhedron functions
  // test intersection
  int hit = polyhedron->IntersectWithLine(p1, p2, tol, t, x, pc, subId); // should hit
  if (!hit)
  {
    cerr << "Expected  intersection, but missed." << std::endl;
    return EXIT_FAILURE;
  }

  // test inside
  int inside = polyhedron->IsInside(p1, tol); // should be out
  if (inside)
  {
    cerr << "Expect point [" << p1[0] << ", " << p1[1] << ", " << p1[2]
         << "] to be outside the polyhedral, but it's inside." << std::endl;
    return EXIT_FAILURE;
  }

  p2[0] = 0.0;
  p2[1] = 0.0;
  p2[2] = 0.0;
  inside = polyhedron->IsInside(p2, tol); // should be in
  if (!inside)
  {
    cerr << "Expect point [" << p2[0] << ", " << p2[1] << ", " << p2[2]
         << "] to be inside the polyhedral, but it's outside." << std::endl;
    return EXIT_FAILURE;
  }

  // test EvaluatePosition and interpolation function
  double weights[8], closestPoint[3], dist2;

  for (int i = 0; i < 8; i++)
  {
    double v;
    poly->GetPointData()->GetScalars()->GetTuple(i, &v);
    std::cout << v << " ";
  }
  std::cout << std::endl;

  // case 0: point on the polyhedron
  x[0] = 5.0;
  x[1] = 0.0;
  x[2] = 0.0;
  polyhedron->EvaluatePosition(x, closestPoint, subId, pc, dist2, weights);

  std::cout << "weights for point [" << x[0] << ", " << x[1] << ", " << x[2] << "]:" << std::endl;
  for (int i = 0; i < 8; i++)
  {
    std::cout << weights[i] << " ";
  }
  std::cout << std::endl;

  double refWeights[8] = { 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25 };
  for (int i = 0; i < 8; i++)
  {
    if (!compare_double(refWeights[i], weights[i], 0.00001))
    {
      std::cout << "Error computing the weights for a point on the polyhedron." << std::endl;
      return EXIT_FAILURE;
    }
  }

  double refClosestPoint[3] = { 5.0, 0.0, 0.0 };
  if (!compare_doublevec(closestPoint, refClosestPoint, 0.00001))
  {
    std::cout << "Error finding the closet point of a point on the polyhedron." << std::endl;
    return EXIT_FAILURE;
  }

  double refDist2 = 0.0;
  if (!compare_double(dist2, refDist2, 0.000001))
  {
    std::cout << "Error computing the distance for a point on the polyhedron." << std::endl;
    return EXIT_FAILURE;
  }

  // case 1: point inside the polyhedron
  x[0] = 0.0;
  x[1] = 0.0;
  x[2] = 0.0;
  polyhedron->EvaluatePosition(x, closestPoint, subId, pc, dist2, weights);

  std::cout << "weights for point [" << x[0] << ", " << x[1] << ", " << x[2] << "]:" << std::endl;
  for (int i = 0; i < 8; i++)
  {
    std::cout << weights[i] << " ";
  }
  std::cout << std::endl;

  double refWeights1[8] = { 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125 };
  for (int i = 0; i < 8; i++)
  {
    if (!compare_double(refWeights1[i], weights[i], 0.00001))
    {
      std::cout << "Error computing the weights for a point inside the polyhedron." << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (!compare_double(dist2, refDist2, 0.000001))
  {
    std::cout << "Error computing the distance for a point inside the polyhedron." << std::endl;
    return EXIT_FAILURE;
  }

  // case 2: point outside the polyhedron
  x[0] = 8.0;
  x[1] = 0.0;
  x[2] = 0.0;
  polyhedron->EvaluatePosition(x, closestPoint, subId, pc, dist2, weights);

  std::cout << "weights for point [" << x[0] << ", " << x[1] << ", " << x[2] << "]:" << std::endl;
  for (int i = 0; i < 8; i++)
  {
    std::cout << weights[i] << " ";
  }
  std::cout << std::endl;

  double refWeights2[8] = { 0.0307, 0.0307, 0.0307, 0.0307, 0.2193, 0.2193, 0.2193, 0.2193 };
  for (int i = 0; i < 8; i++)
  {
    if (!compare_double(refWeights2[i], weights[i], 0.0001))
    {
      std::cout << "Error computing the weights for a point outside the polyhedron." << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (!compare_doublevec(closestPoint, refClosestPoint, 0.00001))
  {
    std::cout << "Error finding the closet point of a point outside the polyhedron." << std::endl;
    return EXIT_FAILURE;
  }

  refDist2 = 9.0;
  if (!compare_double(dist2, refDist2, 0.000001))
  {
    std::cout << "Error computing the distance for a point outside the polyhedron." << std::endl;
    return EXIT_FAILURE;
  }

  // test evaluation location
  double weights1[8];
  polyhedron->EvaluateLocation(subId, pc, x, weights1);

  double refPoint[3] = { 8.0, 0.0, 0.0 };
  if (!compare_doublevec(refPoint, x, 0.00001))
  {
    std::cout << "Error evaluate the point location for its parameter coordinate." << std::endl;
    return EXIT_FAILURE;
  }

  for (int i = 0; i < 8; i++)
  {
    if (!compare_double(refWeights2[i], weights1[i], 0.0001))
    {
      std::cout << "Error computing the weights based on parameter coordinates." << std::endl;
      return EXIT_FAILURE;
    }
  }

  // test derivative
  pc[0] = 0;
  pc[1] = 0.5;
  pc[2] = 0.5;
  polyhedron->EvaluateLocation(subId, pc, x, weights1);

  double deriv[3], values[8];
  svtkDataArray* dataArray = poly->GetPointData()->GetScalars();
  for (int i = 0; i < 8; i++)
  {
    dataArray->GetTuple(i, values + i);
  }
  polyhedron->Derivatives(subId, pc, values, 1, deriv);

  std::cout << "derivative for point [" << x[0] << ", " << x[1] << ", " << x[2]
            << "]:" << std::endl;
  for (int i = 0; i < 3; i++)
  {
    std::cout << deriv[i] << " ";
  }
  std::cout << std::endl;

  double refDeriv[3] = { 0.0, 0.0, 0.05 };
  if (!compare_doublevec(refDeriv, deriv, 0.00001))
  {
    std::cout << "Error computing derivative for a point inside the polyhedron." << std::endl;
    return EXIT_FAILURE;
  }

  // test triangulation
  svtkSmartPointer<svtkPoints> tetraPoints = svtkSmartPointer<svtkPoints>::New();
  svtkSmartPointer<svtkIdList> tetraIdList = svtkSmartPointer<svtkIdList>::New();
  polyhedron->Triangulate(0, tetraIdList, tetraPoints);

  std::cout << std::endl << "Triangulation result:" << std::endl;

  for (int i = 0; i < tetraPoints->GetNumberOfPoints(); i++)
  {
    double* pt = tetraPoints->GetPoint(i);
    std::cout << "point #" << i << ": [" << pt[0] << ", " << pt[1] << ", " << pt[2] << "]"
              << std::endl;
  }

  svtkIdType* ids = tetraIdList->GetPointer(0);
  for (int i = 0; i < tetraIdList->GetNumberOfIds(); i += 4)
  {
    std::cout << "tetra #" << i / 4 << ":" << ids[i] << " " << ids[i + 1] << " " << ids[i + 2]
              << " " << ids[i + 3] << std::endl;
  }

  svtkSmartPointer<svtkUnstructuredGrid> tetraGrid = svtkSmartPointer<svtkUnstructuredGrid>::New();
  for (int i = 0; i < tetraIdList->GetNumberOfIds(); i += 4)
  {
    tetraGrid->InsertNextCell(SVTK_TETRA, 4, ids + i);
  }
  tetraGrid->SetPoints(poly->GetPoints());
  tetraGrid->GetPointData()->DeepCopy(poly->GetPointData());

  // test contour
  svtkSmartPointer<svtkPointLocator> locator = svtkSmartPointer<svtkPointLocator>::New();
  svtkSmartPointer<svtkCellArray> resultPolys = svtkSmartPointer<svtkCellArray>::New();
  svtkSmartPointer<svtkPointData> resultPd = svtkSmartPointer<svtkPointData>::New();
  svtkSmartPointer<svtkCellData> resultCd = svtkSmartPointer<svtkCellData>::New();
  svtkSmartPointer<svtkPoints> resultPoints = svtkSmartPointer<svtkPoints>::New();
  resultPoints->DeepCopy(ugrid0->GetPoints());
  locator->InitPointInsertion(resultPoints, ugrid0->GetBounds());

  polyhedron->Contour(0.5, tetraGrid->GetPointData()->GetScalars(), locator, nullptr, nullptr,
    resultPolys, tetraGrid->GetPointData(), resultPd, tetraGrid->GetCellData(), 0, resultCd);

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
  resultPoints1->DeepCopy(ugrid0->GetPoints());
  locator1->InitPointInsertion(resultPoints1, ugrid0->GetBounds());

  polyhedron->Clip(0.5, tetraGrid->GetPointData()->GetScalars(), locator1, resultPolys1,
    tetraGrid->GetPointData(), resultPd1, tetraGrid->GetCellData(), 0, resultCd1, 0);

  // output the clipped polyhedron
  svtkSmartPointer<svtkUnstructuredGrid> clipResult = svtkSmartPointer<svtkUnstructuredGrid>::New();
  clipResult->SetPoints(locator1->GetPoints());
  clipResult->SetCells(SVTK_POLYHEDRON, resultPolys1);
  clipResult->GetPointData()->DeepCopy(resultPd1);

  // shrink to show the gaps between tetrahedrons.
  svtkSmartPointer<svtkShrinkFilter> shrink = svtkSmartPointer<svtkShrinkFilter>::New();
  shrink->SetInputData(tetraGrid);
  shrink->SetShrinkFactor(0.7);

  // create actors
  svtkSmartPointer<svtkDataSetMapper> mapper = svtkSmartPointer<svtkDataSetMapper>::New();
  mapper->SetInputData(poly);

  svtkSmartPointer<svtkActor> actor = svtkSmartPointer<svtkActor>::New();
  actor->SetMapper(mapper);

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
  prop->SetOpacity(0.8);

  // set property
  actor->SetProperty(prop);
  contourActor->SetProperty(prop);
  clipPolyhedronActor->SetProperty(prop);

  svtkSmartPointer<svtkRenderer> ren = svtkSmartPointer<svtkRenderer>::New();
  ren->AddActor(actor);
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
