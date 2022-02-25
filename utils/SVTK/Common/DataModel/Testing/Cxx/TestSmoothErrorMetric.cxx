/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestSmoothErrorMetric.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// This example demonstrates how to implement a svtkGenericDataSet
// (here svtkBridgeDataSet) and to use svtkGenericDataSetTessellator filter on
// it.
//
// The command line arguments are:
// -I        => run in interactive mode; unless this is used, the program will
//              not allow interaction and exit
// -D <path> => path to the data; the data should be in <path>/Data/

//#define WRITE_GENERIC_RESULT

#include "svtkActor.h"
#include "svtkBridgeDataSet.h"
#include "svtkDebugLeaks.h"
#include "svtkGenericCellTessellator.h"
#include "svtkGenericGeometryFilter.h"
#include "svtkGenericSubdivisionErrorMetric.h"
#include "svtkLookupTable.h"
#include "svtkPointData.h"
#include "svtkPolyData.h"
#include "svtkPolyDataMapper.h"
#include "svtkPolyDataNormals.h"
#include "svtkProperty.h"
#include "svtkRegressionTestImage.h"
#include "svtkRenderWindow.h"
#include "svtkRenderWindowInteractor.h"
#include "svtkRenderer.h"
#include "svtkSimpleCellTessellator.h"
#include "svtkSmartPointer.h"
#include "svtkSmoothErrorMetric.h"
#include "svtkTestUtilities.h"
#include "svtkUnstructuredGrid.h"
#include "svtkXMLUnstructuredGridReader.h"
#include <cassert>

#ifdef WRITE_GENERIC_RESULT
#include "svtkXMLPolyDataWriter.h"
#endif // #ifdef WRITE_GENERIC_RESULT

// Remark about the lookup tables that seem different between the
// GenericGeometryFilter and GenericDataSetTessellator:
// the lookup table is set for the whole unstructured grid, the tetra plus
// the triangle. The lookup table changed because of the tetra: the
// GenericDataSetTessellator need to create inside sub-tetra that have
// minimal attributes, the GenericGeometryFilter just need to tessellate the
// face of the tetra, for which the values at points are not minimal.

int TestSmoothErrorMetric(int argc, char* argv[])
{
  // Standard rendering classes
  svtkSmartPointer<svtkRenderer> renderer = svtkSmartPointer<svtkRenderer>::New();
  svtkSmartPointer<svtkRenderWindow> renWin = svtkSmartPointer<svtkRenderWindow>::New();
  renWin->AddRenderer(renderer);
  svtkSmartPointer<svtkRenderWindowInteractor> iren =
    svtkSmartPointer<svtkRenderWindowInteractor>::New();
  iren->SetRenderWindow(renWin);

  // Load the mesh geometry and data from a file
  svtkSmartPointer<svtkXMLUnstructuredGridReader> reader =
    svtkSmartPointer<svtkXMLUnstructuredGridReader>::New();
  char* cfname = svtkTestUtilities::ExpandDataFileName(argc, argv, "Data/quadraticTetra01.vtu");

  reader->SetFileName(cfname);
  delete[] cfname;

  // Force reading
  reader->Update();

  // Initialize the bridge
  svtkSmartPointer<svtkBridgeDataSet> ds = svtkSmartPointer<svtkBridgeDataSet>::New();
  ds->SetDataSet(reader->GetOutput());

  // Set the smooth error metric thresholds:
  // 1. for the geometric error metric
  svtkSmartPointer<svtkSmoothErrorMetric> smoothError = svtkSmartPointer<svtkSmoothErrorMetric>::New();
  smoothError->SetAngleTolerance(179);
  ds->GetTessellator()->GetErrorMetrics()->AddItem(smoothError);

  // 2. for the attribute error metric
  //  svtkAttributesErrorMetric *attributesError=svtkAttributesErrorMetric::New();
  //  attributesError->SetAttributeTolerance(0.01);

  //  ds->GetTessellator()->GetErrorMetrics()->AddItem(attributesError);
  //  attributesError->Delete();

  cout << "input unstructured grid: " << ds << endl;

  static_cast<svtkSimpleCellTessellator*>(ds->GetTessellator())->SetMaxSubdivisionLevel(100);

  svtkIndent indent;
  ds->PrintSelf(cout, indent);

  // Create the filter
  svtkSmartPointer<svtkGenericGeometryFilter> geom = svtkSmartPointer<svtkGenericGeometryFilter>::New();
  geom->SetInputData(ds);

  geom->Update(); // So that we can call GetRange() on the scalars

  assert(geom->GetOutput() != nullptr);

  // This creates a blue to red lut.
  svtkSmartPointer<svtkLookupTable> lut = svtkSmartPointer<svtkLookupTable>::New();
  lut->SetHueRange(0.667, 0.0);

  svtkSmartPointer<svtkPolyDataMapper> mapper = svtkSmartPointer<svtkPolyDataMapper>::New();

  mapper->ScalarVisibilityOff();

#if 0
  svtkSmartPointer<svtkPolyDataNormals> normalGenerator=
    svtkSmartPointer<svtkPolyDataNormals>::New();
  normalGenerator->SetFeatureAngle(0.1);
  normalGenerator->SetSplitting(1);
  normalGenerator->SetConsistency(0);
  normalGenerator->SetAutoOrientNormals(0);
  normalGenerator->SetComputePointNormals(1);
  normalGenerator->SetComputeCellNormals(0);
  normalGenerator->SetFlipNormals(0);
  normalGenerator->SetNonManifoldTraversal(1);
  normalGenerator->SetInputConnection( geom->GetOutputPort() );
  mapper->SetInputConnection(normalGenerator->GetOutputPort() );
  normalGenerator->Delete( );
#else
  mapper->SetInputConnection(geom->GetOutputPort());
#endif

  if (geom->GetOutput()->GetPointData() != nullptr)
  {
    if (geom->GetOutput()->GetPointData()->GetScalars() != nullptr)
    {
      mapper->SetScalarRange(geom->GetOutput()->GetPointData()->GetScalars()->GetRange());
    }
  }

  svtkSmartPointer<svtkActor> actor = svtkSmartPointer<svtkActor>::New();
  actor->SetMapper(mapper);
  renderer->AddActor(actor);

#ifdef WRITE_GENERIC_RESULT
  // Save the result of the filter in a file
  svtkSmartPointer<svtkXMLPolyDataWriter> writer = svtkSmartPointer<svtkXMLPolyDataWriter>::New();
  writer->SetInputConnection(geom->GetOutputPort());
  writer->SetFileName("geometry.vtp");
  writer->SetDataModeToAscii();
  writer->Write();
  writer->Delete();
#endif // #ifdef WRITE_GENERIC_RESULT

  // Standard testing code.
  renderer->SetBackground(0.5, 0.5, 0.5);
  renWin->SetSize(300, 300);
  renWin->Render();
  int retVal = svtkRegressionTestImage(renWin);
  if (retVal == svtkRegressionTester::DO_INTERACTOR)
  {
    iren->Start();
  }

  return !retVal;
}
