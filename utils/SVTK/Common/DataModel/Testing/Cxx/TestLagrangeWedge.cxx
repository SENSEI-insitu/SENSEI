/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestInterpolationDerivs.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#define SVTK_EPSILON 1e-10

#include "svtkLagrangeWedge.h"
#include "svtkPoints.h"
#include "svtkSmartPointer.h"
#include <svtkCellArray.h>
#include <svtkMinimalStandardRandomSequence.h>

#include "svtkClipDataSet.h"
#include "svtkDataSetSurfaceFilter.h"
#include "svtkDoubleArray.h"
#include "svtkPointData.h"
#include "svtkPolyData.h"
#include "svtkUnstructuredGrid.h"

#define VISUAL_DEBUG 1

#ifdef VISUAL_DEBUG
#include <svtkActor.h>
#include <svtkCamera.h>
#include <svtkPolyData.h>
#include <svtkPolyDataMapper.h>
#include <svtkProperty.h>
#include <svtkRegressionTestImage.h>
#include <svtkRenderWindow.h>
#include <svtkRenderWindowInteractor.h>
#include <svtkRenderer.h>
#include <svtkVersion.h>
#endif

// #define DEBUG_POINTS

#ifdef DEBUG_POINTS
#include <svtkNew.h>
#include <svtkPolyData.h>
#include <svtkXMLPolyDataWriter.h>
#endif

#include <cassert>

#include <cmath>

namespace
{
svtkSmartPointer<svtkLagrangeWedge> CreateWedge(int nPoints)
{
  svtkSmartPointer<svtkLagrangeWedge> w = svtkSmartPointer<svtkLagrangeWedge>::New();

  w->GetPointIds()->SetNumberOfIds(nPoints);
  w->GetPoints()->SetNumberOfPoints(nPoints);
  w->Initialize();
  w->SetUniformOrderFromNumPoints(nPoints);
  double* points = w->GetParametricCoords();
  for (svtkIdType i = 0; i < nPoints; i++)
  {
    w->GetPointIds()->SetId(i, i);
    w->GetPoints()->SetPoint(i, &points[3 * i]);
  }

  return w;
}

int TestInterpolationFunction(svtkSmartPointer<svtkLagrangeWedge> cell, double eps = SVTK_EPSILON)
{
  int numPts = cell->GetNumberOfPoints();
  double* sf = new double[numPts];
  double* coords = cell->GetParametricCoords();
  int r = 0;
  for (int i = 0; i < numPts; ++i)
  {
    double* point = coords + 3 * i;
    double sum = 0.;
    cell->InterpolateFunctions(point, sf); // virtual function
    for (int j = 0; j < numPts; j++)
    {
      sum += sf[j];
      if (j == i)
      {
        if (fabs(sf[j] - 1) > eps)
        {
          std::cout << "fabs(sf[" << j << "] - 1): " << fabs(sf[j] - 1) << std::endl;
          ++r;
        }
      }
      else
      {
        if (fabs(sf[j] - 0) > eps)
        {
          std::cout << "fabs(sf[" << j << "] - 0): " << fabs(sf[j] - 0) << std::endl;
          ++r;
        }
      }
    }
    if (fabs(sum - 1) > eps)
    {
      std::cout << "fabs(" << sum << " - 1): " << fabs(sum - 1) << std::endl;
      ++r;
    }
  }

  // Let's test unity condition on the center point:
  double center[3];
  cell->GetParametricCenter(center);
  cell->InterpolateFunctions(center, sf); // virtual function
  double sum = 0.;
  for (int j = 0; j < numPts; j++)
  {
    sum += sf[j];
  }
  if (fabs(sum - 1) > eps)
  {
    std::cout << "center: fabs(" << sum << " - 1): " << fabs(sum - 1) << std::endl;
    ++r;
  }

  delete[] sf;
  return r;
}

void InterpolateDerivsNumeric(svtkSmartPointer<svtkLagrangeWedge> cell, double pcoords[3],
  double* derivs, double eps = SVTK_EPSILON)
{
  svtkIdType nPoints = cell->GetPoints()->GetNumberOfPoints();
  double* valp = new double[nPoints];
  double* valm = new double[nPoints];

  double pcoordsp[3], pcoordsm[3];

  for (svtkIdType i = 0; i < 3; i++)
  {
    pcoordsp[i] = pcoordsm[i] = pcoords[i];
  }
  pcoordsp[0] += eps;
  cell->InterpolateFunctions(pcoordsp, valp);

  pcoordsm[0] -= eps;
  cell->InterpolateFunctions(pcoordsm, valm);

  for (svtkIdType idx = 0; idx < nPoints; idx++)
  {
    derivs[idx] = (valp[idx] - valm[idx]) / (2. * eps);
  }

  for (svtkIdType i = 0; i < 3; i++)
  {
    pcoordsp[i] = pcoordsm[i] = pcoords[i];
  }
  pcoordsp[1] += eps;
  cell->InterpolateFunctions(pcoordsp, valp);

  pcoordsm[1] -= eps;
  cell->InterpolateFunctions(pcoordsm, valm);

  for (svtkIdType idx = 0; idx < nPoints; idx++)
  {
    derivs[nPoints + idx] = (valp[idx] - valm[idx]) / (2. * eps);
  }

  if (cell->GetCellDimension() == 3)
  {
    for (svtkIdType i = 0; i < 3; i++)
    {
      pcoordsp[i] = pcoordsm[i] = pcoords[i];
    }
    pcoordsp[2] += eps;
    cell->InterpolateFunctions(pcoordsp, valp);

    pcoordsm[2] -= eps;
    cell->InterpolateFunctions(pcoordsm, valm);

    for (svtkIdType idx = 0; idx < nPoints; idx++)
    {
      derivs[2 * nPoints + idx] = (valp[idx] - valm[idx]) / (2. * eps);
    }
  }

  delete[] valp;
  delete[] valm;
}

int TestInterpolationDerivs(svtkSmartPointer<svtkLagrangeWedge> cell, double eps = SVTK_EPSILON)
{
  int numPts = cell->GetNumberOfPoints();
  int dim = cell->GetCellDimension();
  double* derivs = new double[dim * numPts];
  double* derivs_n = new double[dim * numPts];
  double* coords = cell->GetParametricCoords();
  int r = 0;
  for (int i = 0; i < numPts; ++i)
  {
    double* point = coords + 3 * i;
    double sum = 0.;
    cell->InterpolateDerivs(point, derivs);
    InterpolateDerivsNumeric(cell, point, derivs_n, 1.e-10);
    for (int j = 0; j < dim * numPts; j++)
    {
      sum += derivs[j];
      if (fabs(derivs[j] - derivs_n[j]) >
        1.e-5 * (fabs(derivs[j]) > numPts ? fabs(derivs[j]) : numPts))
      {
        std::cout << j << " is different from numeric! " << derivs[j] << " " << derivs_n[j] << " "
                  << fabs(derivs[j] - derivs_n[j]) << std::endl;
        ++r;
      }
    }
    if (fabs(sum) > eps * numPts)
    {
      std::cout << "nonzero! " << sum << std::endl;
      ++r;
    }
  }

  // Let's test zero condition on the center point:
  double center[3];
  cell->GetParametricCenter(center);
  cell->InterpolateDerivs(center, derivs);
  double sum = 0.;
  for (int j = 0; j < dim * numPts; j++)
  {
    sum += derivs[j];
  }
  if (fabs(sum) > eps)
  {
    std::cout << "center: nonzero!" << std::endl;
    ++r;
  }

  delete[] derivs;
  delete[] derivs_n;
  return r;
}

void ViewportRange(int testNum, double* range)
{
  range[0] = .25 * (testNum % 4);
  range[1] = range[0] + .25;
  range[2] = .25 * (testNum / 4);
  range[3] = range[2] + .25;
}

void RandomCircle(
  svtkMinimalStandardRandomSequence* sequence, double radius, double* offset, double* value)
{
  double theta = 2. * svtkMath::Pi() * sequence->GetValue();
  sequence->Next();
  value[0] = radius * cos(theta) + offset[0];
  value[1] = radius * sin(theta) + offset[1];
}

void RandomSphere(
  svtkMinimalStandardRandomSequence* sequence, double radius, double* offset, double* value)
{
  double theta = 2. * svtkMath::Pi() * sequence->GetValue();
  sequence->Next();
  double phi = svtkMath::Pi() * sequence->GetValue();
  sequence->Next();
  value[0] = radius * cos(theta) * sin(phi) + offset[0];
  value[1] = radius * sin(theta) * sin(phi) + offset[1];
  value[2] = radius * cos(phi) + offset[2];
}

static int testNum = 0;

svtkIdType IntersectWithCell(unsigned nTest, svtkMinimalStandardRandomSequence* sequence,
  bool threeDimensional, double radius, double* offset, svtkCell* cell
#ifdef VISUAL_DEBUG
  ,
  svtkSmartPointer<svtkRenderWindow> renderWindow
#endif
)
{
  double p[2][3];
  p[0][2] = p[1][2] = 0.;
  double tol = 1.e-7;
  double t;
  double intersect[3];
  double pcoords[3];
  int subId;
  svtkIdType counter = 0;

  svtkSmartPointer<svtkPoints> points = svtkSmartPointer<svtkPoints>::New();
  svtkSmartPointer<svtkCellArray> vertices = svtkSmartPointer<svtkCellArray>::New();

  for (unsigned i = 0; i < nTest; i++)
  {
    if (threeDimensional)
    {
      RandomSphere(sequence, radius, offset, p[0]);
      RandomSphere(sequence, radius, offset, p[1]);
    }
    else
    {
      RandomCircle(sequence, radius, offset, p[0]);
      RandomCircle(sequence, radius, offset, p[1]);
    }

    if (cell->IntersectWithLine(p[0], p[1], tol, t, intersect, pcoords, subId))
    {
      counter++;
      svtkIdType pid = points->InsertNextPoint(intersect);
      vertices->InsertNextCell(1, &pid);
    }
  }

#ifdef DEBUG_POINTS
  {
    svtkSmartPointer<svtkLagrangeWedge> wed = CreateWedge(18);
    svtkNew<svtkPolyData> pts;
    pts->SetPoints(points);
    pts->SetVerts(vertices);
    svtkNew<svtkXMLPolyDataWriter> w;
    w->SetFileName("intersection.vtp");
    w->SetInputData(pts.GetPointer());
    w->Write();
  }
#endif

#ifdef VISUAL_DEBUG
  svtkSmartPointer<svtkCamera> camera = svtkSmartPointer<svtkCamera>::New();
  camera->SetPosition(2, 2, 2);
  camera->SetFocalPoint(offset[0], offset[1], offset[2]);

  svtkSmartPointer<svtkRenderer> renderer = svtkSmartPointer<svtkRenderer>::New();
  renderer->SetActiveCamera(camera);
  renderWindow->AddRenderer(renderer);
  double dim[4];
  ViewportRange(testNum++, dim);
  renderer->SetViewport(dim[0], dim[2], dim[1], dim[3]);

  svtkSmartPointer<svtkPolyData> point = svtkSmartPointer<svtkPolyData>::New();

  point->SetPoints(points);
  point->SetVerts(vertices);

  svtkSmartPointer<svtkPolyDataMapper> mapper = svtkSmartPointer<svtkPolyDataMapper>::New();
  mapper->SetInputData(point);

  svtkSmartPointer<svtkActor> actor = svtkSmartPointer<svtkActor>::New();
  actor->SetMapper(mapper);
  renderer->AddActor(actor);
  renderer->ResetCamera();

  renderWindow->Render();
#endif

  return counter;
}

svtkIdType TestClip(svtkCell* cell
#ifdef VISUAL_DEBUG
  ,
  svtkSmartPointer<svtkRenderWindow> renderWindow
#endif
)

{
  svtkSmartPointer<svtkUnstructuredGrid> unstructuredGrid =
    svtkSmartPointer<svtkUnstructuredGrid>::New();
  unstructuredGrid->SetPoints(cell->GetPoints());

  svtkSmartPointer<svtkCellArray> cellArray = svtkSmartPointer<svtkCellArray>::New();
  cellArray->InsertNextCell(cell);
  unstructuredGrid->SetCells(cell->GetCellType(), cellArray);

  svtkSmartPointer<svtkDoubleArray> radiant = svtkSmartPointer<svtkDoubleArray>::New();
  radiant->SetName("Distance from Origin");
  radiant->SetNumberOfTuples(cell->GetPointIds()->GetNumberOfIds());

  double maxDist = 0.;
  for (svtkIdType i = 0; i < cell->GetPointIds()->GetNumberOfIds(); i++)
  {
    double xyz[3];
    cell->GetPoints()->GetPoint(i, xyz);
    double dist = sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);
    radiant->SetTypedTuple(i, &dist);
    maxDist = (dist > maxDist ? dist : maxDist);
  }

  unstructuredGrid->GetPointData()->AddArray(radiant);
  unstructuredGrid->GetPointData()->SetScalars(radiant);

  svtkSmartPointer<svtkClipDataSet> clip = svtkSmartPointer<svtkClipDataSet>::New();
  clip->SetValue(maxDist * .5);
  clip->SetInputData(unstructuredGrid);

  svtkSmartPointer<svtkDataSetSurfaceFilter> surfaceFilter =
    svtkSmartPointer<svtkDataSetSurfaceFilter>::New();
  surfaceFilter->SetInputConnection(clip->GetOutputPort());
  surfaceFilter->Update();
  svtkPolyData* polydata = surfaceFilter->GetOutput();

#ifdef VISUAL_DEBUG
  svtkSmartPointer<svtkCamera> camera = svtkSmartPointer<svtkCamera>::New();
  camera->SetPosition(2. * maxDist, 0, -3. * maxDist);
  camera->SetFocalPoint(0, 0, 0);

  svtkSmartPointer<svtkRenderer> renderer = svtkSmartPointer<svtkRenderer>::New();
  renderer->SetActiveCamera(camera);
  renderWindow->AddRenderer(renderer);
  double dim[4];
  ViewportRange(testNum++, dim);
  renderer->SetViewport(dim[0], dim[2], dim[1], dim[3]);

  svtkSmartPointer<svtkPolyDataMapper> mapper = svtkSmartPointer<svtkPolyDataMapper>::New();
  mapper->SetInputData(polydata);
  mapper->SetScalarRange(maxDist * .5, maxDist);

  svtkSmartPointer<svtkActor> actor = svtkSmartPointer<svtkActor>::New();
  actor->SetMapper(mapper);
  renderer->AddActor(actor);

  renderWindow->Render();
#endif

  return polydata->GetNumberOfPoints();
}
}

int TestLagrangeWedge(int argc, char* argv[])
{
#ifdef VISUAL_DEBUG
  svtkSmartPointer<svtkRenderWindow> renderWindow = svtkSmartPointer<svtkRenderWindow>::New();
  renderWindow->SetSize(500, 500);

  svtkSmartPointer<svtkRenderWindowInteractor> renderWindowInteractor =
    svtkSmartPointer<svtkRenderWindowInteractor>::New();

  renderWindowInteractor->SetRenderWindow(renderWindow);
#endif

  int r = 0;

  // empirically determined values, verified visually
  static const svtkIdType nIntersections = 107;
  static const svtkIdType nClippedElems[8] = { 0, 8, 30, 61, 110, 168, 242, 39 };

  svtkIdType nPointsForOrder[8];
  nPointsForOrder[0] = -1;

  for (svtkIdType order = 1; order < 7; ++order)
  {
    nPointsForOrder[order] = (order + 1) * (order + 2) * (order + 1) / 2;
  }
  nPointsForOrder[7] = 21;

  for (svtkIdType order = 1; order <= 7; ++order)
  {
    svtkSmartPointer<svtkLagrangeWedge> t = CreateWedge(nPointsForOrder[order]);

#ifdef DEBUG_POINTS
    {
      svtkNew<svtkPolyData> p;
      p->SetPoints(t->GetPoints());
      svtkNew<svtkXMLPolyDataWriter> w;
      w->SetFileName("test.vtp");
      w->SetInputData(p.GetPointer());
      w->Write();
    }
#endif

    r += TestInterpolationFunction(t);
    if (r)
    {
      std::cout << "Order " << order << " function failed!" << std::endl;
      break;
    }
    r += TestInterpolationDerivs(t);
    if (r)
    {
      std::cout << "Order " << order << " derivs failed!" << std::endl;
      break;
    }

    {
      svtkMinimalStandardRandomSequence* sequence = svtkMinimalStandardRandomSequence::New();

      sequence->SetSeed(1);

      unsigned nTest = 5.e2;
      double radius = 1.5;
      double center[3] = { 0.5, 0.5, 0. };
      svtkIdType nHits = IntersectWithCell(nTest, sequence, true, radius, center, t
#ifdef VISUAL_DEBUG
        ,
        renderWindow
#endif
      );

      sequence->Delete();

      r += (nHits == nIntersections) ? 0 : 1;

      if (r)
      {
        std::cout << "Order " << order << " intersection failed!" << std::endl;
        break;
      }
    }

    {
      svtkIdType nClippedElements = TestClip(t
#ifdef VISUAL_DEBUG
        ,
        renderWindow
#endif
      );
      r += (nClippedElements == nClippedElems[order]) ? 0 : 1;
      if (r)
      {
        std::cout << "Order " << order << " clip failed!" << std::endl;
        break;
      }
    }
  }

#ifdef VISUAL_DEBUG
  for (svtkIdType i = testNum; i < 16; i++)
  {
    svtkSmartPointer<svtkRenderer> renderer = svtkSmartPointer<svtkRenderer>::New();
    renderWindow->AddRenderer(renderer);
    double dim[4];
    ViewportRange(testNum++, dim);
    renderer->SetViewport(dim[0], dim[2], dim[1], dim[3]);
    renderer->SetBackground(0., 0., 0.);
  }

  renderWindowInteractor->Initialize();

  int retVal = svtkRegressionTestImage(renderWindow.GetPointer());
  if (retVal == svtkRegressionTester::DO_INTERACTOR)
  {
    renderWindowInteractor->Start();
    retVal = svtkRegressionTester::PASSED;
  }

  r += (retVal == svtkRegressionTester::PASSED) ? 0 : 1;
#endif

  return r;
}
