/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestPolygon.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// .NAME
// .SECTION Description
// this program tests the BoundedTriangulate method in Polygon

#include "svtkIdList.h"
#include "svtkNew.h"
#include "svtkPoints.h"
#include "svtkPolygon.h"

// #define VISUAL_DEBUG 1

#ifdef VISUAL_DEBUG
#include <svtkActor.h>
#include <svtkCellArray.h>
#include <svtkPolyData.h>
#include <svtkPolyDataMapper.h>
#include <svtkProperty.h>
#include <svtkRenderWindow.h>
#include <svtkRenderWindowInteractor.h>
#include <svtkRenderer.h>
#endif

#include <vector>

bool ValidTessellation(svtkPolygon* polygon, svtkIdList* outTris)
{
  // Check that there are enough triangles
  if (outTris->GetNumberOfIds() / 3 != polygon->GetNumberOfPoints() - 2)
  {
    return false;
  }

  // Check that all of the edges of the polygon are represented
  std::vector<bool> edges(polygon->GetNumberOfPoints(), false);

  for (int i = 0; i < polygon->GetNumberOfPoints(); i++)
  {
    svtkIdType edge[2] = { polygon->GetPointId(i),
      polygon->GetPointId((i + 1) % polygon->GetNumberOfPoints()) };
    for (int j = 0; j < outTris->GetNumberOfIds(); j += 3)
    {
      for (int k = 0; k < 3; k++)
      {
        svtkIdType triedge[2] = { polygon->PointIds->GetId(outTris->GetId(j + k)),
          polygon->PointIds->GetId(outTris->GetId(j + ((k + 1) % 3))) };

        if ((triedge[0] == edge[0] && triedge[1] == edge[1]) ||
          (triedge[0] == edge[1] && triedge[1] == edge[0]))
        {
          edges[i] = true;
          break;
        }
      }
      if (edges[i] == true)
        break;
    }
    if (edges[i] == false)
      break;
  }

  for (std::size_t i = 0; i < edges.size(); i++)
  {
    if (!edges[i])
    {
      return false;
    }
  }

  return true;
}

int TestPolygonBoundedTriangulate(int, char*[])
{
  svtkNew<svtkPolygon> polygon;

  polygon->GetPoints()->InsertNextPoint(125.703, 149.84, 45.852);
  polygon->GetPoints()->InsertNextPoint(126.438, 147.984, 44.3112);
  polygon->GetPoints()->InsertNextPoint(126.219, 148.174, 44.4463);
  polygon->GetPoints()->InsertNextPoint(126.196, 148.202, 44.4683);
  polygon->GetPoints()->InsertNextPoint(126.042, 148.398, 44.6184);
  polygon->GetPoints()->InsertNextPoint(125.854, 148.635, 44.8);
  polygon->GetPoints()->InsertNextPoint(125.598, 148.958, 45.0485);
  polygon->GetPoints()->InsertNextPoint(125.346, 149.24, 45.26);
  polygon->GetPoints()->InsertNextPoint(125.124, 149.441, 45.4041);

  polygon->GetPointIds()->SetNumberOfIds(polygon->GetPoints()->GetNumberOfPoints());
  for (svtkIdType i = 0; i < polygon->GetPoints()->GetNumberOfPoints(); i++)
  {
    polygon->GetPointIds()->SetId(i, i);
  }

  svtkNew<svtkIdList> outTris;

  int success = polygon->BoundedTriangulate(outTris, 1.e-2);

  if (!success || !ValidTessellation(polygon, outTris))
  {
    cerr << "ERROR:  svtkPolygon::BoundedTriangulate should triangulate this polygon" << endl;
    return EXIT_FAILURE;
  }

#ifdef VISUAL_DEBUG
  svtkNew<svtkCellArray> triangles;
  for (svtkIdType i = 0; i < outTris->GetNumberOfIds(); i += 3)
  {
    svtkIdType t[3] = { outTris->GetId(i), outTris->GetId(i + 1), outTris->GetId(i + 2) };
    triangles->InsertNextCell(3, t);
  }

  svtkNew<svtkPolyData> polydata;
  polydata->SetPoints(polygon->GetPoints());
  polydata->SetPolys(triangles);

  svtkNew<svtkPolyDataMapper> mapper;
  mapper->SetInputData(polydata);

  svtkNew<svtkActor> actor;
  actor->SetMapper(mapper);
  actor->GetProperty()->SetRepresentationToWireframe();

  // Create a renderer, render window, and an interactor
  svtkNew<svtkRenderer> renderer;
  svtkNew<svtkRenderWindow> renderWindow;
  renderWindow->AddRenderer(renderer);
  svtkNew<svtkRenderWindowInteractor> renderWindowInteractor;
  renderWindowInteractor->SetRenderWindow(renderWindow);

  // Add the actors to the scene
  renderer->AddActor(actor);
  renderer->SetBackground(.1, .2, .4);

  // Render and interact
  renderWindow->Render();
  renderWindowInteractor->Start();
#endif

  return EXIT_SUCCESS;
}
