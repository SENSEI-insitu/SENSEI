#include "VTKmContourCompositeAnalysis.h"
#include "DataAdaptor.h"
#include "CinemaHelper.h"
#include <Timer.h>

#include <vtkActor.h>
#include <vtkCompositePolyDataMapper.h>
#include <vtkContourFilter.h>
#include <vtkExtractSurface.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>

#include <algorithm>
#include <vector>

namespace sensei
{

struct PipelineHandler
{
    std::vector<vtkSmartPointer<vtkContourFilter>> Contours;
    std::vector<vtkSmartPointer<vtkExtractSurface>> ExtractSurface;
    std::vector<vtkSmartPointer<vtkCompositePolyDataMapper>> Mappers;
    std::vector<vtkSmartPointer<vtkActor>> Actors;

    bool Created;

    PipelineHandler() : Created(false) {}
    ~PipelineHandler() {}
};

//-----------------------------------------------------------------------------
senseiNewMacro(VTKmContourCompositeAnalysis);

//-----------------------------------------------------------------------------
VTKmContourCompositeAnalysis::VTKmContourCompositeAnalysis() : Communicator(MPI_COMM_WORLD)
{
    this->Helper = new CinemaHelper();
    this->Pipeline = new PipelineHandler();
}

//-----------------------------------------------------------------------------
VTKmContourCompositeAnalysis::~VTKmContourCompositeAnalysis()
{
    delete this->Helper;
    delete this->Pipeline;
}

//-----------------------------------------------------------------------------
void VTKmContourCompositeAnalysis::Initialize(
  MPI_Comm comm, const std::string& workingDirectory, int* imageSize,
  const std::string& contours, const std::string& camera)
{
  this->Communicator = comm;
  this->Helper->SetImageSize(imageSize[0], imageSize[1]);
  this->Helper->SetWorkingDirectory(workingDirectory);
  this->Helper->SetExportType("sorted-composite");

  this->Helper->SetCameraConfig(camera);
  this->Helper->SetContours(contours);
}

//-----------------------------------------------------------------------------
void VTKmContourCompositeAnalysis::AddContour(double value)
{
    vtkNew<vtkContourFilter> contour;
    vtkNew<vtkExtractSurface> surface;
    vtkNew<vtkCompositePolyDataMapper> mapper;
    vtkNew<vtkActor> actor;

    surface->SetInputConnection(contour->GetOutputPort());
    mapper->SetInputConnection(surface->GetOutputPort());
    actor->SetMapper(mapper);

    this->Pipeline->Contours.push_back(contour.GetPointer());
    this->Pipeline->ExtractSurface.push_back(surface.GetPointer());
    this->Pipeline->Mappers.push_back(mapper.GetPointer());
    this->Pipeline->Actors.push_back(actor.GetPointer());
}

//-----------------------------------------------------------------------------
bool VTKmContourCompositeAnalysis::Execute(DataAdaptor* data)
{
  timer::MarkEvent mark("VTKmContourCompositeAnalysis::execute");

  vtkDataObject* mesh = data->GetMesh(/*structure_only*/true);
  if (mesh == NULL || !data->AddArray(mesh, vtkDataObject::FIELD_ASSOCIATION_CELLS, "data")) // FIXME data name
    {
    return false;
    }

  // Create pipeline if needed
  if (!this->Pipeline->Created)
    {
    this->Pipeline->Created = true;

    int nbContourToCreate = this->Helper->GetNumberOfContours();
    for (int idx = 0; idx < nbContourToCreate; idx++)
        {
        this->AddContour(this->Helper->GetContourValue(idx));
        }
    }

  // Update pipeline input
  std::vector<vtkSmartPointer<vtkContourFilter>>::iterator contourIter;
  for (contourIter = this->Pipeline->Contours.begin(); contourIter != this->Pipeline->Contours.end();++contourIter)
    {
    (*contourIter)->SetInputData(mesh);
    }

  // Render contours
  cout << "Done..." << endl;

  return true;
}

}
