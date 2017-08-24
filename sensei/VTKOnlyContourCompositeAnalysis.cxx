#include "VTKOnlyContourCompositeAnalysis.h"
#include "DataAdaptor.h"
#include "CinemaHelper.h"
#include <Timer.h>

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCameraPass.h>
#include <vtkCompositePolyDataMapper.h>
#include <vtkContourFilter.h>
#include <vtkExtractSurface.h>
#include <vtkIceTCompositePass.h>
#include <vtkLightsPass.h>
#include <vtkMPICommunicator.h>
#include <vtkMPIController.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkOpaquePass.h>
#include <vtkPieceScalars.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderPassCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSequencePass.h>
#include <vtkSmartPointer.h>
#include <vtkSynchronizedRenderers.h>
#include <vtkSynchronizedRenderWindows.h>
#include <vtkUnsignedCharArray.h>

#include <vtkDataSetMapper.h>
#include <vtkCellDataToPointData.h>
#include <vtkCompositeDataGeometryFilter.h>

#include <vtkLightingMapPass.h>

#include <algorithm>
#include <vector>

namespace sensei
{

struct PipelineHandler
{
    std::vector<vtkSmartPointer<vtkCellDataToPointData>> Cell2Point;
    std::vector<vtkSmartPointer<vtkContourFilter>> Contours;
    std::vector<vtkSmartPointer<vtkCompositeDataGeometryFilter>> ExtractSurface;
    std::vector<vtkSmartPointer<vtkCompositePolyDataMapper>> Mappers;
    std::vector<vtkSmartPointer<vtkActor>> Actors;

    vtkSmartPointer<vtkIceTCompositePass> IceTCompositePass;
    vtkSmartPointer<vtkLightingMapPass> LightingMapPass;
    vtkSmartPointer<vtkCameraPass> CameraPass;
    vtkSmartPointer<vtkRenderer> Renderer;
    vtkSmartPointer<vtkRenderWindow> RenderWindow;
    vtkSmartPointer<vtkSynchronizedRenderWindows> SynchRenderWindow;
    vtkSmartPointer<vtkSynchronizedRenderers> SyncRenderers;
    bool RenderingCreated;
    bool Created;

    PipelineHandler() : Created(false), RenderingCreated(false) {}
    ~PipelineHandler() {}

    void CreateRendering(vtkMultiProcessController* controller)
    {
      if (this->RenderingCreated)
        {
        return;
        }
      this->RenderingCreated = true;
      this->Renderer = vtkSmartPointer<vtkRenderer>::New();
      this->Renderer->SetInteractive(0);

      std::vector<vtkSmartPointer<vtkActor>>::iterator actorIter;
      for (actorIter = this->Actors.begin(); actorIter != this->Actors.end(); ++actorIter)
        {
        this->Renderer->AddActor(*actorIter);
        }

      this->RenderWindow = vtkSmartPointer<vtkRenderWindow>::New();
      this->RenderWindow->AddRenderer(this->Renderer);

      //---------------------------------------------------------------------------
      // the rendering passes
      vtkCameraPass* cameraP = vtkCameraPass::New();
      vtkSequencePass* seq = vtkSequencePass::New();
      vtkOpaquePass* opaque = vtkOpaquePass::New();
      vtkLightsPass* lights = vtkLightsPass::New();

      vtkRenderPassCollection* passes = vtkRenderPassCollection::New();
      passes->AddItem(lights);
      passes->AddItem(opaque);
      seq->SetPasses(passes);

      // Each processes only has part of the data, so each process will render only
      // part of the data. To ensure that root node gets a composited result (or in
      // case of tile-display mode all nodes show part of tile), we use
      // vtkIceTCompositePass.
      vtkIceTCompositePass* iceTPass = vtkIceTCompositePass::New();
      iceTPass->SetController(controller);
      iceTPass->SetRenderPass(seq);
      cameraP->SetDelegatePass(iceTPass);
      this->Renderer->SetPass(cameraP);

      // Keep it to capture zBuffer
      this->IceTCompositePass = iceTPass;
      this->LightingMapPass = vtkSmartPointer<vtkLightingMapPass>::New();
      this->CameraPass = cameraP;

      iceTPass->Delete();
      opaque->Delete();
      seq->Delete();
      passes->Delete();
      cameraP->Delete();
      lights->Delete();

      //---------------------------------------------------------------------------
      // In parallel configurations, typically one node acts as the driver i.e. the
      // node where the user interacts with the window e.g. mouse interactions,
      // resizing windows etc. Typically that's the root-node.
      // To ensure that the window parameters get propagated to all processes from
      // the root node, we use the vtkSynchronizedRenderWindows.
      this->SynchRenderWindow =
        vtkSmartPointer<vtkSynchronizedRenderWindows>::New();
      this->SynchRenderWindow->SetRenderWindow(this->RenderWindow);
      this->SynchRenderWindow->SetParallelController(controller);

      // Since there could be multiple render windows that could be synced
      // separately, to identify the windows uniquely among all processes, we need
      // to give each vtkSynchronizedRenderWindows a unique id that's consistent
      // across all the processes.
      this->SynchRenderWindow->SetIdentifier(1);

      // Now we need to ensure that the render is synchronized as well. This is
      // essential to ensure all processes have the same camera orientation etc.
      // This is done using the vtkSynchronizedRenderers class.
      this->SyncRenderers =
        vtkSmartPointer<vtkSynchronizedRenderers>::New();
      this->SyncRenderers->SetRenderer(this->Renderer);
      this->SyncRenderers->SetParallelController(controller);
    }

    // void UseLuminancePass(bool use)
    // {
    //   this->CameraPass->SetDelegatePass(use ? this->LightingMapPass : this->IceTCompositePass);
    // }

    void Render(vtkMultiProcessController* controller)
    {
      if (controller->GetLocalProcessId() == 0)
        {
        this->RenderWindow->Render();
        controller->TriggerBreakRMIs();
        controller->Barrier();
        }
      else
        {
        controller->ProcessRMIs();
        controller->Barrier();
        }
    }
};

//-----------------------------------------------------------------------------
senseiNewMacro(VTKOnlyContourCompositeAnalysis);

//-----------------------------------------------------------------------------
VTKOnlyContourCompositeAnalysis::VTKOnlyContourCompositeAnalysis() : Communicator(MPI_COMM_WORLD), Helper(NULL)
{
    this->Pipeline = new PipelineHandler();
}

//-----------------------------------------------------------------------------
VTKOnlyContourCompositeAnalysis::~VTKOnlyContourCompositeAnalysis()
{
    delete this->Helper;
    delete this->Pipeline;
}

//-----------------------------------------------------------------------------
void VTKOnlyContourCompositeAnalysis::Initialize(
  MPI_Comm comm, const std::string& workingDirectory, int* imageSize,
  const std::string& contours, const std::string& camera)
{
  this->Communicator = comm;

  vtkNew<vtkMPIController> con;
  con->Initialize(0, 0, 1);
  vtkMultiProcessController::SetGlobalController(con.GetPointer());
  con->Register(NULL); // Keep ref

  this->Helper = new CinemaHelper();
  this->Helper->SetImageSize(imageSize[0], imageSize[1]);
  this->Helper->SetWorkingDirectory(workingDirectory);
  this->Helper->SetCameraConfig(camera);
  this->Helper->SetContours(contours);
  this->Helper->SetExportType("sorted-composite");
}

//-----------------------------------------------------------------------------
void VTKOnlyContourCompositeAnalysis::AddContour(double value)
{
    vtkNew<vtkCellDataToPointData> cell2Point;
    vtkNew<vtkContourFilter> contour;
    vtkNew<vtkCompositeDataGeometryFilter> surface;
    vtkNew<vtkCompositePolyDataMapper> mapper;
    vtkNew<vtkActor> actor;

    contour->SetInputConnection(cell2Point->GetOutputPort());
    surface->SetInputConnection(contour->GetOutputPort());
    mapper->SetInputConnection(surface->GetOutputPort());
    actor->SetMapper(mapper);

    contour->SetNumberOfContours(1);
    contour->SetValue(0, value);

    this->Pipeline->Cell2Point.push_back(cell2Point.GetPointer());
    this->Pipeline->Contours.push_back(contour.GetPointer());
    this->Pipeline->ExtractSurface.push_back(surface.GetPointer());
    this->Pipeline->Mappers.push_back(mapper.GetPointer());
    this->Pipeline->Actors.push_back(actor.GetPointer());

    this->Helper->RegisterLayer("Contour", actor.GetPointer(), value);
}

//-----------------------------------------------------------------------------
bool VTKOnlyContourCompositeAnalysis::Execute(DataAdaptor* data)
{
  timer::MarkEvent mark("VTKOnlyContourCompositeAnalysis::execute");
  vtkMultiProcessController* controller = vtkMultiProcessController::GetGlobalController();

  this->Helper->AddTimeEntry();

  vtkDataObject* mesh = data->GetMesh(/*structure_only*/true);
  bool dataError = !data->AddArray(mesh, vtkDataObject::FIELD_ASSOCIATION_CELLS, "data");
  // FIXME: dataError is true on satelites for no reason => BUG!!!
  // if (mesh == NULL || dataError) // FIXME data name
  //   {
  //   cout << "Exit due to mesh: " << mesh << " for " << controller->GetLocalProcessId() << endl;
  //   return false;
  //   }

  // Create pipeline if needed
  if (!this->Pipeline->Created)
    {
    timer::MarkStartEvent("Pipeline creation");
    this->Pipeline->Created = true;

    int nbContourToCreate = this->Helper->GetNumberOfContours();
    for (int idx = 0; idx < nbContourToCreate; idx++)
        {
        this->AddContour(this->Helper->GetContourValue(idx));
        }
    timer::MarkEndEvent("Pipeline creation");
    }

  // Update pipeline input
  timer::MarkStartEvent("Update pipeline input");
  std::vector<vtkSmartPointer<vtkCellDataToPointData>>::iterator filterIter;
  for (filterIter = this->Pipeline->Cell2Point.begin(); filterIter != this->Pipeline->Cell2Point.end();++filterIter)
    {
    (*filterIter)->SetInputData(mesh);
    }
  timer::MarkEndEvent("Update pipeline input");

  if (!this->Pipeline->RenderingCreated)
    {
    timer::MarkStartEvent("Create rendering");
    this->Pipeline->CreateRendering(controller);
    timer::MarkEndEvent("Create rendering");
    }

  timer::MarkStartEvent("Cinema Composite contours export");
  int nbCamera = this->Helper->GetNumberOfCameraPositions();
  for (int i = 0; i < nbCamera; i++)
  {
    this->Helper->ApplyCameraPosition(this->Pipeline->Renderer->GetActiveCamera(), i);
    // this->Pipeline->Render(controller);
    this->Helper->CaptureSortedCompositeData(this->Pipeline->RenderWindow, this->Pipeline->Renderer, this->Pipeline->CameraPass, this->Pipeline->IceTCompositePass, this->Pipeline->LightingMapPass);
  }
  this->Helper->WriteMetadata();
  timer::MarkEndEvent("Cinema Composite contours export");

  return true;
}

}
