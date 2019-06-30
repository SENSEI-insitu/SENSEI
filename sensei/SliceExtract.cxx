#include "SliceExtract.h"
#include "MeshMetadataMap.h"
#include "PlanarSlicePartitioner.h"
#include "IsoSurfacePartitioner.h"
#include "InTransitDataAdaptor.h"
#include "VTKPosthocIO.h"
#include "VTKDataAdaptor.h"
#include "VTKUtils.h"
#include "Timer.h"
#include "Error.h"

#include <vtkObjectFactory.h>
#include <vtkCellData.h>
#include <vtkDataObjectAlgorithm.h>
#include <vtkCellDataToPointData.h>
#include <vtkContourFilter.h>
#include <vtkCutter.h>
#include <vtkPlane.h>
#include <vtkDataObject.h>
#include <vtkCompositeDataSet.h>
#include <vtkCompositeDataIterator.h>
#include <vtkMultiBlockDataSet.h>

using vtkDataObjectAlgorithmPtr = vtkSmartPointer<vtkDataObjectAlgorithm>;
using vtkCellDataToPointDataPtr = vtkSmartPointer<vtkCellDataToPointData>;
using vtkContourFilterPtr = vtkSmartPointer<vtkContourFilter>;
using vtkCutterPtr = vtkSmartPointer<vtkCutter>;
using vtkPlanePtr = vtkSmartPointer<vtkPlane>;

namespace sensei
{

struct SliceExtract::InternalsType
{
  InternalsType() : Operation(OP_PLANAR_SLICE)
  {
    this->SlicePartitioner = PlanarSlicePartitioner::New();
    this->IsoValPartitioner = IsoSurfacePartitioner::New();
    this->Writer = VTKPosthocIOPtr::New();
  }

  int Operation;
  int NumIsoValues;
  std::vector<double> IsoValues;
  std::array<double,3> Point;
  std::array<double,3> Normal;
  DataRequirements Requirements;
  IsoSurfacePartitionerPtr IsoValPartitioner;
  PlanarSlicePartitionerPtr SlicePartitioner;
  VTKPosthocIOPtr Writer;
};



//-----------------------------------------------------------------------------
senseiNewMacro(SliceExtract);

// --------------------------------------------------------------------------
SliceExtract::SliceExtract()
{
  this->Internals = new InternalsType;
}

// --------------------------------------------------------------------------
SliceExtract::~SliceExtract()
{
  delete this->Internals;
}

// --------------------------------------------------------------------------
int SliceExtract::SetOperation(int op)
{
  if ((op != OP_PLANAR_SLICE) || (op != OP_ISO_SURFACE))
    {
    SENSEI_ERROR("Invalid operation " << op)
    return -1;
    }

  this->Internals->Operation = op;
  return 0;
}

// --------------------------------------------------------------------------
int SliceExtract::SetOperation(std::string opStr)
{
  unsigned int n = opStr.size();
  for (unsigned int i = 0; i < n; ++i)
    opStr[i] = tolower(opStr[i]);

  int op = 0;
  if (opStr == "planar_slice")
    {
    op = OP_PLANAR_SLICE;
    }
  else if (opStr == "iso_surface")
    {
    op = OP_ISO_SURFACE;
    }
  else
    {
    SENSEI_ERROR("invalid operation \"" << opStr << "\"")
    return -1;
    }

  this->Internals->Operation = op;
  return 0;
}

// --------------------------------------------------------------------------
void SliceExtract::SetIsoValues(const std::string &mesh,
  const std::string &arrayName, int arrayCentering,
  const std::vector<double> &vals)
{
  this->Internals->IsoValPartitioner->SetIsoValues(mesh,
    arrayName, arrayCentering, vals);
}

// --------------------------------------------------------------------------
void SliceExtract::SetNumberOfIsoValues(const std::string &mesh,
  const std::string &array, int centering, int numIsos)
{
  // TODO
  SENSEI_ERROR("Not implemented")
  (void)mesh;
  (void)array;
  (void)centering;
  (void)numIsos;
}

// --------------------------------------------------------------------------
void SliceExtract::SetVerbose(int val)
{
  this->AnalysisAdaptor::SetVerbose(val);
  this->Internals->SlicePartitioner->SetVerbose(val);
  this->Internals->IsoValPartitioner->SetVerbose(val);
}

// --------------------------------------------------------------------------
int SliceExtract::SetWriterOutputDir(const std::string &outputDir)
{
  return this->Internals->Writer->SetOutputDir(outputDir);
}

// --------------------------------------------------------------------------
int SliceExtract::SetWriterMode(const std::string &mode)
{
  return this->Internals->Writer->SetMode(mode);
}

// --------------------------------------------------------------------------
int SliceExtract::SetWriterWriter(const std::string &writer)
{
  return this->Internals->Writer->SetWriter(writer);
}

// --------------------------------------------------------------------------
int SliceExtract::SetPoint(const std::array<double,3> &point)
{
  this->Internals->SlicePartitioner->SetPoint(point);
  return 0;
}

// --------------------------------------------------------------------------
int SliceExtract::SetNormal(const std::array<double,3> &normal)
{
  this->Internals->SlicePartitioner->SetNormal(normal);
  return 0;
}

//-----------------------------------------------------------------------------
int SliceExtract::SetDataRequirements(const DataRequirements &reqs)
{
  this->Internals->Requirements = reqs;
  return 0;
}

//-----------------------------------------------------------------------------
int SliceExtract::AddDataRequirement(const std::string &meshName,
  int association, const std::vector<std::string> &arrays)
{
  this->Internals->Requirements.AddRequirement(meshName, association, arrays);
  return 0;
}

// --------------------------------------------------------------------------
bool SliceExtract::Execute(DataAdaptor* dataAdaptor)
{
  if (this->Internals->Operation == OP_PLANAR_SLICE)
    {
    return this->ExecuteSlice(dataAdaptor);
    }
  else if (this->Internals->Operation == OP_ISO_SURFACE)
    {
    return this->ExecuteIsoSurface(dataAdaptor);
    }

  SENSEI_ERROR("Invalid operation " << this->Internals->Operation)
  return false;
}

// --------------------------------------------------------------------------
bool SliceExtract::ExecuteIsoSurface(DataAdaptor* dataAdaptor)
{
  timer::MarkEvent mark("SliceExtract::ExecuteIsoSurface");

  // get the mesh array and iso values
  std::string meshName;
  std::string arrayName;
  int arrayCentering = vtkDataObject::POINT;
  std::vector<double> isoVals;

  if (this->Internals->IsoValPartitioner->GetIsoValues(meshName,
    arrayName, arrayCentering, isoVals))
    {
    SENSEI_ERROR("Iso-values have not been provided")
    return false;
    }

  // if we are runnigng in transit, set the partitioner that will pull
  // only the blocks that intersect the slice plane
  if (InTransitDataAdaptor *itDataAdaptor = dynamic_cast<InTransitDataAdaptor*>(dataAdaptor))
    itDataAdaptor->SetPartitioner(this->Internals->IsoValPartitioner);

  // figure out what the simulation can provide
  MeshMetadataFlags flags;
  flags.SetBlockDecomp();
  flags.SetBlockArrayRange();

  MeshMetadataMap mdm;
  if (mdm.Initialize(dataAdaptor, flags))
    {
    SENSEI_ERROR("Failed to get metadata")
    return false;
    }

  // get metadata
  MeshMetadataPtr md;
  if (mdm.GetMeshMetadata(meshName, md))
    {
    SENSEI_ERROR("Failed to get metadata for mesh \"" << meshName << "\"")
    return false;
    }

  // get the mesh
  vtkCompositeDataSet *dobj = nullptr;
  if (dataAdaptor->GetMesh(meshName, false, dobj))
    {
    SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
    return false;
    }

  // add the ghost cell arrays to the mesh
  if (md->NumGhostCells && dataAdaptor->AddGhostCellsArray(dobj, meshName))
    {
    SENSEI_ERROR("Failed to get ghost cells for mesh \"" << meshName << "\"")
    return false;
    }

  // add the ghost node arrays to the mesh
  if (md->NumGhostNodes && dataAdaptor->AddGhostNodesArray(dobj, meshName))
    {
    SENSEI_ERROR("Failed to get ghost nodes for mesh \"" << meshName << "\"")
    return false;
    }

  // add the required arrays
  if (dataAdaptor->AddArray(dobj, meshName, arrayCentering, arrayName))
    {
    SENSEI_ERROR("Failed to add "
      << VTKUtils::GetAttributesName(arrayCentering)
      << " data array \"" <<arrayName << "\" to mesh \""
      << meshName << "\"")
    return false;
    }

  // compute the iso-surfaces
  vtkCompositeDataSet *sliceMesh = nullptr;
  if (this->IsoSurface(dobj, arrayName, arrayCentering, isoVals, sliceMesh))
    {
    SENSEI_ERROR("Failed to extract slice")
    return false;
    }

  // write it to disk
  std::string sliceMeshName  = meshName + "_" + arrayName + "_isos";
  long timeStep = dataAdaptor->GetDataTimeStep();
  double time = dataAdaptor->GetDataTime();
  if (this->WriteExtract(timeStep, time, sliceMeshName, sliceMesh))
    {
    SENSEI_ERROR("Failed to write the extract")
    return false;
    }

  sliceMesh->Delete();

  return true;
}
// --------------------------------------------------------------------------
bool SliceExtract::ExecuteSlice(DataAdaptor* dataAdaptor)
{
  timer::MarkEvent mark("SliceExtract::Execute");

  // require the user to tell us one or more meshes to slice
  if (this->Internals->Requirements.Empty())
    {
    SENSEI_ERROR("No mesh was specified")
    return false;
    }

  // if we are runnigng in transit, set the partitioner that will pull
  // only the blocks that intersect the slice plane
  if (InTransitDataAdaptor *itDataAdaptor = dynamic_cast<InTransitDataAdaptor*>(dataAdaptor))
    itDataAdaptor->SetPartitioner(this->Internals->SlicePartitioner);

  // figure out what the simulation can provide
  MeshMetadataFlags flags;
  flags.SetBlockDecomp();

  MeshMetadataMap mdm;
  if (mdm.Initialize(dataAdaptor, flags))
    {
    SENSEI_ERROR("Failed to get metadata")
    return false;
    }


  // loop over requested meshes, pull the arrays, take slice,
  // and finally write the result
  MeshRequirementsIterator mit =
    this->Internals->Requirements.GetMeshRequirementsIterator();

  while (mit)
    {
    const std::string &meshName = mit.MeshName();
    // get metadata
    MeshMetadataPtr md;
    if (mdm.GetMeshMetadata(meshName, md))
      {
      SENSEI_ERROR("Failed to get metadata for mesh \"" << meshName << "\"")
      return false;
      }

    // get the mesh
    vtkCompositeDataSet *dobj = nullptr;
    if (dataAdaptor->GetMesh(meshName, mit.StructureOnly(), dobj))
      {
      SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
      return false;
      }

    // add the ghost cell arrays to the mesh
    if (md->NumGhostCells && dataAdaptor->AddGhostCellsArray(dobj, meshName))
      {
      SENSEI_ERROR("Failed to get ghost cells for mesh \"" << meshName << "\"")
      return false;
      }

    // add the ghost node arrays to the mesh
    if (md->NumGhostNodes && dataAdaptor->AddGhostNodesArray(dobj, meshName))
      {
      SENSEI_ERROR("Failed to get ghost nodes for mesh \"" << meshName << "\"")
      return false;
      }

    // add the required arrays
    ArrayRequirementsIterator ait =
      this->Internals->Requirements.GetArrayRequirementsIterator(meshName);

    while (ait)
      {
      if (dataAdaptor->AddArray(dobj, meshName,
         ait.Association(), ait.Array()))
        {
        SENSEI_ERROR("Failed to add "
          << VTKUtils::GetAttributesName(ait.Association())
          << " data array \"" << ait.Array() << "\" to mesh \""
          << meshName << "\"")
        return false;
        }
      ++ait;
      }

    // compute the slice
    vtkCompositeDataSet *sliceMesh = nullptr;
    std::array<double,3> point, normal;
    this->Internals->SlicePartitioner->GetPoint(point);
    this->Internals->SlicePartitioner->GetNormal(normal);
    if (this->Slice(dobj, point, normal, sliceMesh))
      {
      SENSEI_ERROR("Failed to extract slice")
      return false;
      }

    // write it to disk
    std::string sliceMeshName  = meshName + "_slice";
    long timeStep = dataAdaptor->GetDataTimeStep();
    double time = dataAdaptor->GetDataTime();
    if (this->WriteExtract(timeStep, time, sliceMeshName, sliceMesh))
      {
      SENSEI_ERROR("Failed to write the extract")
      return false;
      }

    sliceMesh->Delete();

    ++mit;
    }

  return true;
}

// --------------------------------------------------------------------------
int SliceExtract::Finalize()
{
  timer::MarkEvent mark("SliceExtract::Finalize");
  if (this->Internals->Writer->Finalize())
    {
    SENSEI_ERROR("Failed to finalize the writer")
    return -1;
    }
  return 0;
}

// --------------------------------------------------------------------------
int SliceExtract::IsoSurface(vtkCompositeDataSet *input,
  const std::string &arrayName, int arrayCen, const std::vector<double> &vals,
  vtkCompositeDataSet *&output)
{
  // build pipeline
  vtkContourFilterPtr contour = vtkContourFilterPtr::New();
  contour->SetComputeScalars(1);

  contour->SetInputArrayToProcess(0, 0, 0,
    vtkDataObject::FIELD_ASSOCIATION_POINTS, arrayName.c_str());

  unsigned int nVals = vals.size();
  contour->SetNumberOfContours(nVals);
  for (unsigned int i = 0; i < nVals; ++i)
    contour->SetValue(i, vals[i]);

  // when processing cell data first convert to point data
  vtkCellDataToPointDataPtr cdpd;
  if (arrayCen == vtkDataObject::CELL)
    {
    cdpd = vtkCellDataToPointDataPtr::New();
    cdpd->SetPassCellData(1);
    contour->SetInputConnection(cdpd->GetOutputPort());
    }

  // allocate output
  vtkCompositeDataIterator *it = input->NewIterator();
  it->SetSkipEmptyNodes(0);

  unsigned int nBlocks = 0;
  for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextItem())
    ++nBlocks;

  vtkMultiBlockDataSet *mbds = vtkMultiBlockDataSet::New();
  mbds->SetNumberOfBlocks(nBlocks);

  // process data
  it->SetSkipEmptyNodes(1);
  for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextItem())
    {
    // get the current block
    unsigned int bid = it->GetCurrentFlatIndex() - 1;
    vtkDataObject *dobjIn = it->GetCurrentDataObject();

    // run the pipeline on the block
    if (arrayCen == vtkDataObject::CELL)
      cdpd->SetInputData(dobjIn);
    else
      contour->SetInputData(dobjIn);
    contour->SetOutput(nullptr);
    contour->Update();

    // save the extract
    vtkDataObject *dobjOut = contour->GetOutput();
    mbds->SetBlock(bid, dobjOut);
    }

  output = mbds;

  return 0;
}

// --------------------------------------------------------------------------
int SliceExtract::Slice(vtkCompositeDataSet *input,
  const std::array<double,3> &point, const std::array<double,3> &normal,
  vtkCompositeDataSet *&output)
{
  // build pipeline
  vtkCutterPtr slice = vtkCutterPtr::New();

  vtkPlanePtr plane = vtkPlanePtr::New();
  plane->SetOrigin(const_cast<double*>(point.data()));
  plane->SetNormal(const_cast<double*>(normal.data()));

  slice->SetCutFunction(plane.GetPointer());

  // allocate output
  vtkCompositeDataIterator *it = input->NewIterator();
  it->SetSkipEmptyNodes(0);

  unsigned int nBlocks = 0;
  for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextItem())
    ++nBlocks;

  vtkMultiBlockDataSet *mbds = vtkMultiBlockDataSet::New();
  mbds->SetNumberOfBlocks(nBlocks);

  // process data
  it->SetSkipEmptyNodes(1);
  for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextItem())
    {
    // get the current block
    unsigned int bid = it->GetCurrentFlatIndex() - 1;
    vtkDataObject *dobjIn = it->GetCurrentDataObject();

    // set up and run the pipeline
    slice->SetInputData(dobjIn);
    slice->SetOutput(nullptr);
    slice->Update();

    // save the extract
    vtkDataObject *dobjOut = slice->GetOutput();
    mbds->SetBlock(bid, dobjOut);
    }

  output = mbds;

  return 0;
}

// --------------------------------------------------------------------------
int SliceExtract::WriteExtract(long timeStep, double time,
  const std::string &mesh, vtkCompositeDataSet *input)
{
  VTKDataAdaptor *dataAdaptor = VTKDataAdaptor::New();

  dataAdaptor->SetDataObject(mesh, input);
  dataAdaptor->SetDataTimeStep(timeStep);
  dataAdaptor->SetDataTime(time);

  if (!this->Internals->Writer->Execute(dataAdaptor))
    {
    SENSEI_ERROR("Failed to write time step " << timeStep)
    return -1;
    }

  return 0;
}

}
