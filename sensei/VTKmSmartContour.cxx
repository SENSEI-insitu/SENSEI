#include "VTKmSmartContour.h"

#include "DataAdaptor.h"
#include "VTKDataAdaptor.h"
#include "CatalystAnalysisAdaptor.h"
#include "VTKPosthocIO.h"
#include "Timer.h"
#include "Error.h"

#include <vtkSmartPointer.h>
#include <vtkCompositeDataIterator.h>
#include <vtkCompositeDataSet.h>
#include <vtkDataSet.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkFieldData.h>
#include <vtkDataArray.h>
#include <vtkDataArrayTemplate.h>
#include <vtkObjectFactory.h>

#include <vtkImageData.h>
#include <vtkDoubleArray.h>

#if defined(USE_VTKM_CONTOUR)
// TODO
#else
#include <vtkContourFilter.h>
#endif

// Use the TBB parallel backend by default
#ifndef VTKM_DEVICE_ADAPTER
#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_TBB
#endif

#define DEBUG_TIMING

#ifdef DEBUG_TIMING
#include <sys/time.h>
#endif


#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/cont/DeviceAdapter.h>
#include <vtkm/cont/Storage.h>
#include <vtkm/filter/ContourTreeUniformPPP2.h>
#include <vtkm/worklet/contourtree_ppp2/PrintVectors.h>

#include <vtkm/worklet/contourtree_ppp2/ContourTree.h>
#include <vtkm/worklet/contourtree_ppp2/ProcessContourTree.h>
#include <vtkm/worklet/contourtree_ppp2/ProcessContourTree_Inc/Branch.h>
#include <vtkm/worklet/contourtree_ppp2/ProcessContourTree_Inc/PiecewiseLinearFunction.h>

using namespace vtkm::worklet::contourtree_ppp2;
using namespace vtkm::worklet::contourtree_ppp2::process_contourtree_inc;

template <typename NT>
using ArrayHandleType = vtkm::cont::ArrayHandle<NT, vtkm::cont::StorageTagBasic>;

template <typename NT>
using BranchType = vtkm::worklet::contourtree_ppp2::process_contourtree_inc::Branch<NT>;

#include <iostream>
using namespace std;

namespace sensei
{
class VTKmSmartContour::InternalsType
{
public:
  InternalsType() : Comm(MPI_COMM_WORLD), ScalarField(""),
    ScalarFieldAssociation(vtkDataObject::POINT), UseMarchingCubes(0),
    UsePersistenceSorter(1), NumberOfLevels(10), NumberOfComps(11),
    ContourType(0), Eps(1.0e-5), SelectMethod(0), CatalystScript(""),
    CatalystAdaptor(nullptr), OutputDir(""), IOAdaptor(nullptr) {}

  vtkDataArray* GetScalarField(vtkDataObject* dobj);

  int VTKmImage(vtkDataObject *dobj, vtkm::cont::DataSet &vtkmds);

  template<typename n_t>
  int VTKmImage(double *dx, double *x0, int nx, int ny, int nz,
    n_t *data, vtkm::cont::DataSet &ds);

  bool XZPlane(int nx, int ny, int nz);
  bool XYPlane(int nx, int ny, int nz);
  bool YZPlane(int nx, int ny, int nz);
  bool Planar(int nx, int ny, int nz);

  const char *AssociationStr(int assoc);

  const char *AssociationStr()
  { return this->AssociationStr(this->ScalarFieldAssociation); }

public:
  MPI_Comm Comm;
  std::string ScalarField;
  int ScalarFieldAssociation;
  int UseMarchingCubes;
  int UsePersistenceSorter;
  int NumberOfLevels;
  int NumberOfComps;
  int ContourType;
  double Eps;
  int SelectMethod;
  std::string CatalystScript;
  sensei::CatalystAnalysisAdaptor *CatalystAdaptor;
  std::string OutputDir;
  sensei::VTKPosthocIO *IOAdaptor;
  std::vector<double> ContourValues;
};

// --------------------------------------------------------------------------
const char *VTKmSmartContour::InternalsType::AssociationStr(int assoc)
{
  return assoc == vtkDataObject::POINT ? "point data" :
   assoc == vtkDataObject::CELL ? "cell data" : "invalid association";
}

// --------------------------------------------------------------------------
vtkDataArray* VTKmSmartContour::InternalsType::GetScalarField(vtkDataObject* dobj)
{
  if (vtkFieldData* fd = dobj->GetAttributesAsFieldData(this->ScalarFieldAssociation))
    {
    if (vtkDataArray *da = fd->GetArray(this->ScalarField.c_str()))
      {
      return da;
      }
    }
  SENSEI_ERROR("No field array named " << this->ScalarField
   << " in " << this->AssociationStr(this->ScalarFieldAssociation))
  return nullptr;
}

// --------------------------------------------------------------------------
bool VTKmSmartContour::InternalsType::Planar(int nx, int ny, int nz)
{
  return this->XZPlane(nx, ny, nz) || this->XYPlane(nx, ny, nz) ||
    this->YZPlane(nx, ny, nz);
}

// --------------------------------------------------------------------------
bool VTKmSmartContour::InternalsType::XZPlane(int nx, int ny, int nz)
{
  return ((ny == 1) && (nx > 1) && (nz > 1));
}

// --------------------------------------------------------------------------
bool VTKmSmartContour::InternalsType::XYPlane(int nx, int ny, int nz)
{
  return ((nz == 1) && (nx > 1) && (ny > 1));
}

// --------------------------------------------------------------------------
bool VTKmSmartContour::InternalsType::YZPlane(int nx, int ny, int nz)
{
  return ((nx == 1) && (ny > 1) && (nz > 1));
}

// --------------------------------------------------------------------------
template<typename n_t>
int VTKmSmartContour::InternalsType::VTKmImage(double *dx, double *x0,
  int nx, int ny, int nz, n_t *data, vtkm::cont::DataSet &ds)
{
  // build the input dataset
  vtkm::cont::DataSetBuilderUniform dsb;

  if (this->XYPlane(nx, ny, nz))
    {
    ds = dsb.Create(vtkm::Id2(nx, ny),
      vtkm::Vec<double, 2>(x0[0], x0[1]),
      vtkm::Vec<double, 2>(dx[0], dx[1]));
    }
  else if (this->XZPlane(nx, ny, nz))
    {
    ds = dsb.Create(vtkm::Id2(nx, nz),
      vtkm::Vec<double, 2>(x0[0], x0[2]),
      vtkm::Vec<double, 2>(dx[0], dx[2]));
    }
  else if (this->YZPlane(nx, ny, nz))
    {
    ds = dsb.Create(vtkm::Id2(ny, nz),
      vtkm::Vec<double, 2>(x0[1], x0[2]),
      vtkm::Vec<double, 2>(dx[1], dx[2]));
    }
  else
    {
    ds = dsb.Create(vtkm::Id3(nx, ny, nz),
       vtkm::Vec<double, 3>(x0[0], x0[1], x0[2]),
       vtkm::Vec<double, 3>(dx[0], dx[1], dx[2]));
    }

  // Wrap the values array in a vtkm array handle to avoid that the data is
  // copied when adding it a as point field to the dataset
  long n = nx*ny*nz;
  vtkm::cont::ArrayHandle<n_t> hData =
    vtkm::cont::make_ArrayHandle(data, n);

  // Add the data values to the dataset
  vtkm::cont::DataSetFieldAdd dsf;
  if (this->ScalarFieldAssociation == vtkDataObject::POINT)
    {
    dsf.AddPointField(ds, this->ScalarField, hData);
    }
  else if (this->ScalarFieldAssociation == vtkDataObject::CELL)
    {
    dsf.AddCellField(ds, this->ScalarField, hData);
    }
  else
    {
    SENSEI_ERROR("inavlid association")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int VTKmSmartContour::InternalsType::VTKmImage(vtkDataObject *dobj,
  vtkm::cont::DataSet &vtkmds)
{
  if (vtkImageData *image = dynamic_cast<vtkImageData*>(dobj))
    {
    int ext[6] = {0};
    double x0[3] = {0.0};
    double dx[3] = {0.0};
    image->GetExtent(ext);
    image->GetSpacing(dx);
    image->GetOrigin(x0);

    long nx = ext[1] - ext[0] + 1;
    long ny = ext[3] - ext[2] + 1;
    long nz = ext[5] - ext[4] + 1;

    vtkDataArray* array = this->GetScalarField(image);
    if (!array)
      {
      SENSEI_ERROR("failed to locate the scalar field")
      return -1;
      }

    // convert the array
    switch (array->GetDataType())
      {
      vtkTemplateMacro(
        VTK_TT *data = static_cast<vtkDataArrayTemplate<VTK_TT>*>(array)->GetPointer(0);
        if (this->VTKmImage(dx, x0, nx, ny, nz, data, vtkmds))
          {
          SENSEI_ERROR("Failed to convert array to VTKm")
          return -1;
          });
      default:
        SENSEI_ERROR("Invalid data type")
        return -1;
      }

    return 0;
    }

 SENSEI_ERROR("Not an image")
 return -1;
}





//-----------------------------------------------------------------------------
senseiNewMacro(VTKmSmartContour);

//----------------------------------------------------------------------------
VTKmSmartContour::VTKmSmartContour() : Internals(nullptr)
{
  this->Internals = new VTKmSmartContour::InternalsType;
}

//----------------------------------------------------------------------------
VTKmSmartContour::~VTKmSmartContour()
{
  delete this->Internals;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetCommunicator(MPI_Comm comm)
{
  this->Internals->Comm = comm;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetScalarField(const std::string &scalarField)
{
  this->Internals->ScalarField = scalarField;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetScalarFieldAssociation(int association)
{
  this->Internals->ScalarFieldAssociation = association;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetUseMarchingCubes(int useMarchingCubes)
{
  this->Internals->UseMarchingCubes = useMarchingCubes;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetUsePersistenceSorter(int usePersistenceSorter)
{
  this->Internals->UsePersistenceSorter = usePersistenceSorter;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetNumberOfLevels(int numberOfLevels)
{
  this->Internals->NumberOfLevels = numberOfLevels;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetContourType(int contourType)
{
  this->Internals->ContourType = contourType;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetEps(double eps)
{
  this->Internals->Eps = eps;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetSelectMethod(int selectMethod)
{
  this->Internals->SelectMethod = selectMethod;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetNumberOfComps(int numberOfComps)
{
  this->Internals->NumberOfComps = numberOfComps;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetCatalystScript(const std::string &catalystScript)
{
  this->Internals->CatalystScript = catalystScript;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetOutputDir(const std::string &outputFileName)
{
  this->Internals->OutputDir = outputFileName;
}

//-----------------------------------------------------------------------------
int VTKmSmartContour::Initialize()
{
#if defined(ENABLE_CATALYST)
  if (!this->Internals->CatalystScript.empty())
    {
    if (this->Internals->CatalystAdaptor)
      this->Internals->CatalystAdaptor->Delete();

    this->Internals->CatalystAdaptor = sensei::CatalystAnalysisAdaptor::New();

    this->Internals->CatalystAdaptor->AddPythonScriptPipeline(
      this->Internals->CatalystScript);
    }
#endif

  if (!this->Internals->OutputDir.empty())
    {
    if (this->Internals->IOAdaptor)
      this->Internals->IOAdaptor->Delete();

    this->Internals->IOAdaptor = sensei::VTKPosthocIO::New();
    this->Internals->IOAdaptor->SetCommunicator(this->Internals->Comm);
    this->Internals->IOAdaptor->SetOutputDir(this->Internals->OutputDir);
    this->Internals->IOAdaptor->SetMode(VTKPosthocIO::MODE_PARAVIEW);
    }

  return 0;
}

//-----------------------------------------------------------------------------
int VTKmSmartContour::Finalize()
{
  if (this->Internals->CatalystAdaptor)
    {
    this->Internals->CatalystAdaptor->Delete();
    this->Internals->CatalystAdaptor = nullptr;
    }

  if (this->Internals->IOAdaptor)
    {
    this->Internals->IOAdaptor->Finalize();
    this->Internals->IOAdaptor->Delete();
    this->Internals->IOAdaptor = nullptr;
    }

  return 0;
}

//-----------------------------------------------------------------------------
bool VTKmSmartContour::Execute(DataAdaptor* data)
{
  timer::MarkEvent mark("VTKmSmartContour::Execute");

  int rank = 0;
  int nRanks = 1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

  if (nRanks > 1)
    {
    SENSEI_ERROR("VTKmSmartContour is not MPI parallel "
      "and you are running " << nRanks << " MPI processes")
    return false;
    }

  // Convert the vtk data to VTKM
  // TODO -- sensei 2.0 api
  std::string meshName = "mesh";
  vtkDataObject* mesh = nullptr;
  if (data->GetMesh(meshName, false, mesh))
    {
    SENSEI_ERROR("Failed to get mesh")
    return false;
    }

  if (data->AddArray(mesh, meshName, this->Internals->ScalarFieldAssociation,
    this->Internals->ScalarField.c_str()))
    {
    SENSEI_ERROR("Faild to add "
      << this->Internals->AssociationStr() << " array \""
      << this->Internals->ScalarField << "\"")
    return false;
    }

  // VTK-m dataset variables
  vtkm::cont::DataSet inDataSet;
  if (vtkCompositeDataSet* cd = dynamic_cast<vtkCompositeDataSet*>(mesh))
    {
    vtkSmartPointer<vtkCompositeDataIterator> iter;
    iter.TakeReference(cd->NewIterator());
    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
      {
      // get the local mesh
      vtkDataObject *dobj = iter->GetCurrentDataObject();
      if (this->Internals->VTKmImage(dobj, inDataSet))
        {
        SENSEI_ERROR("failed to convert composite dataset node "
          << iter->GetCurrentFlatIndex())
        return false;
        }
      }
    }
  else
    {
    if (this->Internals->VTKmImage(mesh, inDataSet))
      {
      SENSEI_ERROR("failed to convert dataset")
      return false;
      }
    }

  ////////////////////////////////////////////
  // Build the contour tree
  ////////////////////////////////////////////
  // Output data set is pairs of saddle and peak vertex IDs
  vtkm::filter::Result result;

  // Convert the mesh of values into contour tree, pairs of vertex ids
  vtkm::filter::ContourTreePPP2 filter(this->Internals->UseMarchingCubes);
  result = filter.Execute(inDataSet,
                          this->Internals->ScalarField);

#ifdef DEBUG_PRINT
  // dump the contour tree  
  vtkm::cont::Field resultField =  result.GetField();
  vtkm::cont::ArrayHandle<vtkm::Pair<vtkm::Id, vtkm::Id> > saddlePeak;
  resultField.GetData().CopyTo(saddlePeak);
  std::cerr<<"Contour Tree"<<std::endl;
  std::cerr<<"============"<<std::endl;
  printEdgePairArray(saddlePeak);
  //contourTree.SortedArcPrint(mesh.sortOrder);
  //contourTree.PrintDotSuperStructure();
#endif

  ////////////////////////////////////////////
  // Compute the branch decomposition
  ////////////////////////////////////////////
  // compute the volume for each hyperarc and superarc
  IdArrayType superarcIntrinsicWeight;
  IdArrayType superarcDependentWeight;
  IdArrayType supernodeTransferWeight;
  IdArrayType hyperarcDependentWeight;

  ProcessContourTree::ComputeVolumeWeights<DeviceAdapter>(
    filter.GetContourTree(), filter.GetNumIterations(), superarcIntrinsicWeight,
    superarcDependentWeight, supernodeTransferWeight, hyperarcDependentWeight);

  // compute the branch decomposition by volume
  IdArrayType whichBranch;
  IdArrayType branchMinimum;
  IdArrayType branchMaximum;
  IdArrayType branchSaddle;
  IdArrayType branchParent;

  ProcessContourTree::ComputeVolumeBranchDecomposition<DeviceAdapter>(
    filter.GetContourTree(), superarcDependentWeight, superarcIntrinsicWeight,
    whichBranch, branchMinimum, branchMaximum, branchSaddle, branchParent);

  // create explicit representation of the branch decompostion from the array representation
  ArrayHandleType<double> vtkmValues = inDataSet.GetField(
    this->Internals->ScalarField).GetData().CastToTypeStorage<double,
    vtkm::cont::StorageTagBasic>();

  BranchType<double> *branchDecompostionRoot =
    ProcessContourTree::ComputeBranchDecomposition<double>(
      filter.GetContourTree().superparents, filter.GetContourTree().supernodes,
      whichBranch, branchMinimum, branchMaximum, branchSaddle, branchParent,
      filter.GetSortOrder(), vtkmValues);

  // Simplify the contour tree of the branch decompostion
  branchDecompostionRoot->simplifyToSize(this->Internals->NumberOfComps,
                                         this->Internals->UsePersistenceSorter);
#ifdef DEBUG_PRINT
  branchDecompostionRoot->print(std::cerr);
#endif

  // Compute the relevant iso-values
  switch(this->Internals->SelectMethod)
    {
    default:
    case 0:
      this->Internals->ContourValues.clear();
      branchDecompostionRoot->getRelevantValues(
        this->Internals->ContourType, this->Internals->Eps,
        this->Internals->ContourValues);
      break;
    case 1:
      {
      PiecewiseLinearFunction<double> plf;

      branchDecompostionRoot->accumulateIntervals(
        this->Internals->ContourType, this->Internals->Eps, plf);

      this->Internals->ContourValues =
        plf.nLargest(this->Internals->NumberOfLevels);
      }
      break;
    }

  std::sort(this->Internals->ContourValues.begin(),
    this->Internals->ContourValues.end());

  // Print the compute iso values
//#ifdef DEBUG_PRINT
  std::cerr << "Isovalues: ";
  for (double val : this->Internals->ContourValues)
    std::cerr << val << " ";
  std::cerr << std::endl;
//#endif

  // Remove any possible dublicates and retrieve only the unique iso values
  auto it = std::unique(this->Internals->ContourValues.begin(),
                        this->Internals->ContourValues.end());
  this->Internals->ContourValues.resize(std::distance(this->Internals->ContourValues.begin(), it));
  std::cerr << "Isovalues: ";
  for (double val : this->Internals->ContourValues)
      std::cerr << val << " ";
  std::cerr << std::endl;

  // compute contours
  using VTKDataAdaptorPtr = vtkSmartPointer<sensei::VTKDataAdaptor>;

  VTKDataAdaptorPtr contourGeometry = VTKDataAdaptorPtr::New();

#if defined(VTKM_CONTOUR)
  // TODO - use VTKm to compute the contours
  // TODO - convert the VTKm output into a VTK Object
#else
  vtkContourFilter *contour = vtkContourFilter::New();

  contour->SetComputeNormals(1);
  contour->SetComputeScalars(1);

  contour->SetInputArrayToProcess(0,0,0,
    this->Internals->ScalarFieldAssociation,
    this->Internals->ScalarField.c_str());

  int nVals = this->Internals->ContourValues.size();
  contour->SetNumberOfContours(nVals);
  for (int i = 0; i < nVals; ++i)
    contour->SetValue(i, this->Internals->ContourValues[i]);

  contour->SetInputDataObject(mesh);

  contour->Update();

  vtkMultiBlockDataSet *mbds = vtkMultiBlockDataSet::New();
  mbds->SetNumberOfBlocks(nRanks);
  mbds->SetBlock(rank, contour->GetOutputDataObject(0));

  contourGeometry->SetDataObject("mesh", mbds);
  contourGeometry->SetDataTimeStep(data->GetDataTimeStep());
  contourGeometry->SetDataTime(data->GetDataTime());

  mbds->Delete();
  contour->Delete();
#endif

  // render with catalyst
  if (this->Internals->CatalystAdaptor)
    {
    if (this->Internals->CatalystAdaptor->Execute(contourGeometry.GetPointer()))
      {
      SENSEI_ERROR("Catalyst failed")
      return -1;
      }
    }

  // write to disk
  if (this->Internals->IOAdaptor)
    {
    if (this->Internals->IOAdaptor->Execute(contourGeometry.GetPointer()))
      {
      SENSEI_ERROR("I/O failed")
      return -1;
      }
    }

  return true;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::PrintSelf(ostream& os, vtkIndent indent)
{
  this->AnalysisAdaptor::PrintSelf(os, indent);
}

}
