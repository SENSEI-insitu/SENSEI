#include "VTKmSmartContour.h"

#include "DataAdaptor.h"
#include "VTKDataAdaptor.h"
#include "CatalystAnalysisAdaptor.h"
#include "VTKPosthocIO.h"
#include "VTKUtils.h"
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
#include <vtkCellDataToPointData.h>
#include <vtkPointData.h>

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
#include <vtkm/filter/ContourTreeUniformAugmented.h>
#include <vtkm/worklet/contourtree_augmented/PrintVectors.h>

#include <vtkm/worklet/contourtree_augmented/ContourTree.h>
#include <vtkm/worklet/contourtree_augmented/ProcessContourTree.h>
#include <vtkm/worklet/contourtree_augmented/processcontourtree/Branch.h>
#include <vtkm/worklet/contourtree_augmented/processcontourtree/PiecewiseLinearFunction.h>

using namespace vtkm::worklet::contourtree_augmented;
using namespace vtkm::worklet::contourtree_augmented::process_contourtree_inc;

template <typename NT>
using ArrayHandleType = vtkm::cont::ArrayHandle<NT, vtkm::cont::StorageTagBasic>;

template <typename NT>
using BranchType = vtkm::worklet::contourtree_augmented::process_contourtree_inc::Branch<NT>;

#include <iostream>
using namespace std;

namespace sensei
{
class VTKmSmartContour::InternalsType
{
public:
  InternalsType() : MeshName(""), ArrayName(""),
    ArrayCentering(vtkDataObject::POINT), ArrayType(0), UseMarchingCubes(0),
    UsePersistenceSorter(1), NumberOfLevels(10), NumberOfComps(11),
    ContourType(0), Eps(1.0e-5), SelectMethod(0), CatalystScript(""),
    CatalystAdaptor(nullptr), OutputDir(""), IOAdaptor(nullptr) {}

public:
  std::string MeshName;
  std::string ArrayName;
  int ArrayCentering;
  int ArrayType;
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
bool XZPlane(int nx, int ny, int nz)
{
  return ((ny == 1) && (nx > 1) && (nz > 1));
}

// --------------------------------------------------------------------------
bool XYPlane(int nx, int ny, int nz)
{
  return ((nz == 1) && (nx > 1) && (ny > 1));
}

// --------------------------------------------------------------------------
bool YZPlane(int nx, int ny, int nz)
{
  return ((nx == 1) && (ny > 1) && (nz > 1));
}

// --------------------------------------------------------------------------
bool Planar(int nx, int ny, int nz)
{
  return XZPlane(nx, ny, nz) || XYPlane(nx, ny, nz) || YZPlane(nx, ny, nz);
}

// --------------------------------------------------------------------------
template<typename n_t>
int NewVTKmBlock(double *x0, double *dx, int nx, int ny, int nz,
  const std::string &name, int cen, n_t *data, vtkm::cont::DataSet &ds)
{
  // build the input dataset
  vtkm::cont::DataSetBuilderUniform dsb;

  if (XYPlane(nx, ny, nz))
    {
    ds = dsb.Create(vtkm::Id2(nx, ny),
      vtkm::Vec<double, 2>(x0[0], x0[1]),
      vtkm::Vec<double, 2>(dx[0], dx[1]));
    }
  else if (XZPlane(nx, ny, nz))
    {
    ds = dsb.Create(vtkm::Id2(nx, nz),
      vtkm::Vec<double, 2>(x0[0], x0[2]),
      vtkm::Vec<double, 2>(dx[0], dx[2]));
    }
  else if (YZPlane(nx, ny, nz))
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
  if (cen == vtkDataObject::POINT)
    {
    dsf.AddPointField(ds, name, hData);
    }
  else if (cen == vtkDataObject::CELL)
    {
    dsf.AddCellField(ds, name, hData);
    }
  else
    {
    SENSEI_ERROR("inavlid association")
    return -1;
    }

  return 0;
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
void VTKmSmartContour::SetMeshName(const std::string &name)
{
  this->Internals->MeshName = name;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetArrayName(const std::string &name)
{
  this->Internals->ArrayName = name;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetArrayCentering(int association)
{
  this->Internals->ArrayCentering = association;
}

//----------------------------------------------------------------------------
void VTKmSmartContour::SetArrayType(int type)
{
  this->Internals->ArrayType = type;
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

// Usable AlmostEqual function
bool equal(double a, double b)
{
    int maxUlps = 4;
    float A = a;
    float B = b;

    // Make sure maxUlps is non-negative and small enough that the
    // default NAN won't compare as equal to anything.
    //assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);
    int aInt = *(int*)&A;
    // Make aInt lexicographically ordered as a twos-complement int
    if (aInt < 0)
        aInt = 0x80000000 - aInt;
    // Make bInt lexicographically ordered as a twos-complement int
    int bInt = *(int*)&B;
    if (bInt < 0)
        bInt = 0x80000000 - bInt;
    int intDiff = abs(aInt - bInt);
    if (intDiff <= maxUlps)
        return true;
    return false;
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
  vtkDataObject* mesh = nullptr;
  if (data->GetMesh(this->Internals->MeshName, false, mesh))
    {
    SENSEI_ERROR("Failed to get mesh \"" << this->Internals->MeshName << "\"")
    return false;
    }

  if (this->Internals->ArrayType == VTKmSmartContour::ARRAY_TYPE_SCALAR)
    {
    if (data->AddArray(mesh, this->Internals->MeshName,
      this->Internals->ArrayCentering, this->Internals->ArrayName))
      {
      SENSEI_ERROR("Failed to add "
        << VTKUtils::GetAttributesName(this->Internals->ArrayCentering) << " array \""
        << this->Internals->ArrayName << "\"")
      return false;
      }
    }
  else if (this->Internals->ArrayType == VTKmSmartContour::ARRAY_TYPE_VECTOR)
    {
    // calculate magnitude of vector
    std::string vxname = this->Internals->ArrayName + 'x';
    std::string vyname = this->Internals->ArrayName + 'y';
    std::string vzname = this->Internals->ArrayName + 'z';

    if (data->AddArray(mesh, this->Internals->MeshName, this->Internals->ArrayCentering, vxname) ||
      data->AddArray(mesh, this->Internals->MeshName, this->Internals->ArrayCentering, vyname) ||
      data->AddArray(mesh, this->Internals->MeshName, this->Internals->ArrayCentering, vzname))
      {
      SENSEI_ERROR("Failed to add vector components for "
        << VTKUtils::GetAttributesName(this->Internals->ArrayCentering) << " array \""
        << this->Internals->ArrayName << "\"")
      return false;
      }

      VTKUtils::DatasetFunction l2norm = [&](vtkDataSet *ds) -> int
      {
        vtkFieldData *fd = ds->GetAttributesAsFieldData(this->Internals->ArrayCentering);
        vtkDataArray *dax = fd->GetArray(vxname.c_str());
        vtkDataArray *day = fd->GetArray(vyname.c_str());
        vtkDataArray *daz = fd->GetArray(vzname.c_str());

        vtkIdType n = dax->GetNumberOfTuples();

        vtkDataArray *mag = dax->NewInstance();
        mag->SetName(this->Internals->ArrayName.c_str());
        mag->SetNumberOfTuples(n);
        fd->AddArray(mag);
        mag->Delete();

        switch (dax->GetDataType())
          {
          vtkTemplateMacro(
            VTK_TT *vx = static_cast<vtkDataArrayTemplate<VTK_TT>*>(dax)->GetPointer(0);
            VTK_TT *vy = static_cast<vtkDataArrayTemplate<VTK_TT>*>(day)->GetPointer(0);
            VTK_TT *vz = static_cast<vtkDataArrayTemplate<VTK_TT>*>(daz)->GetPointer(0);
            VTK_TT *vm = static_cast<vtkDataArrayTemplate<VTK_TT>*>(mag)->GetPointer(0);
            for (vtkIdType i = 0; i < n; ++i)
              {
              vm[i] = sqrt(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
              }
            );
          }
        return 0;
      };

      if (VTKUtils::Apply(mesh, l2norm))
        {
        SENSEI_ERROR("Failed to compute magnitude of "
          << VTKUtils::GetAttributesName(this->Internals->ArrayCentering)
          << " vector \"" << this->Internals->ArrayName << "\"")
        return false;
        }
    }
  else
    {
    SENSEI_ERROR("Invalid array type")
    return false;
    }

  int cen = this->Internals->ArrayCentering;
  if (cen == vtkDataObject::CELL)
    {
    // convert to point centered data
    VTKUtils::DatasetFunction cellToPoint = [&](vtkDataSet *ds) -> int
    {
      vtkCellDataToPointData *cdpd = vtkCellDataToPointData::New();
      cdpd->SetPassCellData(0);
      cdpd->SetInputData(ds);
      cdpd->Update();
      vtkDataSet *dso = cdpd->GetOutput();
      ds->GetPointData()->ShallowCopy(dso->GetPointData());
      cdpd->Delete();
      return 0;
    };

    if (VTKUtils::Apply(mesh, cellToPoint))
      {
      SENSEI_ERROR("Failed to convert cell to point centering")
      return false;
      }

    cen = vtkDataObject::POINT;
    }

  // ocnvert to VTK-m dataset
  vtkm::cont::DataSet inDataSet;

  VTKUtils::DatasetFunction vtkToVtkm = [&](vtkDataSet *ds) -> int
  {
    vtkImageData *im = dynamic_cast<vtkImageData*>(ds);
    if (!im)
      {
      SENSEI_ERROR("Cartesian blocks are required")
      return -1;
      }

    int ext[6] = {0};
    double x0[3] = {0.0};
    double dx[3] = {0.0};

    im->GetExtent(ext);
    im->GetSpacing(dx);
    im->GetOrigin(x0);

    long nx = ext[1] - ext[0] + 1;
    long ny = ext[3] - ext[2] + 1;
    long nz = ext[5] - ext[4] + 1;

    vtkFieldData* fd = im->GetAttributesAsFieldData(cen);
    vtkDataArray *da = fd->GetArray(this->Internals->ArrayName.c_str());
    if (!da)
      {
      SENSEI_ERROR("failed to locate " << VTKUtils::GetAttributesName(cen)
        << " array \"" << this->Internals->ArrayName << "\"")
      return -1;
      }

    // convert the da
    switch (da->GetDataType())
      {
      vtkTemplateMacro(
        VTK_TT *pda = static_cast<vtkDataArrayTemplate<VTK_TT>*>(da)->GetPointer(0);
        if (NewVTKmBlock(x0, dx, nx, ny, nz, this->Internals->ArrayName, cen, pda, inDataSet))
          {
          SENSEI_ERROR("Failed to convert block")
          return -1;
          });
      default:
        SENSEI_ERROR("Invalid array type")
        return -1;
      }

    return 0;
  };

  if (VTKUtils::Apply(mesh, vtkToVtkm))
    {
    SENSEI_ERROR("Failed to construct a VTK-m data set")
    return false;
    }

  ////////////////////////////////////////////
  // Build the contour tree
  ////////////////////////////////////////////
  // Output data set is pairs of saddle and peak vertex IDs
  vtkm::cont::DataSet result;

  // Convert the mesh of values into contour tree, pairs of vertex ids
  vtkm::filter::ContourTreePPP2 filter(this->Internals->UseMarchingCubes);
  filter.SetActiveField(this->Internals->ArrayName);
  result = filter.Execute(inDataSet);

#ifdef DEBUG_PRINT
  // dump the contour tree
  vtkm::cont::Field resultField =  result.GetField();
  vtkm::cont::ArrayHandle<vtkm::Pair<vtkm::Id, vtkm::Id> > saddlePeak;
  resultField.GetData().CopyTo(saddlePeak);
  std::cerr<<"Contour Tree"<<std::endl;
  std::cerr<<"============"<<std::endl;
  printEdgePairArray(saddlePeak);
  contourTree.SortedArcPrint(mesh.sortOrder);
  contourTree.PrintDotSuperStructure();
#endif

  ////////////////////////////////////////////
  // Compute the branch decomposition
  ////////////////////////////////////////////
  // compute the volume for each hyperarc and superarc
  IdArrayType superarcIntrinsicWeight;
  IdArrayType superarcDependentWeight;
  IdArrayType supernodeTransferWeight;
  IdArrayType hyperarcDependentWeight;

  ProcessContourTree::ComputeVolumeWeights(
    filter.GetContourTree(), filter.GetNumIterations(), superarcIntrinsicWeight,
    superarcDependentWeight, supernodeTransferWeight, hyperarcDependentWeight);

  // compute the branch decomposition by volume
  IdArrayType whichBranch;
  IdArrayType branchMinimum;
  IdArrayType branchMaximum;
  IdArrayType branchSaddle;
  IdArrayType branchParent;

  ProcessContourTree::ComputeVolumeBranchDecomposition(
    filter.GetContourTree(), superarcDependentWeight, superarcIntrinsicWeight,
    whichBranch, branchMinimum, branchMaximum, branchSaddle, branchParent);

  // create explicit representation of the branch decompostion from the array representation
  ArrayHandleType<double> vtkmValues = inDataSet.GetField(
    this->Internals->ArrayName).GetData().CastToTypeStorage<double,
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
                        this->Internals->ContourValues.end(),
                        equal);


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
    cen, this->Internals->ArrayName.c_str());

  int nVals = this->Internals->ContourValues.size();
  contour->SetNumberOfContours(nVals);
  for (int i = 0; i < nVals; ++i)
    contour->SetValue(i, this->Internals->ContourValues[i]);

  contour->SetInputDataObject(mesh);

  contour->Update();

  vtkMultiBlockDataSet *mbds = vtkMultiBlockDataSet::New();
  mbds->SetNumberOfBlocks(nRanks);
  mbds->SetBlock(rank, contour->GetOutputDataObject(0));

  contourGeometry->SetDataObject(this->Internals->MeshName, mbds);
  contourGeometry->SetDataTimeStep(data->GetDataTimeStep());
  contourGeometry->SetDataTime(data->GetDataTime());

  mbds->Delete();
  contour->Delete();
#endif

  // render with catalyst
  if (this->Internals->CatalystAdaptor)
    {
    if (!this->Internals->CatalystAdaptor->Execute(contourGeometry.GetPointer()))
      {
      SENSEI_ERROR("Catalyst failed")
      return false;
      }
    }

  // write to disk
  if (this->Internals->IOAdaptor)
    {
    if (!this->Internals->IOAdaptor->Execute(contourGeometry.GetPointer()))
      {
      SENSEI_ERROR("I/O failed")
      return false;
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
